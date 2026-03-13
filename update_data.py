"""
fetch_data.py  —  Synoptic Map data pipeline
Extracted from Upper_air___Miller.ipynb

Fetches METAR, upper-air soundings, and vorticity tiles,
runs all analysis blocks, and writes JSON files to OUTPUT_DIR.

Output files (match what the push cell expects in the repo root):
  syn_ts_data.json        — per-timestamp METAR station SVGs
  syn_slp.json            — SLP/HL/T-Td/Temp contours per timestamp
  syn_ua.json             — upper-air contours (850/700/500/250 hPa)
  syn_ua_stns.json        — upper-air station popup/SVG data
  fire_zones_geojson.json — Alberta fire weather zones (static)
  vort_images.json        — vorticity tiles (base64 PNG, yellow/red filtered)
  conv_data.json          — convergence zones (SFC, 850, SFC trough)

Usage:
  python fetch_data.py                   # write to current directory
  python fetch_data.py --outdir /path/   # write to custom directory
"""

import argparse
import base64
import concurrent.futures
import csv
import io
import json
import math
import os
import re
import time
import warnings
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from urllib.parse import quote
from xml.etree import ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image as PILImage
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import (gaussian_filter, label,
                            maximum_filter, minimum_filter)
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  (mirrors Cell 1.5 defaults)
# ══════════════════════════════════════════════════════════════════════════
CSV_URL       = 'http://orangecore.net/met/wxchart/AP_location.csv'
METAR_API     = 'https://aviationweather.gov/api/data/metar'
COVERAGE      = 'standard'
INTERP_METHOD = 'rbf'
SLP_INTERVAL  = 4
GRID_N        = 240
RBF_SMOOTHING = 0.0
SIGMA_SMOOTH  = 1.0
SYMBOL_SCALE  = 28
FONT_SCALE    = 10
HL_NEIGHBORHOOD = 5
HL_MIN_DELTA    = 0.5
HL_SIGMA        = 1.0

# Smoothing parameters (Cell 10)
sigmaT700500      = 1.5
TMP_RBF_SMOOTHING = 0.05
TMP_SIGMA         = 1.0
TTD_RBF_SMOOTHING = 0.100
TTD_SIGMA         = 1.0

# Ridge/trough detection parameters (Cell 9)
M5_SIGMA      = 2.0
M5_PROMINENCE = 0.5
M5_WIDTH      = 2
M5_MAX_PEAKS  = 4
M5_MAX_DIST   = 4.0
M5_MIN_PTS    = 4

# Station model
_CR           = 0.14
FEATHER_ANGLE = 110
FEATHER_SIDE  = +1


# ══════════════════════════════════════════════════════════════════════════
#  UPPER AIR STATION LIST  (Cell 3)
# ══════════════════════════════════════════════════════════════════════════
UPPER_AIR_STATIONS = [
    {'id':'CYLT','name':'Alert',               'lat':82.50,'lon':-62.33,'wmo':'71082'},
    {'id':'CYEU','name':'Eureka',              'lat':79.98,'lon':-85.93,'wmo':'71917'},
    {'id':'YRB', 'name':'Resolute Bay',        'lat':74.72,'lon':-94.98,'wmo':'71924'},
    {'id':'YCB', 'name':'Clyde River',         'lat':70.49,'lon':-68.52,'wmo':'71925'},
    {'id':'CYFB','name':'Iqaluit',             'lat':63.75,'lon':-68.53,'wmo':'71909'},
    {'id':'YBK', 'name':'Baker Lake',          'lat':64.30,'lon':-96.00,'wmo':'71926'},
    {'id':'CYRK','name':'Rankin Inlet',        'lat':62.82,'lon':-92.12,'wmo':'71907'},
    {'id':'CYVP','name':'Kuujjuaq',            'lat':58.10,'lon':-68.42,'wmo':'71906'},
    {'id':'CYCO','name':'Coral Harbour',       'lat':64.19,'lon':-83.36,'wmo':'71815'},
    {'id':'WSE', 'name':'Goose Bay',           'lat':53.32,'lon':-60.42,'wmo':'71816'},
    {'id':'CYYT','name':"St. John's",          'lat':47.62,'lon':-52.74,'wmo':'71801'},
    {'id':'CYSJ','name':'Sable Island',        'lat':43.93,'lon':-60.02,'wmo':'71600'},
    {'id':'CYHZ','name':'Yarmouth',            'lat':43.83,'lon':-66.08,'wmo':'71603'},
    {'id':'CYBG','name':'Bagotville',          'lat':48.33,'lon':-70.99,'wmo':'71722'},
    {'id':'CYYY','name':'Mont Joli',           'lat':48.60,'lon':-68.22,'wmo':'71714'},
    {'id':'CYBR','name':'Brandon',             'lat':49.91,'lon':-99.95,'wmo':'71869'},
    {'id':'CYLA','name':'La Grande IV',        'lat':53.75,'lon':-73.67,'wmo':'71823'},
    {'id':'CYMO','name':'Moosonee',            'lat':51.29,'lon':-80.60,'wmo':'71836'},
    {'id':'CYTL','name':'Big Trout Lake',      'lat':53.82,'lon':-89.87,'wmo':'71845'},
    {'id':'CYYU','name':'Kapuskasing',         'lat':49.41,'lon':-82.47,'wmo':'71731'},
    {'id':'YYQ', 'name':'Churchill',           'lat':58.75,'lon':-94.07,'wmo':'71913'},
    {'id':'YQD', 'name':'The Pas',             'lat':53.97,'lon':-101.10,'wmo':'71867'},
    {'id':'CYXE','name':'Saskatoon',           'lat':52.17,'lon':-106.68,'wmo':'71866'},
    {'id':'XEG', 'name':'Edmonton Stony Plain','lat':53.55,'lon':-114.11,'wmo':'71119'},
    {'id':'YSM', 'name':'Fort Smith',          'lat':60.02,'lon':-111.96,'wmo':'71934'},
    {'id':'YVQ', 'name':'Norman Wells',        'lat':65.28,'lon':-126.80,'wmo':'71043'},
    {'id':'YXY', 'name':'Whitehorse',          'lat':60.72,'lon':-135.07,'wmo':'71964'},
    {'id':'YYE', 'name':'Fort Nelson',         'lat':58.84,'lon':-122.60,'wmo':'71945'},
    {'id':'YEV', 'name':'Inuvik',              'lat':68.30,'lon':-133.48,'wmo':'71957'},
    {'id':'CWSA','name':'Sachs Harbour',       'lat':71.99,'lon':-125.26,'wmo':'71038'},
    {'id':'CYZF','name':'Yellowknife',         'lat':62.46,'lon':-114.44,'wmo':'71936'},
    {'id':'CYVQ','name':'Norman Wells',        'lat':65.28,'lon':-126.80,'wmo':'71043'},
    {'id':'CYPA','name':'Prince Albert',       'lat':53.21,'lon':-105.67,'wmo':'71863'},
    {'id':'CYWG','name':'Winnipeg',            'lat':49.90,'lon':-97.24, 'wmo':'71852'},
    {'id':'CYQT','name':'Thunder Bay',         'lat':48.37,'lon':-89.32, 'wmo':'71734'},
    {'id':'CYSM','name':'Fort Smith',          'lat':60.02,'lon':-111.96,'wmo':'71934'},
    {'id':'CYXY','name':'Whitehorse',          'lat':60.72,'lon':-135.07,'wmo':'71964'},
    # US stations
    {'id':'OUN', 'name':'Norman OK',           'lat':35.18,'lon':-97.44, 'wmo':'72357'},
    {'id':'DNR', 'name':'Denver CO',           'lat':39.75,'lon':-104.87,'wmo':'72469'},
    {'id':'GJT', 'name':'Grand Junction CO',   'lat':39.12,'lon':-108.53,'wmo':'72476'},
    {'id':'SLC', 'name':'Salt Lake City UT',   'lat':40.77,'lon':-111.97,'wmo':'72572'},
    {'id':'BOI', 'name':'Boise ID',            'lat':43.57,'lon':-116.22,'wmo':'72681'},
    {'id':'GEG', 'name':'Spokane WA',          'lat':47.63,'lon':-117.63,'wmo':'72786'},
    {'id':'UIL', 'name':'Quillayute WA',       'lat':47.93,'lon':-124.55,'wmo':'72797'},
    {'id':'OTX', 'name':'Spokane WA',          'lat':47.68,'lon':-117.63,'wmo':'72786'},
    {'id':'REV', 'name':'Reno NV',             'lat':39.57,'lon':-119.80,'wmo':'72489'},
    {'id':'VBG', 'name':'Vandenberg CA',       'lat':34.73,'lon':-120.57,'wmo':'72393'},
    {'id':'NKX', 'name':'San Diego CA',        'lat':32.87,'lon':-117.15,'wmo':'72293'},
    {'id':'LKN', 'name':'Elko NV',             'lat':40.87,'lon':-115.73,'wmo':'72582'},
    {'id':'TFX', 'name':'Great Falls MT',      'lat':47.46,'lon':-111.38,'wmo':'72776'},
    {'id':'GGW', 'name':'Glasgow MT',          'lat':48.21,'lon':-106.62,'wmo':'72768'},
    {'id':'BIS', 'name':'Bismarck ND',         'lat':46.77,'lon':-100.75,'wmo':'72764'},
    {'id':'ABR', 'name':'Aberdeen SD',         'lat':45.45,'lon':-98.41, 'wmo':'72659'},
    {'id':'MPX', 'name':'Minneapolis MN',      'lat':44.85,'lon':-93.57, 'wmo':'72649'},
    {'id':'DVN', 'name':'Davenport IA',        'lat':41.61,'lon':-90.58, 'wmo':'74455'},
    {'id':'ILX', 'name':'Lincoln IL',          'lat':40.15,'lon':-89.34, 'wmo':'74560'},
    {'id':'APX', 'name':'Gaylord MI',          'lat':44.91,'lon':-84.72, 'wmo':'72634'},
    {'id':'DTX', 'name':'Detroit MI',          'lat':42.70,'lon':-83.47, 'wmo':'72632'},
    {'id':'BUF', 'name':'Buffalo NY',          'lat':42.93,'lon':-78.73, 'wmo':'72528'},
    {'id':'OKX', 'name':'New York NY',         'lat':40.87,'lon':-72.86, 'wmo':'72501'},
    {'id':'CHH', 'name':'Chatham MA',          'lat':41.67,'lon':-69.97, 'wmo':'72509'},
    {'id':'GYX', 'name':'Gray ME',             'lat':43.89,'lon':-70.25, 'wmo':'72606'},
    {'id':'IAD', 'name':'Sterling VA',         'lat':38.98,'lon':-77.47, 'wmo':'72403'},
    {'id':'RNK', 'name':'Blacksburg VA',       'lat':37.21,'lon':-80.41, 'wmo':'72318'},
    {'id':'GSO', 'name':'Greensboro NC',       'lat':36.10,'lon':-79.95, 'wmo':'72317'},
    {'id':'CHS', 'name':'Charleston SC',       'lat':32.90,'lon':-80.04, 'wmo':'72208'},
    {'id':'JAX', 'name':'Jacksonville FL',     'lat':30.49,'lon':-81.70, 'wmo':'72206'},
    {'id':'TBW', 'name':'Tampa FL',            'lat':27.71,'lon':-82.40, 'wmo':'72210'},
    {'id':'MFL', 'name':'Miami FL',            'lat':25.61,'lon':-80.41, 'wmo':'72202'},
    {'id':'BMX', 'name':'Birmingham AL',       'lat':33.17,'lon':-86.77, 'wmo':'72230'},
    {'id':'FFC', 'name':'Peachtree City GA',   'lat':33.36,'lon':-84.57, 'wmo':'72215'},
    {'id':'JAN', 'name':'Jackson MS',          'lat':32.32,'lon':-90.08, 'wmo':'72235'},
    {'id':'LCH', 'name':'Lake Charles LA',     'lat':30.12,'lon':-93.22, 'wmo':'72240'},
    {'id':'SHV', 'name':'Shreveport LA',       'lat':32.45,'lon':-93.82, 'wmo':'72248'},
    {'id':'FWD', 'name':'Fort Worth TX',       'lat':32.83,'lon':-97.30, 'wmo':'72249'},
    {'id':'MAF', 'name':'Midland TX',          'lat':31.94,'lon':-102.19,'wmo':'72265'},
    {'id':'DRT', 'name':'Del Rio TX',          'lat':29.37,'lon':-100.92,'wmo':'72261'},
    {'id':'BRO', 'name':'Brownsville TX',      'lat':25.92,'lon':-97.42, 'wmo':'72250'},
    {'id':'CRP', 'name':'Corpus Christi TX',   'lat':27.77,'lon':-97.51, 'wmo':'72251'},
    {'id':'EPZ', 'name':'El Paso TX',          'lat':31.87,'lon':-106.70,'wmo':'72364'},
    {'id':'ABQ', 'name':'Albuquerque NM',      'lat':35.04,'lon':-106.62,'wmo':'72365'},
    {'id':'TOP', 'name':'Topeka KS',           'lat':39.07,'lon':-95.62, 'wmo':'72456'},
    {'id':'DDC', 'name':'Dodge City KS',       'lat':37.76,'lon':-99.97, 'wmo':'72451'},
    {'id':'LBF', 'name':'North Platte NE',     'lat':41.13,'lon':-100.68,'wmo':'72562'},
    {'id':'PIH', 'name':'Pocatello ID',        'lat':42.91,'lon':-112.60,'wmo':'72578'},
    {'id':'MFR', 'name':'Medford OR',          'lat':42.37,'lon':-122.87,'wmo':'72694'},
    {'id':'SLE', 'name':'Salem OR',            'lat':44.91,'lon':-123.00,'wmo':'72694'},
    {'id':'OAK', 'name':'Oakland CA',          'lat':37.73,'lon':-122.22,'wmo':'72493'},
    {'id':'VEF', 'name':'Las Vegas NV',        'lat':36.05,'lon':-115.18,'wmo':'72390'},
]


# ══════════════════════════════════════════════════════════════════════════
#  STATION MODEL SVG  (Cell 2)
# ══════════════════════════════════════════════════════════════════════════
def cloud_circle_svg(cx, cy, R, oktas):
    lw = max(0.9, R * 0.13)
    s = []
    if oktas == 9:
        s.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="black" stroke="black" stroke-width="{lw}"/>')
        s.append(f'<line x1="{cx-R*.55:.2f}" y1="{cy-R*.55:.2f}" x2="{cx+R*.55:.2f}" y2="{cy+R*.55:.2f}" stroke="white" stroke-width="{lw*.85:.2f}"/>')
        s.append(f'<line x1="{cx+R*.55:.2f}" y1="{cy-R*.55:.2f}" x2="{cx-R*.55:.2f}" y2="{cy+R*.55:.2f}" stroke="white" stroke-width="{lw*.85:.2f}"/>')
        return ''.join(s)
    s.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="white" stroke="black" stroke-width="{lw}"/>')
    if oktas <= 0: return ''.join(s)
    if oktas >= 8:
        s.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="black" stroke="black" stroke-width="{lw}"/>')
        return ''.join(s)
    if oktas == 2:
        s.append(f'<path d="M{cx},{cy} L{cx},{cy-R:.2f} A{R:.2f},{R:.2f} 0 0,1 {cx+R:.2f},{cy} Z" fill="black"/>')
    elif oktas == 4:
        s.append(f'<path d="M{cx},{cy} L{cx},{cy-R:.2f} A{R:.2f},{R:.2f} 0 1,1 {cx},{cy+R:.2f} Z" fill="black"/>')
    elif oktas == 6:
        s.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="black" stroke="black" stroke-width="{lw}"/>')
        s.append(f'<path d="M{cx},{cy} L{cx-R:.2f},{cy} A{R:.2f},{R:.2f} 0 0,1 {cx},{cy-R:.2f} Z" fill="white"/>')
    s.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="none" stroke="black" stroke-width="{lw}"/>')
    return ''.join(s)


def wind_barb_svg(cx, cy, R, wind_dir, wind_spd, wind_gust, S):
    if wind_dir is None or wind_spd is None: return ''
    if wind_spd < 3:
        return (f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{R*1.5:.2f}" '
                f'fill="none" stroke="black" stroke-width="1"/>')
    sl = S * 1.0; blen = S * 0.30; blen_penn = S * 0.45
    bspc = S * 0.115; lw = max(0.9, S * 0.038)
    staff_base_y = -R; staff_tip_y = -(R + sl)
    fx_full = FEATHER_SIDE * blen; fx_half = FEATHER_SIDE * blen * 0.5
    tilt = math.tan(math.radians(FEATHER_ANGLE - 90)) * blen
    spd = int(round(wind_spd / 5.0)) * 5
    pn = spd // 50; spd -= pn * 50
    fu = spd // 10; spd -= fu * 10
    ha = spd // 5
    parts = [f'<line x1="0" y1="{staff_base_y:.2f}" x2="0" y2="{staff_tip_y:.2f}" '
             f'stroke="black" stroke-width="{lw:.2f}" stroke-linecap="round"/>']
    pos = 0.0
    if pn == 0 and fu == 0 and ha == 1:
        hy = staff_tip_y + 0.28 * sl
        parts.append(f'<line x1="0" y1="{hy:.2f}" x2="{fx_half:.2f}" y2="{hy - tilt*0.5:.2f}" '
                     f'stroke="black" stroke-width="{lw:.2f}" stroke-linecap="round"/>')
    else:
        for _ in range(pn):
            ay = staff_tip_y + pos; by2 = staff_tip_y + pos + bspc * 2
            pts_str = f'0,{ay:.2f} {fx_full:.2f},{ay - tilt:.2f} 0,{by2:.2f}'
            parts.append(f'<polygon points="{pts_str}" fill="black"/>'); pos += bspc * 1.5
        for _ in range(fu):
            fy = staff_tip_y + pos
            parts.append(f'<line x1="0" y1="{fy:.2f}" x2="{fx_full:.2f}" y2="{fy - tilt:.2f}" '
                         f'stroke="black" stroke-width="{lw:.2f}" stroke-linecap="round"/>'); pos += bspc
        for _ in range(ha):
            hy = staff_tip_y + pos
            parts.append(f'<line x1="0" y1="{hy:.2f}" x2="{fx_half:.2f}" y2="{hy - tilt*0.5:.2f}" '
                         f'stroke="black" stroke-width="{lw:.2f}" stroke-linecap="round"/>'); pos += bspc
    inner = ''.join(parts)
    return (f'<g transform="translate({cx:.2f},{cy:.2f}) rotate({wind_dir:.1f})">'
            f'{inner}</g>')


def pressure_tendency_svg(cx, cy, R, tendency, S):
    _map = {'rising':2,'falling':7,'steady':4,'rising_falling':0,
            'falling_rising':5,'rising_steady':1,'falling_steady':6}
    if isinstance(tendency, str): tendency = _map.get(tendency.lower(), None)
    if tendency is None: return ''
    lw = max(0.9, S * 0.042)
    ox = cx + R + S * 0.09 + S * 0.52
    slp_y = cy - R * 0.6 - 7; oy = slp_y + S * 0.55
    arm = S * 0.22; rise = S * 0.20
    def line(x1,y1,x2,y2):
        return (f'<line x1="{ox+x1:.2f}" y1="{oy+y1:.2f}" x2="{ox+x2:.2f}" y2="{oy+y2:.2f}" '
                f'stroke="black" stroke-width="{lw:.2f}" stroke-linecap="round" stroke-linejoin="round"/>')
    parts = []
    if   tendency == 2: parts.append(line(-arm, rise*0.5,  arm, -rise*0.5))
    elif tendency == 7: parts.append(line(-arm,-rise*0.5,  arm,  rise*0.5))
    elif tendency == 4: parts.append(line(-arm, 0,          arm,  0))
    elif tendency == 0: parts.append(line(-arm, rise*0.5,   0,  -rise*0.5)); parts.append(line(0,-rise*0.5,arm,rise*0.5))
    elif tendency == 5: parts.append(line(-arm,-rise*0.5,   0,   rise*0.5)); parts.append(line(0, rise*0.5,arm,-rise*0.5))
    elif tendency == 1: parts.append(line(-arm, rise*0.5,   0,  -rise*0.5)); parts.append(line(0,-rise*0.5,arm,-rise*0.5))
    elif tendency == 6: parts.append(line(-arm,-rise*0.5,   0,   rise*0.5)); parts.append(line(0, rise*0.5,arm, rise*0.5))
    return ''.join(parts)


def near_dewpoint_svg(cx, cy, R, temp, dew, is_surface=False):
    if not is_surface: return ''
    if temp is None or dew is None: return ''
    if (temp - dew) > 2: return ''
    radius = R * 3.2
    return (f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{radius:.2f}" '
            f'fill="#22cc44" fill-opacity="0.25" stroke="#22cc44" stroke-width="1.5" />')


def station_model_svg(d, S=34):
    PAD = S * 1.2; W = S * 3 + PAD * 2; H = S * 3 + PAD * 2
    cx = W / 2; cy = H / 2; R = S * _CR
    fs = FONT_SCALE or max(7, int(S * 0.36))
    off = R + S * 0.09
    parts = []
    parts.append(near_dewpoint_svg(cx, cy, R, d['temp'], d['dew'],
                                   is_surface=d.get('is_surface', False)))
    has_cloud = d.get('has_sky_obs', False)
    if has_cloud:
        parts.append(cloud_circle_svg(cx, cy, R, d['oktas']))
    else:
        th = R * 1.6
        parts.append(f'<polygon points="{cx:.2f},{cy-th:.2f} {cx-th:.2f},{cy+th*0.65:.2f} {cx+th:.2f},{cy+th*0.65:.2f}" fill="black" stroke="none"/>')
    parts.append(wind_barb_svg(cx, cy, R, d['wind_dir'], d['wind_spd'],
                               d.get('wind_gust', 0), S))

    def txt(x, y, text, anchor='end', bold=False, size=None):
        sz = size or fs; fw = 'bold' if bold else 'normal'
        return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
                f'dominant-baseline="central" font-size="{sz}px" font-weight="{fw}" '
                f'font-family="Courier New,monospace" fill="black" '
                f'paint-order="stroke" stroke="white" stroke-width="2" stroke-linejoin="round">'
                f'{text}</text>')

    if d['temp'] is not None:
        parts.append(txt(cx - off, cy - R * 0.6 - 6, str(d['temp'])))
    v = d['vis']
    vs = (str(int(v)) if v is not None and v >= 10 else
          str(int(v)) if v is not None and v % 1 == 0 else
          f'{v:.1f}' if v is not None else None)
    wx = ' '.join(x for x in [vs, d['weather'] or None] if x)
    if wx: parts.append(txt(cx - off - 4, cy, wx))
    if d['dew'] is not None:
        parts.append(txt(cx - off, cy + R * 0.6 + 6, str(d['dew'])))
    if d['slp_label']:
        parts.append(txt(cx + off, cy - R * 0.6 - 7, d['slp_label'], anchor='start'))
    tendency = d.get('tendency'); pressure_change = d.get('pressure_change')
    if tendency is not None:
        tend_y = cy - R * 0.6 - 7 + S * 0.55
        has_number = tendency != 'steady' and pressure_change is not None
        if has_number:
            pc_str = ('+' if pressure_change > 0 else '-' if pressure_change < 0 else '') + str(abs(pressure_change))
            parts.append(txt(cx + off, tend_y, pc_str, anchor='start'))
            parts.append(pressure_tendency_svg(cx + off, cy, R, tendency, S))
        else:
            parts.append(pressure_tendency_svg(cx + off - S * 0.52, cy, R, tendency, S))
    if d['lowest_sig'] and d['lowest_sig']['height'] <= 120:
        _cb = math.ceil(d['lowest_sig']['height'] / 10)
        parts.append(txt(cx, cy + R + fs * 0.9, str(_cb), anchor='middle'))
    _name_y = cy + R + fs * 0.9 + fs * 1.2
    parts.append(txt(cx, _name_y, d['icao'][-3:], anchor='middle'))
    return (f'<svg width="{W:.0f}" height="{H:.0f}" viewBox="0 0 {W:.2f} {H:.2f}" '
            f'xmlns="http://www.w3.org/2000/svg" style="overflow:visible">'
            + ''.join(parts) + '</svg>'), W, H


# ══════════════════════════════════════════════════════════════════════════
#  METAR  (Cells 5, 5b)
# ══════════════════════════════════════════════════════════════════════════
def load_stations(url, coverage='standard'):
    r = requests.get(url, timeout=15); r.raise_for_status()
    reader = csv.DictReader(io.StringIO(r.text))
    stations = {}
    for row in reader:
        icao = row.get('Code', '').strip()
        if not icao: continue
        if coverage == 'chart':
            chart_keys = [k for k in row.keys() if k.strip().lower() == 'chart']
            chart_val = row.get(chart_keys[0], '').strip() if chart_keys else ''
            if not chart_val: continue
        else:
            tier_map = {'essential': 1, 'standard': 2, 'all': 3}
            max_tier = tier_map.get(coverage, 2)
            tier = (1 if row.get('ESSENTIAL', '').strip() else
                    2 if row.get('STANDARD', '').strip() else 3)
            if tier > max_tier: continue
        try:
            stations[icao] = {
                'icao': icao, 'name': row.get('Name', '').strip(),
                'lat': float(row['Latitude']), 'lon': float(row['Longitude']),
                'tier': 0 if coverage == 'chart' else tier
            }
        except (ValueError, KeyError):
            pass
    return stations


def fetch_chunk(codes, hours=12, retries=3, backoff=2):
    for attempt in range(retries):
        try:
            params = {'ids': ','.join(codes), 'format': 'raw',
                      'hours': hours, 'mostRecent': 'false'}
            r = requests.get(METAR_API, params=params, timeout=30)
            if r.ok and r.text.strip(): return r.text, []
            time.sleep(backoff * (attempt + 1))
        except Exception:
            time.sleep(backoff * (attempt + 1))
    return '', codes


def fetch_all_metars(station_codes, chunk_size=25, max_workers=6, hours=12):
    chunks = [station_codes[i:i+chunk_size]
              for i in range(0, len(station_codes), chunk_size)]
    raw_parts = []; failed_codes = []; done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_chunk, c, hours): c for c in chunks}
        for fut in concurrent.futures.as_completed(futures):
            text, failed = fut.result()
            if text: raw_parts.append(text)
            if failed: failed_codes.extend(failed)
            done += len(futures[fut])
            print(f'  METAR {done}/{len(station_codes)}', end='\r')
    print()
    # Pass 2: retry silent-missing
    joined = '\n'.join(raw_parts)
    seen_icaos = set()
    for line in joined.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            icao = parts[1] if parts[0] in ('METAR', 'SPECI') else parts[0]
            seen_icaos.add(icao)
    silent_missing = [s for s in station_codes
                      if s not in seen_icaos and s not in failed_codes]
    if silent_missing:
        print(f'  Pass 2: retrying {len(silent_missing)} stations...')
        retry_chunks = [silent_missing[i:i+chunk_size]
                        for i in range(0, len(silent_missing), chunk_size)]
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures2 = {ex.submit(fetch_chunk, c, hours, 2, 1): c for c in retry_chunks}
                for fut in concurrent.futures.as_completed(futures2, timeout=30):
                    text, failed = fut.result()
                    if text: raw_parts.append(text)
        except concurrent.futures.TimeoutError:
            pass
    return '\n'.join(raw_parts), failed_codes


def parse_metar_line(line, stations):
    parts = line.strip().split()
    if len(parts) < 5: return None
    idx = 0
    if parts[0] == 'SPECI': return None
    if parts[0] == 'METAR': idx = 1
    if idx >= len(parts): return None
    icao = parts[idx]
    if icao not in stations: return None
    st = stations[icao]
    ts_raw = parts[idx+1] if idx+1 < len(parts) else ''
    if not re.match(r'^\d{6}Z$', ts_raw): return None
    day, hour, minute = int(ts_raw[0:2]), int(ts_raw[2:4]), int(ts_raw[4:6])
    if minute >= 35: hour = (hour + 1) % 24; minute = 0
    elif minute <= 25: minute = 0
    else: return None
    timestamp = f'{day:02d}{hour:02d}00Z'
    rest = [p for p in parts[idx+2:] if p not in ('MISG', 'MSIG')]
    # wind
    wind_dir = wind_spd = wind_gust = None
    for p in rest:
        m = re.match(r'^(\d{3})(\d{2,3})(?:G(\d{2,3}))?KT$', p)
        if m: wind_dir, wind_spd = int(m[1]), int(m[2]); wind_gust = int(m[3]) if m[3] else 0; break
        if re.match(r'^00000KT$', p): wind_dir=0; wind_spd=0; wind_gust=0; break
    # visibility
    vis = None
    for i, p in enumerate(rest):
        if p.endswith('SM'):
            whole = int(rest[i-1]) if i > 0 and rest[i-1].isdigit() else 0
            frac_str = p[:-2].lstrip('M')
            if '/' in frac_str:
                try: n, d2 = frac_str.split('/'); vis = whole + int(n) / int(d2)
                except: vis = 0.0
            else:
                try: vis = whole + float(frac_str) if frac_str else float(whole)
                except: vis = None
            break
    # clouds
    cloud_re = re.compile(r'^(FEW|SCT|BKN|OVC|VV)(\d{3})')
    clouds = []
    for p in rest:
        m = cloud_re.match(p)
        if m: clouds.append({'cover': m[1], 'height': int(m[2]), 'raw': p})
    clouds.sort(key=lambda c: c['height'])
    clr = any(p in ('CLR', 'SKC', 'CAVOK') for p in rest)
    has_sky_obs = clr or bool(clouds)
    cover_rank = {'CLR':0,'SKC':0,'FEW':2,'SCT':4,'BKN':6,'OVC':8,'VV':9}
    if clr or not clouds: oktas = 0
    else:
        sig = [c for c in clouds if c['cover'] in ('BKN','OVC','VV')]
        pool = sig if sig else clouds
        oktas = max(cover_rank.get(c['cover'], 0) for c in pool)
    sig_clouds = [c for c in clouds if c['cover'] in ('BKN','OVC','VV')]
    ceiling = sig_clouds[0]['height'] * 100 if sig_clouds else 99999
    lowest_sig = sig_clouds[0] if sig_clouds else None
    # temp/dew
    temp = dew = None
    for p in rest:
        m = re.match(r'^(M?\d{1,2})/(M?\d{1,2})$', p)
        if m:
            def td(s): return -(int(s[1:])) if s.startswith('M') else int(s)
            temp, dew = td(m[1]), td(m[2]); break
    # SLP
    slp = None
    for p in rest:
        m = re.match(r'^SLP(\d{3})$', p)
        if m: v = int(m[1]); slp = (900 + v/10) if v >= 500 else (1000 + v/10); break
    # weather
    wx_re = re.compile(
        r'^[+-]?(FZ|SH|BL|TS|MI|PR|BC|DR)?'
        r'(DZ|RA|SN|SG|IC|PL|GR|GS|UP|FG|BR|HZ|FU|VA|DU|SA|SQ|PO|FC|SS|DS){1,3}$')
    wx_parts = [p for p in rest if wx_re.match(p)
                and not re.match(r'^(RMK|SLP|AUTO|COR|AO\d)', p)]
    weather = ' '.join(wx_parts)
    # RH
    rh = None
    if temp is not None and dew is not None:
        a, b = 17.625, 243.04
        rh = round(100 * np.exp((a*dew/(b+dew)) - (a*temp/(b+temp))))
        rh = max(0, min(100, rh))
    t_td = round(temp - dew, 1) if temp is not None and dew is not None else None
    # flight category
    fc_vis = vis if vis is not None else 99
    if ceiling < 500 or fc_vis < 1: flt_cat = 'LIFR'
    elif ceiling < 1000 or fc_vis < 3: flt_cat = 'IFR'
    elif ceiling < 3000 or fc_vis < 5: flt_cat = 'MVFR'
    else: flt_cat = 'VFR'
    slp_label = f'{int(round(slp*10))%1000:03d}' if slp else ''
    return dict(
        icao=icao, name=st['name'], lat=st['lat'], lon=st['lon'],
        timestamp=timestamp, wind_dir=wind_dir, wind_spd=wind_spd,
        wind_gust=wind_gust, vis=vis, temp=temp, dew=dew, rh=rh,
        t_td=t_td, slp=slp, slp_label=slp_label, has_sky_obs=has_sky_obs,
        oktas=oktas, clouds=clouds, lowest_sig=lowest_sig, ceiling=ceiling,
        weather=weather, flt_cat=flt_cat, tendency=None, pressure_change=None
    )


def parse_all(text, stations):
    results = []; seen = set()
    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        if not re.match(r'^(METAR |SPECI |[A-Z]{4} \d{6}Z)', line): continue
        if line.startswith(('MISG','MSIG','SIGMET','AIRMET','PIREP','ATIS')): continue
        d = parse_metar_line(line, stations)
        if d:
            key = (d['icao'], d['timestamp'])
            if key not in seen: seen.add(key); results.append(d)
    return results


def classify_tendency_detailed(slp_series):
    if len(slp_series) < 2: return None, None
    slp_vals = [s for _, s in slp_series if s is not None]
    if len(slp_vals) < 2: return None, None
    first = slp_vals[0]; last = slp_vals[-1]; mid = slp_vals[len(slp_vals)//2]
    diff_total = last - first; diff_first = mid - first; diff_last = last - mid
    STEADY = 1.0; change = int(round(diff_total * 10))
    def sign(x): return 1 if x > STEADY else (-1 if x < -STEADY else 0)
    s1, s2 = sign(diff_first), sign(diff_last)
    if   s1==1  and s2==1:  return 'rising',         change
    elif s1==-1 and s2==-1: return 'falling',        change
    elif s1==0  and s2==0:  return 'steady',         change
    elif s1==1  and s2==-1: return 'rising_falling', change
    elif s1==-1 and s2==1:  return 'falling_rising', change
    elif s1==1  and s2==0:  return 'rising_steady',  change
    elif s1==-1 and s2==0:  return 'falling_steady', change
    elif s1==0  and s2==1:  return 'rising',         change
    elif s1==0  and s2==-1: return 'falling',        change
    else:                    return 'steady',         change


def compute_tendency(metar_records):
    station_slp_series = defaultdict(list)
    for d in metar_records:
        if d['slp'] is not None:
            station_slp_series[d['icao']].append((d['timestamp'], d['slp']))
    for icao in station_slp_series:
        station_slp_series[icao].sort(key=lambda x: x[0])
    for d in metar_records:
        series = [(ts, slp) for ts, slp in station_slp_series[d['icao']]
                  if ts <= d['timestamp']]
        if len(series) >= 2:
            tend, change = classify_tendency_detailed(series)
            d['tendency'] = tend; d['pressure_change'] = change


def fetch_metar():
    print('── METAR ─────────────────────────────────────────────')
    STATIONS = load_stations(CSV_URL, COVERAGE)
    print(f'  Loaded {len(STATIONS)} stations ({COVERAGE})')
    codes = list(STATIONS.keys())
    raw_metar_text, _ = fetch_all_metars(codes, hours=12)
    metar_records = parse_all(raw_metar_text, STATIONS)
    compute_tendency(metar_records)
    print(f'  ✓ {len(metar_records)} METAR records parsed')
    return metar_records


# ══════════════════════════════════════════════════════════════════════════
#  UPPER AIR  (Cells 6, 8)
# ══════════════════════════════════════════════════════════════════════════
WYOMING_BASE = 'https://weather.uwyo.edu/wsgi/sounding'
UA_HOURS = [0, 12]
UA_WORKERS = 12
UA_TIMEOUT = 12


def get_sounding_dt(hour, lookback_days=0):
    now = datetime.now(timezone.utc)
    dt = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if dt > now: dt -= timedelta(days=1)
    return dt - timedelta(days=lookback_days)


def fetch_raw_sounding(stn, hour, lookback=0):
    dt = get_sounding_dt(hour, lookback)
    dt_str = f'{dt.strftime("%Y-%m-%d")} {hour}:00:00'
    url = f'{WYOMING_BASE}?datetime={quote(dt_str)}&id={stn["wmo"]}&src=BUFR&type=TEXT:LIST'
    try:
        r = requests.get(url, timeout=UA_TIMEOUT)
        if not r.ok or len(r.text) < 300: return None
        txt = r.text
        if "Can't get" in txt or 'No Observation' in txt or 'ERROR' in txt: return None
        return txt
    except Exception:
        return None


def parse_sounding(html, stn, hour, lookback=0):
    dt = get_sounding_dt(hour, lookback)
    obs_name = obs_lat = obs_lon = None
    lat_m = re.search(r'Latitude:\s*([-\d.]+)', html, re.I)
    lon_m = re.search(r'Longitude:\s*([-\d.]+)', html, re.I)
    if lat_m: obs_lat = float(lat_m.group(1))
    if lon_m: obs_lon = float(lon_m.group(1))
    h2 = re.search(r'<h2[^>]*>([\s\S]*?)</h2>', html, re.I)
    if h2:
        txt = re.sub(r'<[^>]+>', ' ', h2.group(1))
        lines = [l.strip() for l in txt.split('\n') if l.strip()]
        if len(lines) >= 2: obs_name = lines[-1]
    pre = re.search(r'<pre>([\s\S]*?)</pre>', html, re.I)
    if not pre: return []
    rows = []
    for line in pre.group(1).split('\n'):
        cols = line.strip().split()
        if len(cols) < 7: continue
        try: pres = float(cols[0])
        except ValueError: continue
        if not (100 <= pres <= 1100): continue
        def fv(i):
            try: v = float(cols[i]); return None if abs(v) > 9000 else v
            except: return None
        rows.append({
            'icao': stn['id'], 'wmo': stn['wmo'],
            'stn_name': obs_name or stn['name'],
            'lat': obs_lat or stn['lat'], 'lon': obs_lon or stn['lon'],
            'valid_time': dt.strftime('%Y-%m-%d') + f' {hour:02d}Z',
            'hour': hour, 'PRES': pres,
            'HGHT': fv(1), 'TEMP': fv(2), 'DWPT': fv(3),
            'RELH': fv(4), 'MIXR': fv(5), 'DRCT': fv(6), 'SPED': fv(7),
            'THTA': fv(8) if len(cols) > 8 else None,
            'THTE': fv(9) if len(cols) > 9 else None,
            'THTV': fv(10) if len(cols) > 10 else None,
        })
    return rows


def fetch_upper_air():
    print('── Upper Air ─────────────────────────────────────────')
    tasks = [(s, h) for s in UPPER_AIR_STATIONS for h in UA_HOURS]
    all_rows = []; failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=UA_WORKERS) as ex:
        fmap = {ex.submit(fetch_raw_sounding, s, h): (s, h) for s, h in tasks}
        done = 0
        for fut in concurrent.futures.as_completed(fmap):
            s, h = fmap[fut]; done += 1
            print(f'  UA {done}/{len(tasks)}', end='\r')
            html = fut.result()
            if html:
                rows = parse_sounding(html, s, h)
                if rows: all_rows.extend(rows)
                else: failed.append((s, h))
            else: failed.append((s, h))
    print()
    # Pass 2: lookback 1 day
    if failed:
        print(f'  Pass 2: {len(failed)} retry with lookback...')
        retry = failed[:]
        failed = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=UA_WORKERS) as ex:
            fmap2 = {ex.submit(fetch_raw_sounding, s, h, 1): (s, h) for s, h in retry}
            for fut in concurrent.futures.as_completed(fmap2):
                s, h = fmap2[fut]
                html = fut.result()
                if html:
                    rows = parse_sounding(html, s, h, lookback=1)
                    if rows: all_rows.extend(rows); continue
                failed.append((s, h))
    print(f'  ✓ {len(all_rows)} sounding level rows')
    # Build DataFrame
    ua_raw_df = pd.DataFrame(all_rows, columns=[
        'icao','wmo','stn_name','lat','lon','valid_time','hour',
        'PRES','HGHT','TEMP','DWPT','RELH','MIXR','DRCT','SPED','THTA','THTE','THTV'])
    # Keep latest valid_time per hour
    _latest = ua_raw_df.groupby('hour')['valid_time'].max()
    _mask = ua_raw_df['valid_time'] == ua_raw_df['hour'].map(_latest)
    ua_raw_df = ua_raw_df[_mask].reset_index(drop=True)
    # Build standard-level summary
    STANDARD_LEVELS = [850, 700, 500, 250]
    LEVEL_TOL = 25
    FIELDS = ['PRES','HGHT','TEMP','DWPT','RELH','MIXR','DRCT','SPED','THTA','THTE','THTV']
    def find_closest_level(group_df, target_p):
        sub = group_df.copy(); sub['_dist'] = (sub['PRES'] - target_p).abs()
        sub = sub[sub['_dist'] <= LEVEL_TOL]
        if sub.empty: return {f: None for f in FIELDS}
        best = sub.loc[sub['_dist'].idxmin()]
        return {f: best[f] if not pd.isna(best[f]) else None for f in FIELDS}
    summary_rows = []
    for (icao, hour), grp in ua_raw_df.groupby(['icao', 'hour'], sort=False):
        grp = grp.sort_values('PRES', ascending=False).reset_index(drop=True)
        meta = grp.iloc[0]
        base = {'icao': icao, 'wmo': meta['wmo'], 'stn_name': meta['stn_name'],
                'lat': meta['lat'], 'lon': meta['lon'],
                'valid_time': meta['valid_time'], 'hour': hour}
        for lvl in STANDARD_LEVELS:
            vals = find_closest_level(grp, lvl)
            for f in FIELDS: base[f'{f}_{lvl}'] = vals[f]
        summary_rows.append(base)
    ua_summary_df = pd.DataFrame(summary_rows).sort_values(['icao','hour']).reset_index(drop=True)
    _ua_times = ua_summary_df.groupby('hour')['valid_time'].max().reset_index()
    ua_date_map = {}
    for _, row in _ua_times.iterrows():
        ua_date_map[str(int(row['hour']))] = str(row['valid_time'])
        print(f'  UA {row["valid_time"]}  {int(row["hour"])}Z')
    print(f'  ✓ UA summary: {len(ua_summary_df)} rows')
    return ua_summary_df, ua_date_map


# ══════════════════════════════════════════════════════════════════════════
#  GRID / CONTOUR HELPERS  (Cells 11, 12)
# ══════════════════════════════════════════════════════════════════════════
def build_grid(records, field, method='rbf', N=220, pad=1.5,
               rbf_smoothing=0.3, sigma=3.0):
    pts = [(d['lat'], d['lon'], d[field]) for d in records if d.get(field) is not None]
    if len(pts) < 8: return None, None, None, None, None
    _seen = {}
    for la, lo, v in pts:
        key = (round(la, 2), round(lo, 2))
        _seen.setdefault(key, []).append(v)
    pts = [(k[0], k[1], float(np.mean(vs))) for k, vs in _seen.items()]
    if len(pts) < 8: return None, None, None, None, None
    lats = np.array([p[0] for p in pts]); lons = np.array([p[1] for p in pts])
    vals = np.array([p[2] for p in pts], dtype=float)
    lon_vec = np.linspace(lons.min()-pad, lons.max()+pad, N)
    lat_vec = np.linspace(lats.min()-pad, lats.max()+pad, N)
    glon, glat = np.meshgrid(lon_vec, lat_vec)
    obs_xy = np.column_stack([lons, lats])
    try:
        rbf = RBFInterpolator(obs_xy, vals, kernel='thin_plate_spline',
                              smoothing=max(rbf_smoothing * len(pts), 1e-6))
    except np.linalg.LinAlgError:
        rbf = RBFInterpolator(obs_xy, vals, kernel='linear',
                              smoothing=max(rbf_smoothing * len(pts), 1.0))
    grid = rbf(np.column_stack([glon.ravel(), glat.ravel()])).reshape(N, N)
    if sigma > 0: grid = gaussian_filter(grid, sigma=sigma)
    return grid, lon_vec, lat_vec, lons, lats


def find_hl_centers(grid, lon_vec, lat_vec, metar_records,
                    neighborhood=20, min_delta=2.0):
    sg = gaussian_filter(grid, sigma=HL_SIGMA)
    max_f = maximum_filter(sg, size=neighborhood)
    min_f = minimum_filter(sg, size=neighborhood)
    is_max = (sg == max_f) & (sg - min_f > min_delta)
    is_min = (sg == min_f) & (max_f - sg > min_delta)
    centers = []
    for typ, mask in [('H', is_max), ('L', is_min)]:
        lbl, n = label(mask)
        for i in range(1, n+1):
            rows, cols = np.where(lbl == i)
            best = np.argmax(sg[rows,cols]) if typ=='H' else np.argmin(sg[rows,cols])
            r, c = rows[best], cols[best]
            if r < neighborhood or r > len(lat_vec)-neighborhood: continue
            if c < neighborhood or c > len(lon_vec)-neighborhood: continue
            _grid_val = float(grid[r, c]); _thresh = SLP_INTERVAL
            def _grid_at(sta_lat, sta_lon, _lv=lon_vec, _ltv=lat_vec, _g=grid):
                _ri = int(round((sta_lat-_ltv[0])/(_ltv[-1]-_ltv[0])*(len(_ltv)-1)))
                _ci = int(round((sta_lon-_lv[0])/(_lv[-1]-_lv[0])*(len(_lv)-1)))
                return float(_g[max(0,min(len(_ltv)-1,_ri)), max(0,min(len(_lv)-1,_ci))])
            if typ == 'H':
                _inside = [d['slp'] for d in metar_records
                           if d['slp'] is not None and _grid_at(d['lat'],d['lon']) >= _grid_val - _thresh]
                _val = (math.floor(max(_inside)) + 1) if _inside else (math.floor(_grid_val) + 1)
            else:
                _inside = [d['slp'] for d in metar_records
                           if d['slp'] is not None and _grid_at(d['lat'],d['lon']) <= _grid_val + _thresh]
                _val = (math.ceil(min(_inside)) - 1) if _inside else (math.ceil(_grid_val) - 1)
            centers.append(dict(type=typ, lat=lat_vec[r], lon=lon_vec[c], val=float(_val)))
    return centers


# ══════════════════════════════════════════════════════════════════════════
#  RIDGE / TROUGH DETECTION  (Cell 9)
# ══════════════════════════════════════════════════════════════════════════
def _rbf_grid_ua(la, lo, va, N=180, pad=1.5, sigma=1.5):
    seen = {}
    for a, o, v in zip(la, lo, va):
        seen.setdefault((round(a,2), round(o,2)), []).append(v)
    pts = [(k[0], k[1], float(np.mean(vs))) for k, vs in seen.items()]
    if len(pts) < 8: return None, None, None
    la2 = np.array([p[0] for p in pts]); lo2 = np.array([p[1] for p in pts])
    va2 = np.array([p[2] for p in pts])
    lv = np.linspace(lo2.min()-pad, lo2.max()+pad, N)
    ltv = np.linspace(la2.min()-pad, la2.max()+pad, N)
    GL, GLA = np.meshgrid(lv, ltv)
    try:
        rbf = RBFInterpolator(np.column_stack([lo2, la2]), va2,
                              kernel='thin_plate_spline',
                              smoothing=max(0.3*len(pts), 1e-6))
        grid = rbf(np.column_stack([GL.ravel(), GLA.ravel()])).reshape(N, N)
    except Exception:
        return None, None, None
    return gaussian_filter(grid, sigma=sigma), lv, ltv


def _pts_to_segs(pts, max_dist=M5_MAX_DIST, min_pts=M5_MIN_PTS):
    if len(pts) < min_pts: return []
    arr = np.array(pts)
    dedup = {}
    for r in arr:
        k = (round(r[0]*2)/2, round(r[1]*2)/2)
        if k not in dedup or r[2] > dedup[k][2]: dedup[k] = r
    arr = np.array(list(dedup.values()))
    if len(arr) < min_pts: return []
    tree = cKDTree(arr[:,:2]); pairs = tree.query_pairs(max_dist)
    parent = list(range(len(arr)))
    def _find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in pairs:
        ra, rb = _find(a), _find(b)
        if ra != rb: parent[ra] = rb
    clusters = {}
    for i in range(len(arr)): clusters.setdefault(_find(i), []).append(i)
    segs = []
    for idxs in clusters.values():
        if len(idxs) < min_pts: continue
        sp = sorted([(arr[i][0], arr[i][1]) for i in idxs], key=lambda p: p[0])
        sub, cur = [], [sp[0]]
        for k in range(1, len(sp)):
            if (abs(sp[k][0]-sp[k-1][0]) > max_dist*1.5 or
                abs(sp[k][1]-sp[k-1][1]) > max_dist*1.5):
                if len(cur) >= min_pts: sub.append(cur)
                cur = []
            cur.append(sp[k])
        if len(cur) >= min_pts: sub.append(cur)
        segs.extend(sub)
    return segs


def detect_m5(T, lv, ltv):
    Ts = gaussian_filter(T, sigma=M5_SIGMA)
    ridge_pts, trough_pts = [], []
    for j, lat in enumerate(ltv):
        row = Ts[j, :]
        pk, pr = find_peaks(row, prominence=M5_PROMINENCE, width=M5_WIDTH)
        if len(pk):
            top = np.argsort(pr['prominences'])[-M5_MAX_PEAKS:]
            for idx in pk[top]:
                prom = pr['prominences'][np.where(pk == idx)[0][0]]
                ridge_pts.append((lat, lv[idx], float(prom)))
        tr, tpr = find_peaks(-row, prominence=M5_PROMINENCE, width=M5_WIDTH)
        if len(tr):
            top = np.argsort(tpr['prominences'])[-M5_MAX_PEAKS:]
            for idx in tr[top]:
                prom = tpr['prominences'][np.where(tr == idx)[0][0]]
                trough_pts.append((lat, lv[idx], float(prom)))
    return _pts_to_segs(ridge_pts), _pts_to_segs(trough_pts)


def detect_ridges_troughs(ua_summary_df):
    print('── Ridge/Trough Detection ────────────────────────────')
    results = {}
    for _hr in sorted(ua_summary_df['hour'].unique()):
        _df_hr = ua_summary_df[ua_summary_df['hour'] == _hr].copy()
        hr_key = int(_hr)
        for lvl, col in [(850,'TEMP_850'), (700,'TEMP_700'), (500,'TEMP_500')]:
            rows = _df_hr.dropna(subset=[col, 'lat', 'lon'])
            if len(rows) < 8:
                results[f'ridge_{lvl}_{hr_key:02d}'] = []
                results[f'trough_{lvl}_{hr_key:02d}'] = []
                continue
            T, lv, ltv = _rbf_grid_ua(rows['lat'].values, rows['lon'].values,
                                       rows[col].values, sigma=1.5)
            if T is None:
                results[f'ridge_{lvl}_{hr_key:02d}'] = []
                results[f'trough_{lvl}_{hr_key:02d}'] = []
                continue
            r_segs, t_segs = detect_m5(T, lv, ltv)
            results[f'ridge_{lvl}_{hr_key:02d}'] = r_segs
            results[f'trough_{lvl}_{hr_key:02d}'] = t_segs
            print(f'  {lvl}hPa {hr_key:02d}Z  ridge:{len(r_segs)}  trough:{len(t_segs)}')
    return results


# ══════════════════════════════════════════════════════════════════════════
#  SLP / CONTOURS  (Cells 13, 14)
# ══════════════════════════════════════════════════════════════════════════
def build_slp_data(metar_records):
    print('── SLP Contours ──────────────────────────────────────')
    _ts_all = sorted(set(d['timestamp'] for d in metar_records if d['timestamp']))
    _ts_slp = {}
    for _ts in _ts_all:
        _recs = [d for d in metar_records if d['timestamp'] == _ts]
        _grid, _lv, _ltv, _, _ = build_grid(_recs, 'slp', method=INTERP_METHOD,
                                             N=GRID_N, rbf_smoothing=RBF_SMOOTHING,
                                             sigma=SIGMA_SMOOTH)
        if _grid is None:
            _ts_slp[_ts] = {'contours': [], 'hl': []}
            continue
        _glon, _glat = np.meshgrid(_lv, _ltv)
        _slp_min = np.floor(_grid.min() / SLP_INTERVAL) * SLP_INTERVAL
        _slp_max = np.ceil(_grid.max() / SLP_INTERVAL) * SLP_INTERVAL
        _levels = np.arange(_slp_min, _slp_max + SLP_INTERVAL, SLP_INTERVAL)
        _fig, _ax = plt.subplots(figsize=(1, 1))
        _cs = _ax.contour(_glon, _glat, _grid, levels=_levels)
        plt.close(_fig)
        _contours = []
        for _li, _lvl in enumerate(_cs.levels):
            _is_major = (int(_lvl) % 20 == 0)
            _weight = 2.5 if _is_major else (1.4 if int(_lvl) % 8 == 0 else 0.7)
            _opacity = 0.95 if _is_major else (0.65 if int(_lvl) % 8 == 0 else 0.40)
            for _coords in _cs.allsegs[_li]:
                if len(_coords) < 2: continue
                _mid = _coords[len(_coords) // 2]
                _contours.append({'level': float(_lvl), 'weight': _weight,
                                   'opacity': _opacity,
                                   'coords': [[float(c[0]), float(c[1])] for c in _coords],
                                   'label_lon': float(_mid[0]), 'label_lat': float(_mid[1])})
        _hl = find_hl_centers(_grid, _lv, _ltv, _recs,
                              neighborhood=HL_NEIGHBORHOOD, min_delta=HL_MIN_DELTA)
        # T-Td contours
        _ttd_grid, _ttd_lv, _ttd_ltv, _, _ = build_grid(_recs, 't_td', method='rbf',
                                                          N=GRID_N, rbf_smoothing=TTD_RBF_SMOOTHING,
                                                          sigma=TTD_SIGMA)
        _ttd_contours = []
        if _ttd_grid is not None:
            _TTD_INTERVAL = 2.0
            _ttd_levels = np.arange(np.floor(_ttd_grid.min()/_TTD_INTERVAL)*_TTD_INTERVAL,
                                    np.ceil(_ttd_grid.max()/_TTD_INTERVAL)*_TTD_INTERVAL+_TTD_INTERVAL,
                                    _TTD_INTERVAL)
            _glon_t, _glat_t = np.meshgrid(_ttd_lv, _ttd_ltv)
            _fig_t, _ax_t = plt.subplots(figsize=(1,1))
            _cs_t = _ax_t.contour(_glon_t, _glat_t, _ttd_grid, levels=_ttd_levels)
            plt.close(_fig_t)
            for _li_t, _lvl_t in enumerate(_cs_t.levels):
                for _coords_t in _cs_t.allsegs[_li_t]:
                    if len(_coords_t) < 2: continue
                    _mid_t = _coords_t[len(_coords_t)//2]
                    _ttd_contours.append({'level': float(_lvl_t),
                                          'coords': [[float(c[0]),float(c[1])] for c in _coords_t],
                                          'label_lon': float(_mid_t[0]), 'label_lat': float(_mid_t[1])})
        # Temperature contours
        _tmp_grid, _tmp_lv, _tmp_ltv, _, _ = build_grid(_recs, 'temp', method='rbf',
                                                         N=GRID_N, rbf_smoothing=TMP_RBF_SMOOTHING,
                                                         sigma=TMP_SIGMA)
        _tmp_contours = []
        if _tmp_grid is not None:
            _TMP_INTERVAL = 2.0
            _tmp_levels = np.arange(np.floor(_tmp_grid.min()/_TMP_INTERVAL)*_TMP_INTERVAL,
                                    np.ceil(_tmp_grid.max()/_TMP_INTERVAL)*_TMP_INTERVAL+_TMP_INTERVAL,
                                    _TMP_INTERVAL)
            _glon_m, _glat_m = np.meshgrid(_tmp_lv, _tmp_ltv)
            _fig_m, _ax_m = plt.subplots(figsize=(1,1))
            _cs_m = _ax_m.contour(_glon_m, _glat_m, _tmp_grid, levels=_tmp_levels)
            plt.close(_fig_m)
            for _li_m, _lvl_m in enumerate(_cs_m.levels):
                for _coords_m in _cs_m.allsegs[_li_m]:
                    if len(_coords_m) < 2: continue
                    _mid_m = _coords_m[len(_coords_m)//2]
                    _tmp_contours.append({'level': float(_lvl_m),
                                          'coords': [[float(c[0]),float(c[1])] for c in _coords_m],
                                          'label_lon': float(_mid_m[0]), 'label_lat': float(_mid_m[1])})
        _ts_slp[_ts] = {'contours': _contours, 'hl': _hl,
                        'ttd_contours': _ttd_contours, 'tmp_contours': _tmp_contours}
        print(f'  {_ts}: {len(_contours)} SLP, {len(_ttd_contours)} T-Td, '
              f'{len(_tmp_contours)} Temp, '
              f'{sum(1 for x in _hl if x["type"]=="H")}H '
              f'{sum(1 for x in _hl if x["type"]=="L")}L')
    return _ts_slp


# ══════════════════════════════════════════════════════════════════════════
#  UPPER-AIR CONTOURS  (Cell 14)
# ══════════════════════════════════════════════════════════════════════════
_NORMALS_850 = [
    (1,1,-6,-8),(1,4,-8,-10),(1,15,-6,-8),(1,18,-4,-6),(1,24,-6,-8),(1,31,-8,-10),
    (2,4,-6,-8),(3,9,-4,-6),(3,12,-2,-4),(4,3,0,-2),(4,5,2,0),(4,8,4,2),
    (5,2,6,4),(5,11,8,6),(5,23,10,8),(6,1,12,10),(6,27,14,12),
    (8,24,12,10),(9,5,10,8),(9,17,8,6),(10,1,6,4),(10,11,4,2),(10,25,2,0),
    (10,29,0,-2),(11,8,-2,-4),(11,12,0,-2),(11,17,-2,-4),(11,26,-4,-6),(12,2,-6,-8),
]
_NORMALS_500 = [
    (1,1,-28,-30),(1,15,-26,-28),(1,24,-28,-30),(2,23,-30,-32),(2,27,-28,-30),
    (3,9,-26,-28),(3,12,-28,-30),(4,5,-26,-28),(4,19,-24,-26),(4,27,-22,-24),
    (5,11,-20,-22),(5,23,-18,-20),(6,13,-16,-18),(6,27,-14,-16),(8,4,-12,-14),
    (8,11,-14,-16),(8,31,-16,-18),(9,17,-18,-20),(10,3,-20,-22),(10,21,-22,-24),
    (10,29,-24,-26),(11,17,-26,-28),(12,2,-28,-30),
]
_UA_TEMP_BANDS_BASE = [
    (999,16,'#ffffff'),(16,14,'#8b4513'),(14,12,'#ffffff'),(12,10,'#ffc0cb'),
    (10,8,'#ffffff'),(8,6,'#e35335'),(6,4,'#ffffff'),(4,2,'#ff8c00'),
    (2,0,'#ffffff'),(0,-2,'#ffff00'),(-2,-4,'#ffffff'),(-4,-6,'#00cc00'),
    (-6,-8,'#ffffff'),(-8,-10,'#0066ff'),(-10,-12,'#ffffff'),(-12,-14,'#003399'),
    (-14,-16,'#ffffff'),(-16,-18,'#cc88ff'),(-18,-20,'#ffffff'),(-20,-22,'#8800cc'),
    (-22,-24,'#ffffff'),(-24,-26,'#aaaaaa'),(-26,-999,'#ffffff'),
]
_STATIC_GREEN_LO = -6


def _get_normal_band(pressure_level, today=None):
    if today is None: today = date.today()
    table = _NORMALS_850 if pressure_level == 850 else _NORMALS_500
    today_doy = today.timetuple().tm_yday
    best_hi = best_lo = None; best_doy = -1
    for (m, d2, hi, lo) in table:
        try: entry_doy = date(2001, m, d2).timetuple().tm_yday
        except: continue
        if entry_doy <= today_doy and entry_doy > best_doy:
            best_doy = entry_doy; best_hi = hi; best_lo = lo
    if best_hi is None: best_hi, best_lo = table[-1][2], table[-1][3]
    return best_hi, best_lo


def _make_ua_temp_bands(pressure_level, today=None):
    if today is None: today = date.today()
    normal_hi, normal_lo = _get_normal_band(pressure_level, today)
    shift = normal_lo - _STATIC_GREEN_LO
    return [(bhi+shift, blo+shift, col) for (bhi, blo, col) in _UA_TEMP_BANDS_BASE]


def _build_temp_band_fills(grid, lon_vec, lat_vec, ua_temp_bands, tb_base):
    abs_levels = sorted(set(
        [tb_base + b[0] for b in ua_temp_bands] + [tb_base + b[1] for b in ua_temp_bands]))
    if len(abs_levels) < 2: return []
    data_min, data_max = grid.min(), grid.max()
    abs_levels = [l for l in abs_levels if data_min - 0.5 <= l <= data_max + 0.5]
    if len(abs_levels) < 2: return []
    sentinel_lo, sentinel_hi = data_min - 1.0, data_max + 1.0
    abs_levels = [sentinel_lo] + abs_levels + [sentinel_hi]
    grid = np.clip(grid, abs_levels[0] + 0.001, abs_levels[-1] - 0.001)
    colors_list = ['#ffffff']
    for i in range(1, len(abs_levels) - 2):
        lo, hi, matched = abs_levels[i], abs_levels[i+1], '#ffffff'
        for b in ua_temp_bands:
            blo = min(tb_base + b[0], tb_base + b[1])
            bhi = max(tb_base + b[0], tb_base + b[1])
            if abs(lo - blo) < 0.1 and abs(hi - bhi) < 0.1: matched = b[2]; break
        colors_list.append(matched)
    colors_list.append('#ffffff')
    glon, glat = np.meshgrid(lon_vec, lat_vec)
    fig, ax = plt.subplots(figsize=(1, 1))
    try:
        cf = ax.contourf(glon, glat, grid, levels=abs_levels, colors=colors_list, extend='neither')
    except Exception:
        plt.close(fig); return []
    fills = []
    for i, col_hex in enumerate(colors_list):
        if i >= len(cf.levels) - 1: break
        try:
            for seg in cf.allsegs[i]:
                if len(seg) < 3: continue
                fills.append({'color': col_hex,
                              'coords': [[float(v[0]), float(v[1])] for v in seg]})
        except (AttributeError, IndexError): pass
    plt.close(fig)
    return fills


UA_HGHT_LEVELS = {
    850: np.arange(1140, 1650, 30), 700: np.arange(2520, 3180, 60),
    500: np.arange(4800, 6000, 60), 250: None,
}
_INTERVALS = {'HGHT': 6.0, 'TEMP': 2.0, 'TTDP': 2.0, 'SPED': 5.0}


def _build_contours_for_field(recs_df, field_col, interval, fixed_levels=None):
    pts = []
    for _, row in recs_df.iterrows():
        v = row.get(field_col)
        if v is None or (isinstance(v, float) and np.isnan(v)): continue
        pts.append((row['lat'], row['lon'], float(v)))
    if len(pts) < 8: return []
    seen = {}
    for la, lo, v in pts:
        seen.setdefault((round(la,2), round(lo,2)), []).append(v)
    pts = [(k[0], k[1], float(np.mean(vs))) for k, vs in seen.items()]
    if len(pts) < 8: return []
    lats = np.array([p[0] for p in pts]); lons = np.array([p[1] for p in pts])
    vals = np.array([p[2] for p in pts])
    pad = 1.5; N = 180
    lon_vec = np.linspace(lons.min()-pad, lons.max()+pad, N)
    lat_vec = np.linspace(lats.min()-pad, lats.max()+pad, N)
    glon, glat = np.meshgrid(lon_vec, lat_vec)
    try:
        rbf = RBFInterpolator(np.column_stack([lons, lats]), vals,
                              kernel='thin_plate_spline',
                              smoothing=max(0.3*len(pts), 1e-6))
        grid = rbf(np.column_stack([glon.ravel(), glat.ravel()])).reshape(N, N)
    except Exception:
        return []
    grid = gaussian_filter(grid, sigma=2.5)
    if fixed_levels is not None: levels = fixed_levels
    else:
        vmin = np.floor(grid.min()/interval)*interval
        vmax = np.ceil(grid.max()/interval)*interval
        levels = np.arange(vmin, vmax + interval, interval)
    if len(levels) < 2: return []
    fig, ax = plt.subplots(figsize=(1, 1))
    try: cs = ax.contour(glon, glat, grid, levels=levels)
    except Exception: plt.close(fig); return []
    plt.close(fig)
    segments = []
    for li, lv in enumerate(cs.levels):
        for coords in cs.allsegs[li]:
            if len(coords) < 2: continue
            mid = coords[len(coords)//2]
            segments.append({'level': float(lv),
                             'coords': [[float(c[0]),float(c[1])] for c in coords],
                             'label_lon': float(mid[0]), 'label_lat': float(mid[1])})
    return segments


def build_ua_data(ua_summary_df, rt_results):
    print('── Upper-Air Contours ────────────────────────────────')
    _TODAY = date.today()
    UA_TEMP_BANDS_850 = _make_ua_temp_bands(850, _TODAY)
    UA_TEMP_BANDS_500 = _make_ua_temp_bands(500, _TODAY)
    UA_CONTOUR_LEVELS = [850, 700, 500, 250]
    _ts_ua = {}

    for _hr in sorted(ua_summary_df['hour'].unique()):
        _df_hr = ua_summary_df[ua_summary_df['hour'] == _hr].copy()
        _valid = _df_hr['valid_time'].iloc[0] if len(_df_hr) else f'{_hr:02d}Z'
        print(f'  Processing {_valid}...')
        _hr_data = {}

        for _plvl in UA_CONTOUR_LEVELS:
            _lvl_data = {}
            _fixed = UA_HGHT_LEVELS.get(_plvl)
            _segs = (_build_contours_for_field(_df_hr, f'HGHT_{_plvl}',
                      _INTERVALS['HGHT'], fixed_levels=_fixed)
                     if _fixed is not None else [])
            _lvl_data['hght'] = _segs
            _segs = _build_contours_for_field(_df_hr, f'TEMP_{_plvl}', _INTERVALS['TEMP'])
            _lvl_data['temp'] = _segs
            # temp band fills
            _pts_fill = [(float(r['lat']), float(r['lon']), float(r[f'TEMP_{_plvl}']))
                         for _, r in _df_hr.iterrows()
                         if r.get(f'TEMP_{_plvl}') is not None
                         and not (isinstance(r[f'TEMP_{_plvl}'], float) and np.isnan(r[f'TEMP_{_plvl}']))]
            _band_fills = []
            if len(_pts_fill) >= 8:
                _lats_f = np.array([p[0] for p in _pts_fill])
                _lons_f = np.array([p[1] for p in _pts_fill])
                _vals_f = np.array([p[2] for p in _pts_fill])
                _pad_f = 1.5; _NF = 180
                _lv_f = np.linspace(_lons_f.min()-_pad_f, _lons_f.max()+_pad_f, _NF)
                _ltv_f = np.linspace(_lats_f.min()-_pad_f, _lats_f.max()+_pad_f, _NF)
                try:
                    _rbf_f = RBFInterpolator(np.column_stack([_lons_f, _lats_f]), _vals_f,
                                             kernel='thin_plate_spline',
                                             smoothing=max(0.3*len(_pts_fill), 1e-6))
                    _glon_f, _glat_f = np.meshgrid(_lv_f, _ltv_f)
                    _grid_f = _rbf_f(np.column_stack([_glon_f.ravel(), _glat_f.ravel()])).reshape(_NF, _NF)
                    _grid_f = gaussian_filter(_grid_f, sigma=2.5)
                    _bands_for_lvl = UA_TEMP_BANDS_500 if _plvl in [500, 250] else UA_TEMP_BANDS_850
                    _band_fills = _build_temp_band_fills(_grid_f, _lv_f, _ltv_f, _bands_for_lvl, 0.0)
                except Exception: pass
            _lvl_data['temp_band_fills'] = _band_fills
            # T-Td
            _df_ttd = _df_hr.copy()
            _t_col, _d_col = f'TEMP_{_plvl}', f'DWPT_{_plvl}'
            if _t_col in _df_ttd.columns and _d_col in _df_ttd.columns:
                _df_ttd[f'TTDP_{_plvl}'] = _df_ttd[_t_col] - _df_ttd[_d_col]
                _segs = _build_contours_for_field(_df_ttd, f'TTDP_{_plvl}', _INTERVALS['TTDP'])
            else: _segs = []
            _lvl_data['ttdp'] = _segs
            _segs = _build_contours_for_field(_df_hr, f'SPED_{_plvl}', _INTERVALS['SPED'])
            _lvl_data['sped'] = _segs
            _hr_data[str(_plvl)] = _lvl_data

        # Instability contours
        _instab_cts = []
        _stn_lats, _stn_lons, _stn_vals = [], [], []
        for _, _row in _df_hr.iterrows():
            _v5 = _row.get('TEMP_500'); _v7 = _row.get('TEMP_700')
            if (_v5 is None or _v7 is None
                    or (isinstance(_v5, float) and np.isnan(_v5))
                    or (isinstance(_v7, float) and np.isnan(_v7))): continue
            _stn_lats.append(float(_row['lat'])); _stn_lons.append(float(_row['lon']))
            _stn_vals.append(float(_v7) - float(_v5))
        if len(_stn_vals) >= 1:
            _stn_lats = np.array(_stn_lats); _stn_lons = np.array(_stn_lons)
            _stn_vals = np.array(_stn_vals)
            _pad = 1.5; _NI = 180
            _ilon = np.linspace(_stn_lons.min()-_pad, _stn_lons.max()+_pad, _NI)
            _ilat = np.linspace(_stn_lats.min()-_pad, _stn_lats.max()+_pad, _NI)
            _iglon, _iglat = np.meshgrid(_ilon, _ilat)
            _tree = cKDTree(np.column_stack([_stn_lons, _stn_lats]))
            _dists, _idxs = _tree.query(np.column_stack([_iglon.ravel(), _iglat.ravel()]), k=1)
            _nn_vals = _stn_vals[_idxs].astype(float)
            _nn_vals[_dists > 3.5] = np.nan
            _diff_grid = _nn_vals.reshape(_NI, _NI)
            _diff_grid_sm = gaussian_filter(np.where(np.isnan(_diff_grid), 0, _diff_grid),
                                            sigma=sigmaT700500)
            _diff_grid_sm[np.isnan(_diff_grid)] = np.nan
            for _band_lvl in [16, 18]:
                _binary = np.where(
                    (~np.isnan(_diff_grid_sm)) & (_diff_grid_sm >= _band_lvl) &
                    (_diff_grid_sm < (_band_lvl + 2) if _band_lvl == 16 else np.ones_like(_diff_grid_sm, bool)),
                    1.0, 0.0)
                if _binary.max() < 0.5: continue
                _fig_i, _ax_i = plt.subplots(figsize=(1,1))
                try:
                    _cs_i = _ax_i.contour(_iglon, _iglat, _binary, levels=[0.5])
                    for _seg in _cs_i.allsegs[0]:
                        if len(_seg) < 3: continue
                        _mid_i = _seg[len(_seg)//2]
                        _instab_cts.append({'level': float(_band_lvl),
                                            'coords': [[float(p[0]),float(p[1])] for p in _seg],
                                            'label_lon': float(_mid_i[0]), 'label_lat': float(_mid_i[1])})
                except Exception: pass
                plt.close(_fig_i)

        # UA H/L centers
        _ua_hl_all = {}
        for _plvl in [850, 700, 500]:
            _hght_col = f'HGHT_{_plvl}'
            _pts = [(float(r['lat']), float(r['lon']), float(r[_hght_col]))
                    for _, r in _df_hr.iterrows()
                    if r.get(_hght_col) is not None
                    and not (isinstance(r[_hght_col], float) and np.isnan(r[_hght_col]))]
            if len(_pts) < 8: _ua_hl_all[_plvl] = []; continue
            _seen = {}
            for _la, _lo, _v in _pts:
                _seen.setdefault((round(_la,2),round(_lo,2)),[]).append(_v)
            _pts = [(_k[0],_k[1],float(np.mean(_vs))) for _k,_vs in _seen.items()]
            _lats_u = np.array([p[0] for p in _pts]); _lons_u = np.array([p[1] for p in _pts])
            _vals_u = np.array([p[2] for p in _pts])
            _pad = 1.5; _NU = 180
            _lv_u = np.linspace(_lons_u.min()-_pad, _lons_u.max()+_pad, _NU)
            _ltv_u = np.linspace(_lats_u.min()-_pad, _lats_u.max()+_pad, _NU)
            _glon_u, _glat_u = np.meshgrid(_lv_u, _ltv_u)
            try:
                _rbf_u = RBFInterpolator(np.column_stack([_lons_u,_lats_u]), _vals_u,
                                         kernel='thin_plate_spline',
                                         smoothing=max(0.3*len(_pts),1e-6))
                _hght_grid = _rbf_u(np.column_stack([_glon_u.ravel(),_glat_u.ravel()])).reshape(_NU,_NU)
            except Exception: _ua_hl_all[_plvl] = []; continue
            _hght_grid = gaussian_filter(_hght_grid, sigma=2.5)
            sg = gaussian_filter(_hght_grid, sigma=HL_SIGMA)
            max_f = maximum_filter(sg, size=HL_NEIGHBORHOOD)
            min_f = minimum_filter(sg, size=HL_NEIGHBORHOOD)
            is_max = (sg == max_f) & (sg - min_f > 1.0)
            is_min = (sg == min_f) & (max_f - sg > 1.0)
            _ua_hl = []
            for typ, mask in [('H', is_max), ('L', is_min)]:
                lbl2, n = label(mask)
                for i in range(1, n+1):
                    rows2, cols2 = np.where(lbl2 == i)
                    best = np.argmax(sg[rows2,cols2]) if typ=='H' else np.argmin(sg[rows2,cols2])
                    r2, c2 = rows2[best], cols2[best]
                    if r2 < HL_NEIGHBORHOOD or r2 > len(_ltv_u)-HL_NEIGHBORHOOD: continue
                    if c2 < HL_NEIGHBORHOOD or c2 > len(_lv_u)-HL_NEIGHBORHOOD: continue
                    _grid_val = float(_hght_grid[r2, c2])
                    _inside = [r[_hght_col] for _, r in _df_hr.iterrows()
                               if r.get(_hght_col) is not None
                               and not (isinstance(r[_hght_col], float) and np.isnan(r[_hght_col]))]
                    if typ == 'H':
                        _val = (math.floor(max(_inside))+1) if _inside else (math.floor(_grid_val)+1)
                    else:
                        _val = (math.ceil(min(_inside))-1) if _inside else (math.ceil(_grid_val)-1)
                    _ua_hl.append(dict(type=typ, lat=float(_ltv_u[r2]),
                                       lon=float(_lv_u[c2]), val=float(_val)))
            _ua_hl_all[_plvl] = _ua_hl

        def _seg_to_dict(seg):
            mid = seg[len(seg)//2]
            return {'coords': [[float(p[1]),float(p[0])] for p in seg],
                    'label_lon': float(mid[1]), 'label_lat': float(mid[0])}

        hr_key = int(_hr)
        _ts_ua[str(hr_key)] = {
            'levels': _hr_data, 'instab': _instab_cts,
            'thermal_ridge_850':  [_seg_to_dict(s) for s in rt_results.get(f'ridge_850_{hr_key:02d}', [])],
            'thermal_trough_850': [_seg_to_dict(s) for s in rt_results.get(f'trough_850_{hr_key:02d}', [])],
            'thermal_ridge_700':  [_seg_to_dict(s) for s in rt_results.get(f'ridge_700_{hr_key:02d}', [])],
            'thermal_trough_700': [_seg_to_dict(s) for s in rt_results.get(f'trough_700_{hr_key:02d}', [])],
            'thermal_ridge_500':  [_seg_to_dict(s) for s in rt_results.get(f'ridge_500_{hr_key:02d}', [])],
            'thermal_trough_500': [_seg_to_dict(s) for s in rt_results.get(f'trough_500_{hr_key:02d}', [])],
            'dtdx_zero_pts': [],
            **{f'hl_{pl}': _ua_hl_all.get(pl, []) for pl in [850, 700, 500]},
        }
        print(f'  → {_valid}: {len(_instab_cts)} instab segs')
    return _ts_ua


# ══════════════════════════════════════════════════════════════════════════
#  CONVERGENCE  (Cell 15)
# ══════════════════════════════════════════════════════════════════════════
CONV_THRESHOLD  = -1e-5
CONV_GRID_N     = 200
CONV_RBF_SMOOTH = 0.4
CONV_SIGMA      = 3.0


def _deg2m_lat(dlat_deg):   return dlat_deg * 111_320.0
def _deg2m_lon(dlon_deg, lat_deg):
    return dlon_deg * 111_320.0 * np.cos(np.radians(lat_deg))


def compute_divergence_grid(u_grid, v_grid, lon_vec, lat_vec):
    ny, nx = u_grid.shape
    dlon_m = np.zeros((ny, nx))
    dx_deg = lon_vec[1] - lon_vec[0]
    for i in range(ny): dlon_m[i, :] = _deg2m_lon(dx_deg, lat_vec[i])
    du_dx = np.gradient(u_grid, axis=1) / dlon_m
    dlat_m = _deg2m_lat(lat_vec[1] - lat_vec[0])
    dv_dy = np.gradient(v_grid, axis=0) / dlat_m
    return du_dx + dv_dy


def build_wind_grids(pts_u, pts_v, N=CONV_GRID_N, rbf_smooth=CONV_RBF_SMOOTH,
                     sigma=CONV_SIGMA, pad=1.5):
    if len(pts_u) < 6 or len(pts_v) < 6: return None, None, None, None
    def _dedup(pts):
        seen = {}
        for la, lo, v in pts:
            k = (round(la,1), round(lo,1)); seen.setdefault(k,[]).append(v)
        return [(k[0],k[1],float(np.mean(vs))) for k,vs in seen.items()]
    pts_u = _dedup(pts_u); pts_v = _dedup(pts_v)
    if len(pts_u) < 6 or len(pts_v) < 6: return None, None, None, None
    lats_u = np.array([p[0] for p in pts_u]); lons_u = np.array([p[1] for p in pts_u])
    vals_u = np.array([p[2] for p in pts_u])
    lats_v = np.array([p[0] for p in pts_v]); lons_v = np.array([p[1] for p in pts_v])
    vals_v = np.array([p[2] for p in pts_v])
    lat_min = min(lats_u.min(),lats_v.min())-pad; lat_max = max(lats_u.max(),lats_v.max())+pad
    lon_min = min(lons_u.min(),lons_v.min())-pad; lon_max = max(lons_u.max(),lons_v.max())+pad
    lon_vec = np.linspace(lon_min,lon_max,N); lat_vec = np.linspace(lat_min,lat_max,N)
    glon, glat = np.meshgrid(lon_vec,lat_vec); qi = np.column_stack([glon.ravel(),glat.ravel()])
    def _rbf(lons,lats,vals):
        try: rbf = RBFInterpolator(np.column_stack([lons,lats]),vals,kernel='thin_plate_spline',smoothing=max(rbf_smooth*len(vals),1e-6))
        except np.linalg.LinAlgError: rbf = RBFInterpolator(np.column_stack([lons,lats]),vals,kernel='linear',smoothing=max(rbf_smooth*len(vals),1.0))
        return gaussian_filter(rbf(qi).reshape(N,N),sigma=sigma)
    return _rbf(lons_u,lats_u,vals_u), _rbf(lons_v,lats_v,vals_v), lon_vec, lat_vec


def extract_convergence_contours(div_grid, lon_vec, lat_vec, threshold=CONV_THRESHOLD):
    glon, glat = np.meshgrid(lon_vec, lat_vec)
    fig, ax = plt.subplots(figsize=(1,1))
    try: cs = ax.contour(glon, glat, div_grid, levels=[threshold])
    except Exception: plt.close(fig); return []
    plt.close(fig)
    segs = []
    for li, lv in enumerate(cs.levels):
        for coords in cs.allsegs[li]:
            if len(coords) < 3: continue
            mid = coords[len(coords)//2]
            segs.append({'level': float(lv),
                         'coords': [[float(c[0]),float(c[1])] for c in coords],
                         'label_lon': float(mid[0]), 'label_lat': float(mid[1])})
    return segs


def _drct_sped_to_uv(drct_deg, sped_ms):
    rad = math.radians(drct_deg)
    return -sped_ms * math.sin(rad), -sped_ms * math.cos(rad)


def _build_slp_local(recs):
    _pts = [(d['lat'],d['lon'],d['slp']) for d in recs if d.get('slp') is not None]
    if len(_pts) < 8: return None, None, None
    _la = np.array([p[0] for p in _pts]); _lo = np.array([p[1] for p in _pts])
    _va = np.array([p[2] for p in _pts]); _pad=1.5; _N=150
    _lv = np.linspace(_lo.min()-_pad,_lo.max()+_pad,_N)
    _ltv = np.linspace(_la.min()-_pad,_la.max()+_pad,_N)
    _GL, _GLA = np.meshgrid(_lv,_ltv)
    try:
        _rbf = RBFInterpolator(np.column_stack([_lo,_la]),_va,kernel='thin_plate_spline',smoothing=max(0.3*len(_pts),1e-6))
        _grid = _rbf(np.column_stack([_GL.ravel(),_GLA.ravel()])).reshape(_N,_N)
    except Exception: return None, None, None
    return gaussian_filter(_grid, sigma=3.0), _lv, _ltv


def _pts_to_segs_local(pts, max_dist=4.0, min_pts=4):
    if len(pts) < min_pts: return []
    arr = np.array(pts)
    dedup = {}
    for r in arr:
        k = (round(r[0]*2)/2, round(r[1]*2)/2)
        if k not in dedup or r[2] > dedup[k][2]: dedup[k] = r
    arr = np.array(list(dedup.values()))
    if len(arr) < min_pts: return []
    tree = cKDTree(arr[:,:2]); pairs = tree.query_pairs(max_dist)
    parent = list(range(len(arr)))
    def _find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in pairs:
        ra, rb = _find(a), _find(b)
        if ra != rb: parent[ra] = rb
    clusters = {}
    for i in range(len(arr)): clusters.setdefault(_find(i),[]).append(i)
    segs = []
    for idxs in clusters.values():
        if len(idxs) < min_pts: continue
        sp = sorted([(arr[i][0],arr[i][1]) for i in idxs], key=lambda p: p[0])
        sub, cur = [], [sp[0]]
        for k in range(1, len(sp)):
            if (abs(sp[k][0]-sp[k-1][0]) > max_dist*1.5 or
                abs(sp[k][1]-sp[k-1][1]) > max_dist*1.5):
                if len(cur) >= min_pts: sub.append(cur)
                cur = []
            cur.append(sp[k])
        if len(cur) >= min_pts: sub.append(cur)
        segs.extend(sub)
    return segs


def build_convergence_data(metar_records, ua_summary_df):
    print('── Convergence ───────────────────────────────────────')
    _ts_all = sorted(set(d['timestamp'] for d in metar_records if d['timestamp']))
    # SFC convergence
    _conv_sfc_by_ts = {}
    for _ts in _ts_all:
        _recs = [d for d in metar_records if d['timestamp'] == _ts]
        pts_u, pts_v = [], []
        for d in _recs:
            _wd = d.get('wind_dir'); _ws = d.get('wind_spd')
            if _wd is None or _ws is None: continue
            try: _wd = float(_wd); _ws = float(_ws)
            except: continue
            if not (0 <= _wd <= 360) or not (0 <= _ws <= 100): continue
            _ws_ms = _ws * 0.514444 if _ws > 5 else _ws
            _u, _v = _drct_sped_to_uv(_wd, _ws_ms)
            pts_u.append((d['lat'],d['lon'],_u)); pts_v.append((d['lat'],d['lon'],_v))
        u_g, v_g, lv, ltv = build_wind_grids(pts_u, pts_v)
        if u_g is None: _conv_sfc_by_ts[_ts] = []; continue
        div_g = compute_divergence_grid(u_g, v_g, lv, ltv)
        _conv_sfc_by_ts[_ts] = extract_convergence_contours(div_g, lv, ltv)
    print(f'  ✓ SFC: {len(_conv_sfc_by_ts)} timestamps')
    # 850 hPa convergence
    _conv_850_by_hr = {}
    for _hr in sorted(ua_summary_df['hour'].unique()):
        _df = ua_summary_df[ua_summary_df['hour'] == _hr].copy()
        pts_u, pts_v = [], []
        for _, row in _df.iterrows():
            _wd = row.get('DRCT_850'); _ws = row.get('SPED_850')
            if _wd is None or _ws is None: continue
            if isinstance(_wd,float) and np.isnan(_wd): continue
            if isinstance(_ws,float) and np.isnan(_ws): continue
            _u, _v = _drct_sped_to_uv(float(_wd), float(_ws))
            pts_u.append((float(row['lat']),float(row['lon']),_u))
            pts_v.append((float(row['lat']),float(row['lon']),_v))
        u_g, v_g, lv, ltv = build_wind_grids(pts_u, pts_v)
        if u_g is None: _conv_850_by_hr[str(int(_hr))] = []; continue
        div_g = compute_divergence_grid(u_g, v_g, lv, ltv)
        _conv_850_by_hr[str(int(_hr))] = extract_convergence_contours(div_g, lv, ltv)
    print(f'  ✓ 850: {len(_conv_850_by_hr)} hours')
    # SFC trough
    _sfc_trough_by_ts = {}
    for _ts in _ts_all:
        _recs = [d for d in metar_records if d.get('timestamp') == _ts]
        _slp_grid_t, _slp_lv_t, _slp_ltv_t = _build_slp_local(_recs)
        if _slp_grid_t is None: _sfc_trough_by_ts[_ts] = []; continue
        _Ts = gaussian_filter(_slp_grid_t, sigma=2.0)
        _trough_pts = []
        for _j, _lat in enumerate(_slp_ltv_t):
            _row_r = _Ts[_j, :]
            _tr, _tpr = find_peaks(-_row_r, prominence=0.3, width=2)
            if len(_tr):
                _top = np.argsort(_tpr['prominences'])[-4:]
                for _idx in _tr[_top]:
                    _prom = _tpr['prominences'][np.where(_tr == _idx)[0][0]]
                    _trough_pts.append((_lat, _slp_lv_t[_idx], float(_prom)))
        _segs = _pts_to_segs_local(_trough_pts)
        _out = []
        for _seg in _segs:
            _mid = _seg[len(_seg)//2]
            _out.append({'coords':[[p[1],p[0]] for p in _seg],
                         'label_lat': _mid[0], 'label_lon': _mid[1]})
        _sfc_trough_by_ts[_ts] = _out
    total_trough = sum(len(v) for v in _sfc_trough_by_ts.values())
    print(f'  ✓ SFC trough: {total_trough} segs')
    return {'sfc': _conv_sfc_by_ts, '850': _conv_850_by_hr,
            'sfc_trough': _sfc_trough_by_ts, 'threshold': CONV_THRESHOLD}


# ══════════════════════════════════════════════════════════════════════════
#  VORTICITY  (from Cell 16)
# ══════════════════════════════════════════════════════════════════════════
EC_WMS          = 'https://geo.weather.gc.ca/geomet'
VORT_LAYER_NAME = 'RDPS_10km_AbsoluteVorticity_500mb'
_VORT_BBOX      = '-170,40,-50,75'
_VORT_W, _VORT_H = 1200, 800


def fetch_vorticity(ua_date_map):
    print('── Vorticity ─────────────────────────────────────────')
    try:
        cap_resp = requests.get(
            f'{EC_WMS}?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetCapabilities&LAYER={VORT_LAYER_NAME}',
            timeout=30)
        root_ec = ET.fromstring(cap_resp.content)
        raw_times = []
        for _lyr in root_ec.iter('{http://www.opengis.net/wms}Layer'):
            _nel = _lyr.find('{http://www.opengis.net/wms}Name')
            if _nel is not None and _nel.text == VORT_LAYER_NAME:
                for _dim in _lyr.iter('{http://www.opengis.net/wms}Dimension'):
                    if _dim.get('name') == 'time' and _dim.text:
                        raw_times = [t.strip() for t in _dim.text.strip().split(',')]
                        break
        def _expand_times(raw):
            out = []
            for entry in raw:
                if '/' in entry:
                    parts = entry.split('/')
                    s = datetime.fromisoformat(parts[0].replace('Z','+00:00'))
                    e = datetime.fromisoformat(parts[1].replace('Z','+00:00'))
                    step = timedelta(hours=int(parts[2].replace('PT','').replace('H','')))
                    t = s
                    while t <= e: out.append(t.strftime('%Y-%m-%dT%H:%M:%SZ')); t += step
                else: out.append(entry)
            return out
        _ec_times = _expand_times(raw_times)
        print(f'  RDPS: {len(_ec_times)} steps')
        vort_time_map = {}
        for _k, _ds in ua_date_map.items():
            _dt = datetime.strptime(str(_ds).strip(), '%Y-%m-%d %HZ').replace(tzinfo=timezone.utc)
            _best = min(_ec_times, key=lambda t: abs(
                datetime.fromisoformat(t.replace('Z','+00:00')) - _dt))
            vort_time_map[_k] = _best
    except Exception as e:
        print(f'  ✗ GetCapabilities failed: {e}'); return {}

    def _fetch_vort_b64(time_str):
        _vurl = (f'{EC_WMS}?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap'
                 f'&LAYERS={VORT_LAYER_NAME}&BBOX={_VORT_BBOX}&CRS=CRS:84'
                 f'&WIDTH={_VORT_W}&HEIGHT={_VORT_H}'
                 f'&FORMAT=image/png&TRANSPARENT=TRUE&TIME={time_str}&STYLES=')
        try:
            _r = requests.get(_vurl, timeout=30); _r.raise_for_status()
            if _r.content[:4] != b'\x89PNG': return None
            _img = PILImage.open(io.BytesIO(_r.content)).convert('RGBA')
            _arr = np.array(_img, dtype=np.float32)
            _R, _G, _B = _arr[:,:,0], _arr[:,:,1], _arr[:,:,2]
            _is_yellow = (_R > 180) & (_G > 150) & (_B < 80)
            _is_red    = (_R > 180) & (_G < 100) & (_B < 80)
            _keep = _is_yellow | _is_red
            _out = _arr.copy().astype(np.uint8)
            _out[:,:,3] = np.where(_keep, 220, 0)
            _filtered = PILImage.fromarray(_out, 'RGBA')
            _buf = io.BytesIO(); _filtered.save(_buf, format='PNG'); _buf.seek(0)
            _b = base64.b64encode(_buf.read()).decode()
            print(f'  ✓ {time_str} ({int(_keep.sum()):,} px kept)')
            return f'data:image/png;base64,{_b}'
        except Exception as _e: print(f'  ✗ {time_str}: {_e}'); return None

    _vort_images = {}
    for _k, _t in vort_time_map.items():
        _img = _fetch_vort_b64(_t)
        if _img: _vort_images[_k] = _img
    print(f'  ✓ Vort images: {list(_vort_images.keys())}')
    return _vort_images


# ══════════════════════════════════════════════════════════════════════════
#  METAR STATION DATA (ts_data / syn_ts_data.json)
# ══════════════════════════════════════════════════════════════════════════
def build_ts_data(metar_records):
    _ts_all = sorted(set(d['timestamp'] for d in metar_records if d['timestamp']))
    _ts_data = {}
    for _ts in _ts_all:
        _entries = []
        for _d in [d for d in metar_records if d['timestamp'] == _ts]:
            _pop = (f'<div style="font-family:monospace;font-size:11px;min-width:220px">'
                    f'<b style="font-size:13px">{_d["icao"]}</b> '
                    f'<span style="color:#888;font-size:10px">{_d["name"]}</span><br>'
                    f'T: <b>{_d["temp"]}C</b>  Td: <b>{_d["dew"]}C</b>  '
                    f'Wind: <b>{_d["wind_dir"]}/{_d["wind_spd"]}kt</b><br>'
                    f'Vis: <b>{_d["vis"]} SM</b> Wx: <b>{_d["weather"] or "NIL"}</b><br>'
                    f'SLP: <b>{_d["slp"]} hPa</b> RH: <b>{_d["rh"]}%</b><br>'
                    f'Cloud: <b>' + ' '.join(c['raw'] for c in _d['clouds']) + '</b><br>'
                    f'<a href="https://aviationweather.gov/api/data/metar?ids={_d["icao"]}&hours=24&taf=1" '
                    f'target="_blank" style="font-size:10px;color:#1a4a8a;">METAR+TAF ↗</a></div>')
            _svg_str, _sw, _sh = station_model_svg({**_d, 'is_surface': True}, S=34)
            _entries.append({'lat': _d['lat'], 'lon': _d['lon'], 'popup': _pop,
                             'tip': f'{_d["icao"]} {_d["temp"]}C/{_d["dew"]}C {_d["wind_dir"]}/{_d["wind_spd"]}kt',
                             'svg': _svg_str, 'svg_w': int(_sw), 'svg_h': int(_sh)})
        _ts_data[_ts] = _entries
    return _ts_data


# ══════════════════════════════════════════════════════════════════════════
#  UA STATION DATA (syn_ua_stns.json)
# ══════════════════════════════════════════════════════════════════════════
def build_ua_stns(ua_summary_df):
    _ua_stn_data = {}
    for _hour, _grp in ua_summary_df.groupby('hour'):
        _key = str(int(_hour)); _stns = []
        for _, _r in _grp.iterrows():
            def _fmt(v, dec=1):
                return f'{v:.{dec}f}' if v is not None and not (isinstance(v,float) and math.isnan(v)) else '—'
            def _fmti(v):
                return f'{int(round(v))}' if v is not None and not (isinstance(v,float) and math.isnan(v)) else '—'
            _pop = (f'<div style="font-family:monospace;font-size:11px;min-width:240px">'
                    f'<b style="font-size:13px;color:#cc6600">{_r["icao"]}</b> '
                    f'<span style="color:#888;font-size:10px">{_r["stn_name"]}</span><br>'
                    f'<span style="color:#888;font-size:10px">'
                    f'Lat: <b>{float(_r["lat"]):.2f}°N</b> &nbsp; '
                    f'Lon: <b>{float(_r["lon"]):.2f}°E</b> &nbsp; '
                    f'WMO: <b>{_r["wmo"]}</b></span><hr style="margin:4px 0">')
            for _lvl in [850, 700, 500, 250]:
                _h = _fmti(_r.get(f'HGHT_{_lvl}'))
                _t = _fmt(_r.get(f'TEMP_{_lvl}'))
                _td = _fmt(_r.get(f'DWPT_{_lvl}'))
                _tv = _r.get(f'TEMP_{_lvl}'); _tdv = _r.get(f'DWPT_{_lvl}')
                _ttd = (f'{_tv - _tdv:.1f}' if _tv is not None and _tdv is not None
                        and not (isinstance(_tv,float) and math.isnan(_tv))
                        and not (isinstance(_tdv,float) and math.isnan(_tdv)) else '—')
                _wd = _fmti(_r.get(f'DRCT_{_lvl}')); _ws = _fmti(_r.get(f'SPED_{_lvl}'))
                _pop += (f'<b>{_lvl}:</b> H:{_h}m  T:{_t}°C  Td:{_td}°C  '
                         f'T-Td:{_ttd}  {_wd}/{_ws}kts<br>')
            _pop += '</div>'
            _stns.append({'lat': float(_r['lat']), 'lon': float(_r['lon']),
                          'icao': _r['icao'], 'popup': _pop})
        _ua_stn_data[_key] = _stns
    return _ua_stn_data


# ══════════════════════════════════════════════════════════════════════════
#  FIRE ZONES (static)
# ══════════════════════════════════════════════════════════════════════════
def get_fire_zones_json():
    """Read fire zones from notebook Cell 4 — it's a static string."""
    # The actual GeoJSON was embedded in the notebook as a string literal.
    # We return a minimal placeholder here; the full string should be copied
    # from the notebook's _fire_zones_geojson_str variable if needed.
    # For automation, this file never changes so we can just re-use what's
    # already in the repo from a previous push.
    return None   # signal to skip writing this file (keep existing repo copy)


# ══════════════════════════════════════════════════════════════════════════
#  WRITE JSON FILES
# ══════════════════════════════════════════════════════════════════════════
def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)
    sz = os.path.getsize(path) / 1024
    print(f'  → {os.path.basename(path)}  ({sz:.0f} KB)')


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='Synoptic map data pipeline')
    parser.add_argument('--outdir', default='.', help='Output directory for JSON files')
    args = parser.parse_args()
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    print('\n════════════════════════════════════════════════════')
    print('  Synoptic Map Data Pipeline')
    print(f'  {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}')
    print('════════════════════════════════════════════════════\n')

    # 1. METAR
    metar_records = fetch_metar()

    # 2. Upper air
    ua_summary_df, ua_date_map = fetch_upper_air()

    # 3. Ridge/trough
    rt_results = detect_ridges_troughs(ua_summary_df)

    # 4. SLP contours
    ts_slp = build_slp_data(metar_records)

    # 5. UA contours
    ts_ua = build_ua_data(ua_summary_df, rt_results)

    # 6. Convergence
    conv_data = build_convergence_data(metar_records, ua_summary_df)

    # 7. Vorticity
    vort_images = fetch_vorticity(ua_date_map)

    # 8. Station SVG data
    print('── Building station SVG data ─────────────────────────')
    ts_data  = build_ts_data(metar_records)
    ua_stns  = build_ua_stns(ua_summary_df)

    # 9. Write JSON files
    print('\n── Writing JSON files ────────────────────────────────')
    write_json(os.path.join(outdir, 'syn_ts_data.json'), ts_data)
    write_json(os.path.join(outdir, 'syn_slp.json'),     ts_slp)
    ts_ua['_ua_dates'] = ua_date_map
    write_json(os.path.join(outdir, 'syn_ua.json'),      ts_ua)    write_json(os.path.join(outdir, 'syn_ua_stns.json'), ua_stns)
    write_json(os.path.join(outdir, 'vort_images.json'), vort_images)
    write_json(os.path.join(outdir, 'conv_data.json'),   conv_data)
    # fire_zones_geojson.json is static — skip if already in repo
    fire_zones_path = os.path.join(outdir, 'fire_zones_geojson.json')
    if not os.path.exists(fire_zones_path):
        print('  ⚠ fire_zones_geojson.json not found — run once from Colab to seed it')

    print('\n✅ All data files written.')
    print(f'   Output: {os.path.abspath(outdir)}')


if __name__ == '__main__':
    main()
