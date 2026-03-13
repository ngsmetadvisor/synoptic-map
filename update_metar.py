"""
update_metar.py  —  Hourly METAR pipeline
Fetches METAR data, builds SLP/T-Td/Temp contours, convergence, and station SVGs.
Runs hourly via GitHub Actions.

Output files:
  syn_ts_data.json   — per-timestamp METAR station SVGs
  syn_slp.json       — SLP/HL/T-Td/Temp contours per timestamp
  conv_data.json     — convergence zones (SFC, 850, SFC trough)

Usage:
  python update_metar.py                   # write to current directory
  python update_metar.py --outdir /path/   # write to custom directory
"""

import argparse
import csv
import io
import json
import math
import os
import re
import time
import warnings
import concurrent.futures
from collections import defaultdict
from datetime import datetime, timezone
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import (gaussian_filter, label,
                            maximum_filter, minimum_filter)
from scipy.signal import find_peaks

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
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

TMP_RBF_SMOOTHING = 0.05
TMP_SIGMA         = 1.0
TTD_RBF_SMOOTHING = 0.100
TTD_SIGMA         = 1.0

CONV_THRESHOLD  = -1e-5
CONV_GRID_N     = 200
CONV_RBF_SMOOTH = 0.4
CONV_SIGMA      = 3.0

_CR           = 0.14
FEATHER_ANGLE = 110
FEATHER_SIDE  = +1


# ══════════════════════════════════════════════════════════════════════════
#  STATION MODEL SVG
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
#  METAR
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
    wind_dir = wind_spd = wind_gust = None
    for p in rest:
        m = re.match(r'^(\d{3})(\d{2,3})(?:G(\d{2,3}))?KT$', p)
        if m: wind_dir, wind_spd = int(m[1]), int(m[2]); wind_gust = int(m[3]) if m[3] else 0; break
        if re.match(r'^00000KT$', p): wind_dir=0; wind_spd=0; wind_gust=0; break
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
    temp = dew = None
    for p in rest:
        m = re.match(r'^(M?\d{1,2})/(M?\d{1,2})$', p)
        if m:
            def td(s): return -(int(s[1:])) if s.startswith('M') else int(s)
            temp, dew = td(m[1]), td(m[2]); break
    slp = None
    for p in rest:
        m = re.match(r'^SLP(\d{3})$', p)
        if m: v = int(m[1]); slp = (900 + v/10) if v >= 500 else (1000 + v/10); break
    wx_re = re.compile(
        r'^[+-]?(FZ|SH|BL|TS|MI|PR|BC|DR)?'
        r'(DZ|RA|SN|SG|IC|PL|GR|GS|UP|FG|BR|HZ|FU|VA|DU|SA|SQ|PO|FC|SS|DS){1,3}$')
    wx_parts = [p for p in rest if wx_re.match(p)
                and not re.match(r'^(RMK|SLP|AUTO|COR|AO\d)', p)]
    weather = ' '.join(wx_parts)
    rh = None
    if temp is not None and dew is not None:
        a, b = 17.625, 243.04
        rh = round(100 * np.exp((a*dew/(b+dew)) - (a*temp/(b+temp))))
        rh = max(0, min(100, rh))
    t_td = round(temp - dew, 1) if temp is not None and dew is not None else None
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
#  GRID / CONTOUR HELPERS
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
#  SLP / CONTOURS
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
#  CONVERGENCE
# ══════════════════════════════════════════════════════════════════════════
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


def build_convergence_data(metar_records):
    print('── Convergence ───────────────────────────────────────')
    _ts_all = sorted(set(d['timestamp'] for d in metar_records if d['timestamp']))
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
    return {'sfc': _conv_sfc_by_ts, '850': {}, 'sfc_trough': _sfc_trough_by_ts,
            'threshold': CONV_THRESHOLD}


# ══════════════════════════════════════════════════════════════════════════
#  STATION SVG DATA
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
#  WRITE JSON
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
    parser = argparse.ArgumentParser(description='Hourly METAR pipeline')
    parser.add_argument('--outdir', default='.', help='Output directory for JSON files')
    args = parser.parse_args()
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    print('\n════════════════════════════════════════════════════')
    print('  METAR Pipeline (Hourly)')
    print(f'  {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}')
    print('════════════════════════════════════════════════════\n')

    metar_records = fetch_metar()
    ts_slp        = build_slp_data(metar_records)
    conv_data     = build_convergence_data(metar_records)
    ts_data       = build_ts_data(metar_records)

    print('\n── Writing JSON files ────────────────────────────────')
    write_json(os.path.join(outdir, 'syn_ts_data.json'), ts_data)
    write_json(os.path.join(outdir, 'syn_slp.json'),     ts_slp)
    write_json(os.path.join(outdir, 'conv_data.json'),   conv_data)

    print('\n✅ METAR data files written.')
    print(f'   Output: {os.path.abspath(outdir)}')


if __name__ == '__main__':
    main()
