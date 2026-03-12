#!/usr/bin/env python3
"""
Synoptic map data updater — runs on GitHub Actions schedule.
Fetches fresh data and writes JSON files consumed by synoptic_map.html.
"""

import json, os, requests, numpy as np
from datetime import datetime, timezone

# ── same logic as your notebook cells ─────────────────────────────────────

def fetch_metar_data():
    # your Cell 7.x METAR fetching code here
    ...
    return ts_data, slp_data

def fetch_ua_data():
    # your upper-air fetching code here
    ...
    return ua_data, ua_stn_data

def fetch_vort_images():
    # your vorticity WMS fetch + base64 encode here
    ...
    return vort_images

def fetch_conv_data():
    # your convergence zone code here
    ...
    return conv_data

# ── run and write JSONs ────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"Running at {datetime.now(timezone.utc).strftime('%Y-%m-%d %HZ')}")

    ts_data, slp_data = fetch_metar_data()
    ua_data, ua_stn_data = fetch_ua_data()
    vort_images = fetch_vort_images()
    conv_data = fetch_conv_data()

    files = {
        'syn_ts_data.json':       ts_data,
        'syn_slp.json':           slp_data,
        'syn_ua.json':            ua_data,
        'syn_ua_stns.json':       ua_stn_data,
        'vort_images.json':       vort_images,
        'conv_data.json':         conv_data,
    }

    for fname, data in files.items():
        with open(fname, 'w') as f:
            json.dump(data, f)
        print(f"  ✓ wrote {fname} ({os.path.getsize(fname)//1024} KB)")

    print("Done.")
