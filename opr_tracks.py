"""Real flight tracks for xOPR collections, from CReSIS per-segment CSVs.

The STAC catalog geometry is simplified, so for line-km and gridding we use
``https://data.cresis.ku.edu/data/rds/<collection>/csv/Data_<segment>.csv``
(full-rate LAT/LON/TIME/FRAME per record, ~3 MB per segment), cached under
``radar_cache/opr_csv/``. Used for the collections in
``bedmap_common.OPR_SUBSTITUTIONS`` (BedMap files that cannot be chained into
lines, and xOPR-only seasons); ``python opr_tracks.py --verify`` compares
track lengths with the BedMap files they replace or duplicate.
"""

import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Geod
from shapely.geometry import LineString, MultiLineString

from bedmap_common import OPR_SUBSTITUTIONS, geod_km

CSV_URL = "https://data.cresis.ku.edu/data/rds/{collection}/csv/"
CACHE = Path(__file__).parent / "radar_cache" / "opr_csv"
GAP_M = 2000  # split a segment where consecutive records are further apart
_GEOD = Geod(ellps="WGS84")


def _fetch(url, path):
    from extra_sources import _fetch as fetch  # shared retrying download
    return fetch(url, path)


def list_segment_csvs(collection):
    """Segment CSV file names on the CReSIS server (cached listing)."""
    idx = _fetch(CSV_URL.format(collection=collection), CACHE / collection / "index.html")
    return sorted(set(re.findall(r'href="(Data_\d{8}_\d{2}\.csv)"', idx.read_text())))


def _segment_line(df):
    """Chain records into a (Multi)LineString, breaking at gaps > GAP_M."""
    lon, lat = df["LON"].to_numpy(), df["LAT"].to_numpy()
    d = _GEOD.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])[2]
    breaks = np.flatnonzero(d > GAP_M) + 1
    parts = [LineString(c) for c in np.split(np.column_stack([lon, lat]), breaks) if len(c) > 1]
    return MultiLineString(parts) if len(parts) > 1 else parts[0]


def load_opr_tracks(collection, only_stac_segments=True):
    """One row per segment: collection, segment, date, season, provider (STAC
    ``opr:provider``), geometry (lon/lat), line_km."""
    stac_segments, provider = None, "CRESIS"
    if only_stac_segments:
        import xopr
        items = xopr.OPRConnection().query_frames(collections=[collection], exclude_geometry=True)
        stac_segments = {f"{p['opr:date']}_{int(p['opr:segment']):02d}" for p in items["properties"]}
        provider = pd.Series([p.get("opr:provider", "cresis") for p in items["properties"]]).mode()[0].upper()
    rows = []
    for name in list_segment_csvs(collection):
        seg = name[5:-4]
        if stac_segments is not None and seg not in stac_segments:
            continue
        df = pd.read_csv(_fetch(CSV_URL.format(collection=collection) + name, CACHE / collection / name),
                         usecols=["LAT", "LON"])
        df = df[(df["LAT"].abs() <= 90) & (df["LON"].abs() <= 180)]
        if len(df) < 2:
            continue
        date = pd.Timestamp(seg[:8])
        rows.append({"collection": collection, "segment": seg, "date": date,
                     "season": date.year if date.month >= 7 else date.year - 1,
                     "provider": provider, "geometry": _segment_line(df)})
    g = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    g["line_km"] = g["geometry"].apply(geod_km)
    return g


def opr_campaigns():
    """Campaign rows (one per collection) for the substituted xOPR collections."""
    rows = []
    for coll in OPR_SUBSTITUTIONS:
        t = load_opr_tracks(coll)
        season = int(t["season"].mode().iloc[0])
        inst = t["provider"].iloc[0]
        rows.append({"name": f"{inst}_{coll}", "institution": inst, "source": "opr",
                     "season": season, "access": "open", "line_km": t["line_km"].sum(),
                     "geometry": MultiLineString([p for g in t["geometry"]
                                                  for p in (g.geoms if hasattr(g, "geoms") else [g])])})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


# BedMap files that correspond to xOPR collections (from QIceRadar's
# bedmap_labels.py); used only by --verify.
VERIFY_MAP = {
    "2004_Antarctica_P3chile": ["NASA_2004_ICEBRIDGE_AIR_BM2"],
    "2009_Antarctica_TO": OPR_SUBSTITUTIONS["2009_Antarctica_TO"],
    "2009_Antarctica_DC8": ["NASA_2009_ICEBRIDGE_AIR_BM2"],
    "2010_Antarctica_DC8": ["NASA_2010_ICEBRIDGE_AIR_BM2"],
    "2012_Antarctica_DC8": ["NASA_2012_ICEBRIDGE_AIR_BM2"],
    "2013_Antarctica_Basler": ["CRESIS_2013_Siple-Coast_AIR_BM3"],
    "2013_Antarctica_P3": ["NASA_2013_ICEBRIDGE_AIR_BM3"],
    "2019_Antarctica_GV": ["NASA_2019_ICEBRIDGE_AIR_BM3"],
}


def verify():
    """Compare CSV track km with the STAC geometry and the matching BedMap files."""
    import xopr
    from bedmap_common import load_bedmap_catalog
    cat = load_bedmap_catalog(["bedmap2", "bedmap3"], exclude=False).set_index("name")
    conn = xopr.OPRConnection()
    for coll, bm_names in VERIFY_MAP.items():
        t = load_opr_tracks(coll)
        stac = conn.query_frames(collections=[coll], exclude_geometry=False)
        stac_km = stac["geometry"].apply(geod_km).sum()
        bm_km = sum(geod_km(cat.loc[n, "geometry"]) for n in bm_names if n in cat.index)
        print(f"{coll:26s} csv {t['line_km'].sum():9.0f} km | STAC {stac_km:9.0f} km | "
              f"BedMap catalog {bm_km:9.0f} km ({', '.join(bm_names)})")


if __name__ == "__main__":
    import sys
    if "--verify" in sys.argv:
        verify()
    else:
        for coll in OPR_SUBSTITUTIONS:
            t = load_opr_tracks(coll)
            print(f"{coll}: {len(t)} segments, {t['line_km'].sum():.0f} km, seasons {sorted(t['season'].unique())}")
