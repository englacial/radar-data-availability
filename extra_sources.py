"""Campaign tracks from direct provider sources, for all figures.

* AWI ice sounders (EMR, UWB, UWBM) from the AWI map server WFS layer
  ``radar:tracks_3031``, dated per profile, 1994/95 to date. Replaces the
  AWI BedMap files (see ``bedmap_common.EXTERNAL_SOURCE_PREFIXES``).
* UTIG/KOPRI helicopter radar over the Amundsen Sea from the QIceRadar index
  (Lindzey 2026, v0.3.0), layers ASE2-ASE6. Absent from BedMap and xOPR.
* xOPR collections substituted for unusable BedMap files (``opr_tracks``).

``load_extra_campaigns()`` returns rows shaped like ``load_bedmap_catalog``
plus ``source``, ``season`` (July-June start year), ``access`` and
``line_km``; ``extra_segments(epsg)`` returns segment endpoints for gridding.
Raw downloads are cached under ``radar_cache/`` (gitignored).
"""

import urllib.request
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer

from bedmap_common import geod_km

CACHE = Path(__file__).parent / "radar_cache"
AWI_WFS = ("https://maps.awi.de/services/common/radar/ows?service=WFS&version=2.0.0"
           "&request=GetFeature&typeNames=radar:tracks_3031&outputFormat=application/json")
AWI_SOUNDERS = {"EMR", "UWB", "UWBM"}  # ACCU/SNOW/ASIRAS are accumulation radars
AWI_GENERIC_DOI = "10.1594/PANGAEA.972094"  # collection bibliography, not a dataset release
QICERADAR_GPKG = "https://zenodo.org/api/records/21964546/files/qiceradar_antarctic_index.gpkg/content"
# KOPRI Araon cruises are annual and visit Thwaites every other year (last
# 2025/26); ASE3/ASE4 dates confirmed from the TDR file names.
KOPRI_SEASONS = {"ASE2": 2017, "ASE3": 2019, "ASE4": 2021, "ASE5": 2023, "ASE6": 2025}


def _fetch(url, path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def _season(ts):
    return ts.dt.year.where(ts.dt.month >= 7, ts.dt.year - 1)


def load_awi_tracks():
    """All features of the AWI WFS layer (EPSG:3031) with ``km`` added; cached as parquet."""
    pq = CACHE / "awi" / "tracks_3031.parquet"
    if pq.exists():
        return gpd.read_parquet(pq)
    g = gpd.read_file(_fetch(AWI_WFS, CACHE / "awi" / "tracks_3031.geojson")).set_crs(3031, allow_override=True)
    g = g[g.geom_type.isin(["LineString", "MultiLineString"])].copy()
    g["km"] = g.to_crs(4326)["geometry"].apply(geod_km)
    g.to_parquet(pq)
    return g


def load_awi_campaigns():
    """AWI ice-sounder tracks, one row per profile segment, duplicates removed."""
    g = load_awi_tracks()
    g = g[g["radar_system"].isin(AWI_SOUNDERS)].copy()
    # UWB/UWBM list a 'standard' and a 'qlook' product per granule: keep one.
    qlook = g["quick_looks"].fillna("").str.contains("_qlook")
    key = g["prof_id"] + "|" + g["date_time_start"].astype(str)
    g = g[~(qlook & key.isin(key[~qlook]))]
    # EMR lists the 60 ns and 600 ns pulse as separate profiles of the same
    # flight (prof_id 'antrYYYY_..._EMR60[0]_...' or an 8-digit flight number
    # whose third digit is the product): keep the longer product per flight.
    flight = (g["prof_id"].str.replace(r"_EMR600?_", "_EMR_", regex=True)
              .str.replace(r"^(\d{4})\d(\d{3})$", r"\1_\2", regex=True))
    best = g.groupby([flight, g["radar_name"]])["km"].sum().groupby(level=0).idxmax().str[1]
    g = g[g["radar_name"].values == flight.map(best).values]
    ts = pd.to_datetime(g["date_time_start"], utc=True)
    out = gpd.GeoDataFrame({
        "name": "AWI_" + g["season"].str.replace(" ", "_"),
        "institution": "AWI", "source": "awi_wfs",
        "season": _season(ts).astype(int), "ts": ts, "te": pd.to_datetime(g["date_time_end"], utc=True),
        "radar_system": g["radar_system"], "doi": g["doi"],
        "released": g["doi"].notna() & ~g["doi"].str.contains(AWI_GENERIC_DOI, na=False),
        "access": "committed",  # AWI policy; released data are not in xOPR
        "line_km": g["km"],
    }, geometry=g.to_crs(4326)["geometry"].values, crs="EPSG:4326")
    return out.reset_index(drop=True)


def load_kopri_campaigns():
    """UTIG/KOPRI helicopter radar tracks (QIceRadar ASE layers)."""
    gpkg = _fetch(QICERADAR_GPKG, CACHE / "qiceradar" / "qiceradar_antarctic_index.gpkg")
    parts = []
    for layer, season in KOPRI_SEASONS.items():
        g = gpd.read_file(gpkg, layer=layer).to_crs(4326)
        g = g[g.geom_type == "LineString"]
        parts.append(gpd.GeoDataFrame({
            "name": f"KOPRI_{layer}", "institution": "KOPRI", "source": "qiceradar",
            "season": season, "access": "none", "line_km": g["geometry"].apply(geod_km),
        }, geometry=g["geometry"].values, crs="EPSG:4326"))
    return pd.concat(parts, ignore_index=True)


def load_extra_campaigns(include_opr=True):
    parts = [load_awi_campaigns(), load_kopri_campaigns()]
    if include_opr:
        from opr_tracks import opr_campaigns
        parts.append(opr_campaigns())
    return pd.concat(parts, ignore_index=True)


def extra_segments(epsg, campaigns=None):
    """Segment endpoints (x1, y1, x2, y2) in ``epsg`` for gridding line-km."""
    g = load_extra_campaigns() if campaigns is None else campaigns
    tf = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    x1, y1, x2, y2 = [], [], [], []
    for geom in g["geometry"]:
        for line in (geom.geoms if hasattr(geom, "geoms") else [geom]):
            c = np.asarray(line.coords)
            if len(c) < 2:
                continue
            x, y = tf.transform(c[:, 0], c[:, 1])
            x1.append(x[:-1]); y1.append(y[:-1]); x2.append(x[1:]); y2.append(y[1:])
    return tuple(np.concatenate(a) for a in (x1, y1, x2, y2))


if __name__ == "__main__":
    c = load_extra_campaigns()
    print(c.groupby(["source", "institution", "season"])["line_km"].sum().round(0).to_string())
