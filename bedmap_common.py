"""Shared BedMap catalog handling.

Every BedMap consumer (bar charts and the point-data density loader) should
go through here so that campaign exclusions, BM2/BM3 deduplication, and
temporal metadata fixes are applied in exactly one place.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Geod, Transformer
from xopr.bedmap.query import query_bedmap_catalog

# Consecutive BedMap points further apart than this are not joined into a
# flight line. Applied to the density grids and to the per-campaign line-km
# used by the bar charts. 1.5 km keeps the airborne surveys sampled near 1 km
# (SOAR-LVS-WLK, CASERTZ, MAMOG: 95th-percentile spacing <= 1.35 km) and still
# drops ground traverses and the 25 km-spaced GANOVEX files (p95 >= 6.5 km);
# see claude_notes/bedmap_gap_rule_options.csv.
MAX_POINT_SPACING_M = 1500
CAMPAIGN_KM_FILE = Path(__file__).parent / "bedmap_campaign_km.csv"

# Campaigns dropped everywhere. CRESIS_2009_Thwaites is a MUSIC swath product
# (~30 cross-track bed picks per radar record) and the Vostok file is scattered
# points; neither can be chained into flight lines.
EXCLUDE_NAMES = {
    "RNRF_2008_Vostok-Subglacial-Lake_AIR_BM2",
    "CRESIS_2009_Thwaites_AIR_BM3",
}

# BM2 files whose points all lie within 50 m of a differently named BM3 file
# (renamed on resubmission; checked at point level 2026-09-14), so the suffix
# dedup below cannot catch them. UTIG_2008_ICECAP is 88 % inside UTIG_2010_ICECAP.
SUPERSEDED_NAMES = {
    "RNRF_2003_48RAEap5_AIR_BM2", "RNRF_2004_49RAEap5_AIR_BM2",
    "RNRF_2005_50RAEap5_AIR_BM2", "RNRF_2006_51RAEap5_AIR_BM2",
    "RNRF_2007_52RAEap5_AIR_BM2", "RNRF_2008_53RAEap5_AIR_BM2",
    "RNRF_2007_Mirny-Vostok_AIR_BM2", "RNRF_2006_KV1-area_AIR_BM2",
    "UTIG_2008_ICECAP_AIR_BM2",
}

# xOPR collections loaded from real flight tracks (opr_tracks.py): keys are
# collections, values the BedMap files they replace (points that cannot be
# chained into flight lines) or [] for xOPR-only seasons (UTIG COLDEX 2022/23,
# 2023/24 and GHOST2 2024/25 are not in BedMap3).
OPR_SUBSTITUTIONS = {
    "2009_Antarctica_TO": ["CRESIS_2009_Thwaites_AIR_BM3",
                           "CRESIS_2009_AntarcticaTO_AIR_BM3"],
    "2022_Antarctica_BaslerMKB": [],
    "2023_Antarctica_BaslerMKB": [],
    "2024_Antarctica_GroundGHOST2": [],
}

# Institutions taken entirely from a direct, dated source (extra_sources.py)
# instead of their BedMap submissions.
EXTERNAL_SOURCE_PREFIXES = ("AWI_",)

# Files whose catalog dates are known to be wrong: name -> (start, end).
# UTIG_1999_SOAR-LVS-WLK bundles the SOAR surveys of 1996-2001 (TAM/PPT,
# WLK, BSB, IRE, WAZ/TKD, LVS; USAP-DC 601588) under a 1999-2000 placeholder.
TEMPORAL_OVERRIDES = {
    "UTIG_1999_SOAR-LVS-WLK_AIR_BM2": ("1996-01-01", "2001-12-31"),
}

_VERSION_SUFFIX = re.compile(r"_BM[123]$")
_GEOD = Geod(ellps="WGS84")


def load_bedmap_catalog(collections=("bedmap2", "bedmap3"), exclude=True):
    """Return catalog entries with one row per campaign.

    Applies ``EXCLUDE_NAMES``, ``SUPERSEDED_NAMES``, ``OPR_SUBSTITUTIONS`` and
    ``EXTERNAL_SOURCE_PREFIXES``, keeps the newest BedMap version when a
    campaign appears in several (BM3 over BM2 over BM1), and parses temporal
    metadata into ``ts``/``te``. A missing or sentinel end date (e.g. year
    9999) falls back to the start date; ``TEMPORAL_OVERRIDES`` replaces known
    bad ranges.

    ``line_km`` is the gap-filtered length of the campaign's point data
    (see ``campaign_point_km``), not the simplified catalog geometry.

    Columns: collection, geometry, name, base_name, institution,
    temporal_start, temporal_end, ts, te, line_km.
    """
    cat = query_bedmap_catalog(collections=list(collections))
    props = pd.DataFrame(cat["properties"].tolist(), index=cat.index)
    df = pd.concat(
        [cat[["collection", "geometry"]],
         props[["name", "institution", "temporal_start", "temporal_end"]]],
        axis=1,
    )
    if exclude:
        df = df[~df["name"].isin(dropped_names())
                & ~df["name"].str.startswith(EXTERNAL_SOURCE_PREFIXES)]
    df["base_name"] = df["name"].str.replace(_VERSION_SUFFIX, "", regex=True)
    # Suffixes sort BM1 < BM2 < BM3, so keep="last" keeps the newest version.
    df = df.sort_values("name").drop_duplicates("base_name", keep="last")

    df["ts"] = pd.to_datetime(df["temporal_start"], format="ISO8601", errors="coerce")
    te = pd.to_datetime(df["temporal_end"], format="ISO8601", errors="coerce")
    bad_end = te.isna() | (te.dt.year > 2100)
    df["te"] = te.where(~bad_end, df["ts"])
    for name, (start, end) in TEMPORAL_OVERRIDES.items():
        df.loc[df["name"] == name, ["ts", "te"]] = [pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")]
    km = campaign_point_km()
    missing = df.loc[~df["name"].isin(km.index), "name"].tolist()
    if missing:
        print(f"bedmap_common: no point-based km for {missing}; using catalog geometry")
    df["line_km"] = df["name"].map(km).fillna(df["geometry"].apply(geod_km))
    return df.reset_index(drop=True)


def load_bedmap_points(collections=("bedmap1", "bedmap2", "bedmap3"), local_cache=True):
    """Kept BedMap points (lon, lat, source_file, row) sorted along track."""
    from xopr.bedmap import fetch_bedmap, query_bedmap
    fetch_bedmap()
    df = query_bedmap(collections=list(collections),
                      columns=["lon", "lat", "source_file", "row"],
                      local_cache=local_cache, show_progress=True)
    n_files = df["source_file"].nunique()
    df = df[df["source_file"].isin(kept_source_files(collections))]
    print(f"  {len(df)} BedMap points from {df['source_file'].nunique()} files "
          f"({n_files - df['source_file'].nunique()} excluded or duplicate files dropped)")
    return df.sort_values(["source_file", "row"])


def point_segments(df, epsg="EPSG:3031", max_point_spacing_m=MAX_POINT_SPACING_M):
    """Segments between consecutive points of one file, shorter than the gap limit.

    Returns x1, y1, x2, y2 in ``epsg`` plus the segment lengths in geodesic
    km and the source_file of each segment.
    """
    tf = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    lon, lat = df["lon"].to_numpy(), df["lat"].to_numpy()
    xs, ys = tf.transform(lon, lat)
    same_file = df["source_file"].to_numpy()[:-1] == df["source_file"].to_numpy()[1:]
    dists = np.hypot(np.diff(xs), np.diff(ys))
    ok = same_file & (dists < max_point_spacing_m) & (dists > 0)
    km = _GEOD.inv(lon[:-1][ok], lat[:-1][ok], lon[1:][ok], lat[1:][ok])[2] / 1000.0
    return xs[:-1][ok], ys[:-1][ok], xs[1:][ok], ys[1:][ok], km, df["source_file"].to_numpy()[:-1][ok]


def campaign_point_km(rebuild=False):
    """Gap-filtered line-km per BedMap file, from ``bedmap_campaign_km.csv``.

    The table is rebuilt from the point data (all three BedMap versions,
    ~90M points, needs the point cache) when missing or ``rebuild`` is set;
    run ``python bedmap_common.py`` to refresh it after changing the rules.
    """
    if CAMPAIGN_KM_FILE.exists() and not rebuild:
        return pd.read_csv(CAMPAIGN_KM_FILE, index_col="name")["line_km"]
    df = load_bedmap_points()
    *_, seg_km, files = point_segments(df)
    km = pd.Series(seg_km).groupby(files).sum().rename("line_km").rename_axis("name")
    km.to_csv(CAMPAIGN_KM_FILE)
    return km


def dropped_names():
    """BedMap file names removed everywhere (excluded, superseded, substituted)."""
    subst = {n for names in OPR_SUBSTITUTIONS.values() for n in names}
    return EXCLUDE_NAMES | SUPERSEDED_NAMES | subst


def campaign_years(row):
    """Years a campaign row contributes to.

    Rows from dated sources carry a ``season`` (July-June start year) and go
    entirely into that year; BedMap rows are spread over the calendar years
    of their ``ts``..``te`` metadata range (existing convention).
    """
    season = row.get("season") if hasattr(row, "get") else None
    if season is not None and not pd.isna(season):
        return [int(season)]
    return list(range(row["ts"].year, row["te"].year + 1))


def kept_source_files(collections=("bedmap1", "bedmap2", "bedmap3")):
    """Point-data ``source_file`` values to keep after exclusion and dedup.

    ``source_file`` in the point parquet files matches the catalog ``name``.
    """
    return set(_catalog_names(collections))


def _catalog_names(collections):
    """Kept catalog names (exclusions + version dedup) without the km table."""
    cat = query_bedmap_catalog(collections=list(collections))
    df = pd.DataFrame({"name": [p["name"] for p in cat["properties"]]})
    df = df[~df["name"].isin(dropped_names())
            & ~df["name"].str.startswith(EXTERNAL_SOURCE_PREFIXES)]
    df["base"] = df["name"].str.replace(_VERSION_SUFFIX, "", regex=True)
    return df.sort_values("name").drop_duplicates("base", keep="last")["name"]


def geod_km(geometry):
    """Geodesic length in km of a shapely (Multi)LineString in lon/lat."""
    if geometry is None or geometry.is_empty:
        return 0.0
    total = 0.0
    for line in (geometry.geoms if hasattr(geometry, "geoms") else [geometry]):
        c = np.asarray(line.coords)
        if len(c) > 1:
            total += _GEOD.inv(c[:-1, 0], c[:-1, 1], c[1:, 0], c[1:, 1])[2].sum()
    return total / 1000.0


if __name__ == "__main__":
    km = campaign_point_km(rebuild=True)
    print(f"wrote {CAMPAIGN_KM_FILE}: {len(km)} files, {km.sum():,.0f} km")
