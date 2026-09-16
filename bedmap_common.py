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


def bedmap_segments(epsg="EPSG:3031", collections=("bedmap1", "bedmap2", "bedmap3"),
                    max_point_spacing_m=MAX_POINT_SPACING_M, with_km=False):
    """Gap-filtered flight-line segments of all kept BedMap files.

    Files are fetched to ``radar_cache/bedmap/`` if missing and processed one
    at a time (peak memory is one file plus the float32 output, not the 80M
    point table). Returns ``(x1, y1, x2, y2)`` in ``epsg`` and, if
    ``with_km``, a Series of geodesic km per file.
    """
    import pyarrow.parquet as pq
    import shapely
    from xopr.bedmap import fetch_bedmap
    paths = sorted(p for v, ps in fetch_bedmap().items() if v in collections for p in ps)
    keep = kept_source_files(collections)
    tf = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    parts, km, n_pts, n_files = [], {}, 0, 0
    for path in paths:
        name = Path(path).stem
        if name not in keep:
            continue
        t = pq.read_table(path, columns=["geometry", "row"]).sort_by("row")
        c = shapely.get_coordinates(shapely.from_wkb(t.column("geometry").to_numpy(zero_copy_only=False)))
        lon, lat = c[:, 0], c[:, 1]
        xs, ys = tf.transform(lon, lat)
        d = np.hypot(np.diff(xs), np.diff(ys))
        ok = (d < max_point_spacing_m) & (d > 0)
        parts.append(np.column_stack([xs[:-1][ok], ys[:-1][ok], xs[1:][ok], ys[1:][ok]]).astype(np.float32))
        if with_km:
            km[name] = _GEOD.inv(lon[:-1][ok], lat[:-1][ok], lon[1:][ok], lat[1:][ok])[2].sum() / 1000.0
        n_pts += len(lon); n_files += 1
    seg = np.concatenate(parts)
    print(f"  {n_pts} BedMap points from {n_files} files ({len(paths) - n_files} excluded or duplicate files dropped)")
    return tuple(seg[:, i] for i in range(4)), pd.Series(km, name="line_km").rename_axis("name")


def campaign_point_km(rebuild=False):
    """Gap-filtered line-km per BedMap file, from ``bedmap_campaign_km.csv``.

    The table is rebuilt from the point data (all three BedMap versions,
    ~90M points, fetched to the point cache) when missing or ``rebuild`` is set;
    run ``python bedmap_common.py`` to refresh it after changing the rules.
    """
    if CAMPAIGN_KM_FILE.exists() and not rebuild:
        return pd.read_csv(CAMPAIGN_KM_FILE, index_col="name")["line_km"]
    _, km = bedmap_segments(with_km=True)
    km = km.sort_index()
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
