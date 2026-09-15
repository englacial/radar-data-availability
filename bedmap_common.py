"""Shared BedMap catalog handling.

Every BedMap consumer (bar charts and the point-data density loader) should
go through here so that campaign exclusions, BM2/BM3 deduplication, and
temporal metadata fixes are applied in exactly one place.
"""

import re

import numpy as np
import pandas as pd
from pyproj import Geod
from xopr.bedmap.query import query_bedmap_catalog

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

_VERSION_SUFFIX = re.compile(r"_BM[123]$")
_GEOD = Geod(ellps="WGS84")


def load_bedmap_catalog(collections=("bedmap2", "bedmap3"), exclude=True):
    """Return catalog entries with one row per campaign.

    Applies ``EXCLUDE_NAMES``, ``SUPERSEDED_NAMES``, ``OPR_SUBSTITUTIONS`` and
    ``EXTERNAL_SOURCE_PREFIXES``, keeps the newest BedMap version when a
    campaign appears in several (BM3 over BM2 over BM1), and parses temporal
    metadata into ``ts``/``te``. A missing or sentinel end date (e.g. year
    9999) falls back to the start date.

    Columns: collection, geometry, name, base_name, institution,
    temporal_start, temporal_end, ts, te.
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
    return df.reset_index(drop=True)


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
    return set(load_bedmap_catalog(collections)["name"])


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
