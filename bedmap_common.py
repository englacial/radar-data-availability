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

_VERSION_SUFFIX = re.compile(r"_BM[123]$")
_GEOD = Geod(ellps="WGS84")


def load_bedmap_catalog(collections=("bedmap2", "bedmap3"), exclude=True):
    """Return catalog entries with one row per campaign.

    Applies ``EXCLUDE_NAMES``, keeps the newest BedMap version when a campaign
    appears in several (BM3 over BM2 over BM1), and parses temporal metadata
    into ``ts``/``te``. A missing or sentinel end date (e.g. year 9999) falls
    back to the start date.

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
        df = df[~df["name"].isin(EXCLUDE_NAMES)]
    df["base_name"] = df["name"].str.replace(_VERSION_SUFFIX, "", regex=True)
    # Suffixes sort BM1 < BM2 < BM3, so keep="last" keeps the newest version.
    df = df.sort_values("name").drop_duplicates("base_name", keep="last")

    df["ts"] = pd.to_datetime(df["temporal_start"], format="ISO8601", errors="coerce")
    te = pd.to_datetime(df["temporal_end"], format="ISO8601", errors="coerce")
    bad_end = te.isna() | (te.dt.year > 2100)
    df["te"] = te.where(~bad_end, df["ts"])
    return df.reset_index(drop=True)


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
