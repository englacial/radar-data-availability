#!/usr/bin/env python3
"""Combined figure: BedMap vs xOPR data availability.

For 2000-2019, xOPR data is a subset of BedMap. For 2020+, only xOPR exists.
Three categories: "Open access to raw data" (xOPR), "Commitment to release"
(AWI all years + UTIG 2008+), and "Raw data not released" (remainder).
"""

import argparse
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xopr
from pyproj import Geod
from shapely import wkt

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--haps", action="store_true",
               help="Show HAPS Potential bar")
region = p.add_mutually_exclusive_group()
region.add_argument("--antarctica", action="store_true",
                    help="Only plot Antarctic data")
region.add_argument("--greenland", action="store_true",
                    help="Only plot Greenland data")
args = p.parse_args()

SCRIPT_DIR = Path(__file__).parent
OUT_DIR = SCRIPT_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# Excluded BedMap datasets (scattered points, not reliable flight lines)
EXCLUDE_NAMES = {
    "RNRF_2008_Vostok-Subglacial-Lake_AIR_BM2",
    "CRESIS_2009_Thwaites_AIR_BM3",
}


def geod_km_wkt(geometry_wkt):
    """Geodesic length of a WKT geometry in km."""
    geom = wkt.loads(geometry_wkt)
    geod = Geod(ellps="WGS84")
    total = 0.0
    lines = geom.geoms if hasattr(geom, "geoms") else [geom]
    for line in lines:
        coords = list(line.coords)
        for i in range(len(coords) - 1):
            _, _, d = geod.inv(coords[i][0], coords[i][1],
                               coords[i + 1][0], coords[i + 1][1])
            total += d
    return total / 1000.0


def geod_km(geometry):
    """Geodesic length of a shapely geometry in km."""
    if geometry is None or geometry.is_empty:
        return 0.0
    geod = Geod(ellps="WGS84")
    coords = list(geometry.coords)
    total = 0.0
    for i in range(len(coords) - 1):
        _, _, d = geod.inv(coords[i][0], coords[i][1],
                           coords[i + 1][0], coords[i + 1][1])
        total += d
    return total / 1000.0


# --- BedMap line-km per year (2000-2020) --- (Antarctic only, skip for Greenland)
if not args.greenland:
    print("Querying BedMap catalogs...")
    conn = duckdb.connect()
    conn.execute("INSTALL spatial; LOAD spatial; SET enable_progress_bar = false;")
    urls = [f"https://data.source.coop/englacial/bedmap/bedmap{v}.parquet" for v in [2, 3]]
    query = " UNION ALL ".join(
        f"SELECT ST_AsText(geometry) as geom_wkt, name, "
        f"temporal_start, temporal_end FROM read_parquet('{u}')" for u in urls
    )
    bm = conn.execute(query).fetchdf()
    conn.close()

    bm = bm[~bm["name"].isin(EXCLUDE_NAMES)]
    bm["base_name"] = bm["name"].str.replace(r"_BM[123]$", "", regex=True)
    bm = bm.sort_values("name").drop_duplicates(subset="base_name", keep="last")
    bm["line_km"] = bm["geom_wkt"].apply(geod_km_wkt)
    bm["ts"] = pd.to_datetime(bm["temporal_start"], format="ISO8601")
    bm["te"] = pd.to_datetime(bm["temporal_end"], format="ISO8601", errors="coerce")
    bm["te"] = bm["te"].fillna(bm["ts"])

    bm_rows = []
    for _, r in bm.iterrows():
        y_start, y_end = r["ts"].year, r["te"].year
        n_years = y_end - y_start + 1
        prefix = r["name"].split("_")[0]
        committed = prefix == "AWI" or (prefix == "UTIG" and y_start >= 2008)
        for y in range(y_start, y_end + 1):
            bm_rows.append({"year": y, "line_km": r["line_km"] / n_years,
                             "committed": committed})

    bm_df = pd.DataFrame(bm_rows)
    bedmap_yearly = (bm_df.groupby("year")["line_km"].sum()
                     .reindex(range(2001, 2024), fill_value=0))
    committed_yearly = (bm_df[bm_df["committed"]].groupby("year")["line_km"].sum()
                        .reindex(range(2001, 2024), fill_value=0))
else:
    bedmap_yearly = pd.Series(0, index=range(2001, 2024))
    committed_yearly = pd.Series(0, index=range(2001, 2024))

# --- xOPR line-km per year (Antarctic + Greenland) ---
print("Querying xOPR catalog...")
opr_conn = xopr.OPRConnection()
collections = opr_conn.get_collections()
all_ids = sorted(c["id"] for c in collections)

opr_rows = []
greenland_rows = []
for cid in all_ids:
    is_greenland = "Greenland" in cid
    if args.antarctica and is_greenland:
        continue
    if args.greenland and not is_greenland:
        continue
    year = int(cid.split("_")[0])
    items = opr_conn.query_frames(collections=[cid], exclude_geometry=False)
    if items is None or len(items) == 0:
        continue
    total_km = items["geometry"].apply(geod_km).sum()
    opr_rows.append({"year": year, "line_km": total_km})
    if is_greenland:
        greenland_rows.append({"year": year, "line_km": total_km})
    print(f"  {cid}: {total_km:.0f} km")

def _yearly_sum(rows):
    if not rows:
        return pd.Series(0, index=range(2001, 2024), dtype=float)
    return (pd.DataFrame(rows).groupby("year")["line_km"].sum()
            .reindex(range(2001, 2024), fill_value=0))

opr_yearly = _yearly_sum(opr_rows)
# Greenland xOPR data is not in BedMap, so add it to the BedMap total
greenland_yearly = _yearly_sum(greenland_rows)

# --- Combine: open access / commitment / not released ---
# Greenland xOPR is additional to BedMap (Antarctic only), so add it to totals
years = np.arange(2001, 2024)
open_km = opr_yearly.values
total_km = np.maximum(bedmap_yearly.values + greenland_yearly.values, open_km)
# Committed data not yet in xOPR (cap at remaining BedMap after removing xOPR)
remaining = total_km - open_km
commit_km = np.minimum(committed_yearly.values, remaining)
not_released = remaining - commit_km

# --- Summary statistics ---
nonzero_mask = total_km > 0
if nonzero_mask.any():
    avg_km = total_km[nonzero_mask].mean()
    max_km = total_km.max()
    max_year = years[total_km.argmax()]
    region_label = "Greenland" if args.greenland else "Antarctica" if args.antarctica else "All regions"
    print(f"\n{region_label} summary:")
    print(f"  Total line-km (all years): {total_km.sum():,.0f}")
    print(f"    Open access:          {open_km.sum():,.0f}")
    print(f"    Commitment to release: {commit_km.sum():,.0f}")
    print(f"    No access:            {not_released.sum():,.0f}")
    print(f"  Average line-km/year (non-zero years): {avg_km:,.0f}")
    print(f"  Maximum line-km in a single year: {max_km:,.0f} ({max_year})")

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(years))
ax.bar(x, open_km, width=0.8, label="Open access to raw data through xOPR",
       color="tab:blue", alpha=0.85)
ax.bar(x, commit_km, width=0.8, label="Commitment to release data",
       color="lightsteelblue", alpha=0.85, bottom=open_km)
ax.bar(x, not_released, width=0.8, label="No access to raw data",
       color="lightgray", alpha=0.85, bottom=open_km + commit_km)

if args.haps:
    future_x = len(years) + 1  # gap of one bar width
    haps_km = 150000 if (args.antarctica or args.greenland) else 2 * 150000
    ax.bar(future_x, haps_km, width=0.8, color="white",
           edgecolor="tab:blue", linewidth=2, linestyle="--")
    all_x = np.append(x, future_x)
    all_labels = list(years) + ["HAPS\nPotential"]
else:
    all_x = x
    all_labels = list(years)
ax.set_xticks(all_x)
ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=14)
ax.tick_params(axis="y", labelsize=14)
#ax.set_title("Antarctic airborne radar data availability", fontsize=22)
ax.set_xlabel("year", fontsize=18)
ax.set_ylabel("Line-km of radar sounder data", fontsize=18)
ax.legend(fontsize=15, loc="upper left")
plt.tight_layout()
suffix = "_greenland" if args.greenland else "_antarctica" if args.antarctica else ""
out_path = OUT_DIR / f"combined_data_availability{suffix}.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved to {out_path}")
