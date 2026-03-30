#!/usr/bin/env python3
"""Stacked bar chart of BedMap line-km per year, colored by country.

Recreates the reference figure from the BedMap STAC catalog data.
Uses campaign name years from temporal_start/temporal_end metadata,
distributing multi-year campaigns evenly across their span.
"""

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
from pyproj import Geod
from shapely import wkt

SCRIPT_DIR = Path(__file__).parent
OUT_DIR = SCRIPT_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# Institution prefix → country (matching reference notebook mapping)
# Only major national programs are mapped; everything else → "Other"
COUNTRY_MAP = {
    "NASA": "USA", "CRESIS": "USA", "UTIG": "USA", "LDEO": "USA",
    "BAS": "UK",
    "AWI": "Germany",
    "RNRF": "Russia",
    "PRIC": "China",
}
COUNTRY_COLORS = {
    "USA": "tab:blue", "UK": "tab:green", "Germany": "tab:orange",
    "Russia": "tab:red", "China": "tab:purple", "Other": "gray",
}
COUNTRY_ORDER = ["Other", "China", "Russia", "Germany", "UK", "USA"]

# Excluded datasets (scattered points, not reliable flight lines)
EXCLUDE_NAMES = {
    "RNRF_2008_Vostok-Subglacial-Lake_AIR_BM2",
    "CRESIS_2009_Thwaites_AIR_BM3",
}


def institution_to_country(name):
    """Map campaign name prefix to country."""
    prefix = name.split("_")[0]
    return COUNTRY_MAP.get(prefix, "Other")


def line_km(geometry_wkt):
    """Calculate geodesic length of a WKT line geometry in km."""
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


# Query bedmap2 and bedmap3 catalogs (skip bedmap1, matching reference)
conn = duckdb.connect()
conn.execute("INSTALL spatial; LOAD spatial; SET enable_progress_bar = false;")
urls = [f"https://data.source.coop/englacial/bedmap/bedmap{v}.parquet" for v in [2, 3]]
query = " UNION ALL ".join(
    f"SELECT ST_AsText(geometry) as geom_wkt, name, "
    f"temporal_start, temporal_end FROM read_parquet('{u}')" for u in urls
)
df = conn.execute(query).fetchdf()
conn.close()

# Exclude problematic datasets
df = df[~df["name"].isin(EXCLUDE_NAMES)]

# Deduplicate: keep BM3 over BM2 for campaigns in both catalogs
df["base_name"] = df["name"].str.replace(r"_BM[123]$", "", regex=True)
df = df.sort_values("name").drop_duplicates(subset="base_name", keep="last")

# Calculate line-km
df["line_km"] = df["geom_wkt"].apply(line_km)
df["ts"] = pd.to_datetime(df["temporal_start"], format="ISO8601")
df["te"] = pd.to_datetime(df["temporal_end"], format="ISO8601")

# Distribute every campaign evenly across its year range (matching reference)
rows = []
for _, r in df.iterrows():
    y_start, y_end = r["ts"].year, r["te"].year
    n_years = y_end - y_start + 1
    country = institution_to_country(r["name"])
    for y in range(y_start, y_end + 1):
        rows.append({"year": y, "line_km": r["line_km"] / n_years,
                     "country": country})

result = pd.DataFrame(rows)
result = result[(result["year"] >= 2000) & (result["year"] <= 2020)]

# Pivot: line-km per year per country, ensure all years present
pivot = result.pivot_table(index="year", columns="country", values="line_km",
                           aggfunc="sum", fill_value=0)
pivot = pivot.reindex(range(2000, 2021), fill_value=0)
pivot = pivot.reindex(columns=[c for c in COUNTRY_ORDER if c in pivot.columns])

# Plot
fig, ax = plt.subplots(figsize=(14, 7))
pivot.plot.bar(stacked=True, ax=ax,
               color=[COUNTRY_COLORS[c] for c in pivot.columns],
               width=0.8, edgecolor="none")

# HAPS reference line
haps_km = 153000
ax.axhline(haps_km, color="red", linestyle="--", linewidth=2)
ax.annotate("Capability of 1 HAPS UAV, 11 week mission",
            xy=(0.35, haps_km + 2000), xycoords=("axes fraction", "data"),
            fontsize=14, color="red", fontweight="bold", ha="center")

ax.set_title("Line-km of global Antarctic airborne radar surveying", fontsize=16)
ax.set_xlabel("year", fontsize=13)
ax.set_ylabel("IPR surveying flight kilometers", fontsize=13)
ax.legend(title="Country", fontsize=11, title_fontsize=12,
          loc="upper right", framealpha=0.9)
plt.tight_layout()
out_path = OUT_DIR / "bedmap_data_availability.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved to {out_path}")
