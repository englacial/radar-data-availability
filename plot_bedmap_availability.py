#!/usr/bin/env python3
"""Stacked bar chart of BedMap line-km per year, colored by country.

Recreates the reference figure from the BedMap STAC catalog data.
Uses campaign name years from temporal_start/temporal_end metadata,
distributing multi-year campaigns evenly across their span.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from bedmap_common import geod_km, load_bedmap_catalog

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


def institution_to_country(name):
    """Map campaign name prefix to country."""
    prefix = name.split("_")[0]
    return COUNTRY_MAP.get(prefix, "Other")


# Query bedmap2 and bedmap3 catalogs (skip bedmap1, matching reference).
# Exclusions, BM2/BM3 dedup and date parsing live in bedmap_common.
df = load_bedmap_catalog(["bedmap2", "bedmap3"])
df["line_km"] = df["geometry"].apply(geod_km)

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
