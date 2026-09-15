#!/usr/bin/env python3
"""Stacked bar chart of BedMap line-km per year, colored by country.

Recreates the reference figure from the BedMap STAC catalog data, plus the
direct sources in extra_sources (AWI tracks, KOPRI helicopter surveys, xOPR
substitutes). BedMap campaigns are spread evenly over their
temporal_start..temporal_end calendar years; dated sources go to their season.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from bedmap_common import campaign_years, load_bedmap_catalog
from extra_sources import load_extra_campaigns

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--no-haps", action="store_true", help="Omit the HAPS capability line")
args = p.parse_args()

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
# Exclusions, dedup, date parsing and gap-filtered line-km live in bedmap_common.
df = load_bedmap_catalog(["bedmap2", "bedmap3"])
df = pd.concat([df, load_extra_campaigns()], ignore_index=True)

# Distribute every campaign evenly across its years (see campaign_years)
rows = []
for _, r in df.iterrows():
    years = campaign_years(r)
    country = institution_to_country(r["name"])
    for y in years:
        rows.append({"year": y, "line_km": r["line_km"] / len(years),
                     "country": country})

result = pd.DataFrame(rows)
result = result[(result["year"] >= 2000) & (result["year"] <= 2025)]

# Pivot: line-km per year per country, ensure all years present
pivot = result.pivot_table(index="year", columns="country", values="line_km",
                           aggfunc="sum", fill_value=0)
pivot = pivot.reindex(range(2000, 2026), fill_value=0)
pivot = pivot.reindex(columns=[c for c in COUNTRY_ORDER if c in pivot.columns])

# Plot (thousands of km)
fig, ax = plt.subplots(figsize=(14, 7))
(pivot / 1000).plot.bar(stacked=True, ax=ax,
                        color=[COUNTRY_COLORS[c] for c in pivot.columns],
                        width=0.8, edgecolor="none")

if not args.no_haps:
    haps_km = 153
    ax.axhline(haps_km, color="red", linestyle="--", linewidth=2)
    ax.annotate("Capability of 1 HAPS UAV, 11 week mission",
                xy=(0.35, haps_km + 2), xycoords=("axes fraction", "data"),
                fontsize=16, color="red", fontweight="bold", ha="center")

ax.set_title("Line-km of global Antarctic airborne radar surveying", fontsize=20)
ax.set_xlabel("year", fontsize=17)
ax.set_ylabel("IPR surveying flight kilometers (thousands)", fontsize=17)
ax.tick_params(axis="both", labelsize=14)
ax.legend(title="Country", fontsize=14, title_fontsize=15,
          loc="upper right", framealpha=0.9)
plt.tight_layout()
suffix = "_nohaps" if args.no_haps else ""
out_path = OUT_DIR / f"bedmap_data_availability{suffix}.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved to {out_path}")
