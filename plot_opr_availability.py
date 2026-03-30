#!/usr/bin/env python3
"""Stacked bar chart of xOPR Antarctic line-km per year, colored by institution."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import xopr
from pyproj import Geod

SCRIPT_DIR = Path(__file__).parent
OUT_DIR = SCRIPT_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

PROVIDER_LABELS = {"cresis": "CReSIS", "utig": "UTIG"}
PROVIDER_COLORS = {"CReSIS": "tab:blue", "UTIG": "tab:orange"}


def line_km(geometry):
    """Calculate geodesic length of a geometry in km."""
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


# Query all Antarctic frames
opr_conn = xopr.OPRConnection()
collections = opr_conn.get_collections()
antarctic_ids = sorted(c["id"] for c in collections if "Antarctica" in c["id"])

rows = []
for cid in antarctic_ids:
    year = int(cid.split("_")[0])
    items = opr_conn.query_frames(collections=[cid], exclude_geometry=False)
    if items is None or len(items) == 0:
        continue
    provider = items.iloc[0]["properties"].get("opr:provider", "unknown")
    label = PROVIDER_LABELS.get(provider, provider)
    total_km = items["geometry"].apply(line_km).sum()
    rows.append({"year": year, "institution": label, "line_km": total_km})
    print(f"  {cid}: {total_km:.0f} km ({label})")

df = pd.DataFrame(rows)

# Pivot and plot
pivot = df.pivot_table(index="year", columns="institution", values="line_km",
                       aggfunc="sum", fill_value=0)
pivot = pivot.reindex(columns=[c for c in PROVIDER_COLORS if c in pivot.columns])

fig, ax = plt.subplots(figsize=(12, 6))
pivot.plot.bar(stacked=True, ax=ax,
               color=[PROVIDER_COLORS[c] for c in pivot.columns],
               width=0.8, edgecolor="none")

ax.set_title("xOPR Antarctic radar line-km per season year", fontsize=16)
ax.set_xlabel("year", fontsize=13)
ax.set_ylabel("line-km", fontsize=13)
ax.legend(title="Institution", fontsize=11, title_fontsize=12, loc="upper left")
plt.tight_layout()
out_path = OUT_DIR / "opr_data_availability.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved to {out_path}")
