#!/usr/bin/env python3
"""Run coastal_survey_gaps.py across all parameter combinations and summarize."""

import itertools
import subprocess
import sys
from pathlib import Path

OUT_DIR = Path(__file__).parent / "outputs" / "coastal_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

coast_dists = [10, 30, 50]
grid_sizes = [10, 30, 50]
targets = [0.5, 1, 2]

results = []
total = len(coast_dists) * len(grid_sizes) * len(targets)

for i, (cd, gs, tg) in enumerate(
    itertools.product(coast_dists, grid_sizes, targets), 1
):
    print(f"\n[{i}/{total}] coast={cd}km grid={gs}km target={tg}km")
    proc = subprocess.run(
        [
            sys.executable, "coastal_survey_gaps.py",
            "--source", "bedmap_local",
            "--coast-dist-km", str(cd),
            "--grid-km", str(gs),
            "--target-km", str(tg),
        ],
        capture_output=True, text=True,
    )
    print(proc.stdout[-300:] if len(proc.stdout) > 300 else proc.stdout)
    if proc.returncode != 0:
        print("STDERR:", proc.stderr[-500:])
        continue

    # Move the output plot into sweep subfolder
    default_name = f"coastal_density_bedmap_local_{cd:.0f}km.png"
    src = Path("outputs") / default_name
    dst = OUT_DIR / f"coast{cd}_grid{gs}_target{tg}.png"
    if src.exists():
        src.rename(dst)
        print(f"  -> {dst}")

    # Parse total additional line-km from stdout
    for line in proc.stdout.splitlines():
        if "Total additional line-km needed" in line:
            val = line.split(":")[-1].strip().replace(",", "")
            results.append({
                "coast_km": cd, "grid_km": gs, "target_km": tg,
                "additional_line_km": int(float(val)),
            })
            break

# Print summary table
print("\n" + "=" * 70)
print("SUMMARY: Additional line-km needed")
print("=" * 70)
header = f"{'Coast km':>10} {'Grid km':>10} {'Target km':>10} {'Add. line-km':>15}"
print(header)
print("-" * len(header))
for r in results:
    print(f"{r['coast_km']:>10} {r['grid_km']:>10} {r['target_km']:>10} "
          f"{r['additional_line_km']:>15,}")

# Also write CSV
csv_path = OUT_DIR / "sweep_results.csv"
with open(csv_path, "w") as f:
    f.write("coast_dist_km,grid_km,target_km,additional_line_km\n")
    for r in results:
        f.write(f"{r['coast_km']},{r['grid_km']},{r['target_km']},"
                f"{r['additional_line_km']}\n")
print(f"\nCSV saved to {csv_path}")
