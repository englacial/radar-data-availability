# repeat_pass_map

Identifies repeated flight tracks ("corridors") in the xOPR catalog and serves
an interactive 3-panel viewer (overview map, per-corridor detail map,
altitude-along-corridor profile).

All outputs land in `../outputs/`.

## Build the data (run from repo root)

```bash
# 1. Match flight paths and stitch corridors (~2 min)
uv run python repeat_pass_map/match_flights.py

# 2. Fetch per-trace altitudes for corridor members (~60–90 min)
uv run python repeat_pass_map/extract_altitudes.py
```

`match_flights.py` writes `flights.parquet`, `flight_points.parquet`,
`subpaths.parquet`, `corridors.parquet`.

`extract_altitudes.py` reads only the small CSARP_layer assets and writes
`altitudes/*.parquet` (per-frame cache) plus the concatenated `altitudes.parquet`.
Defaults: Greenland, min 5 flights per corridor, 8 worker threads. Frames without
a CSARP_layer asset are skipped. Re-runs reuse the cache.

## Launch the viewer

```bash
uv run panel serve repeat_pass_map/viewer.py --show
```

Sidebar controls: min flights / min total length filters, corridor selector,
single + bulk KML download. Clicking a point on the overview map selects that
corridor.
