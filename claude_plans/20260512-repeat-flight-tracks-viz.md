# Repeat Flight Tracks Visualization

Interactive plot of repeated flight paths in Greenland and/or Antarctica. Identify
clusters where multiple flights follow roughly the same path, visualize the
clusters on a map, and show flight altitudes along the shared corridor.

## Inputs

- xOPR STAC catalog (LineString geometries are 2D: lon/lat only).
- Per-frame data files (loaded on demand for altitude only).

## Pipeline

Two offline cached stages followed by an interactive viewer. Cache to parquet so
the viewer loads pre-computed results.

### Stage 1 — Match flight paths (x/y only)

1. Pull STAC items for a region (start with one season or one ice sheet).
2. Densify each line to evenly-spaced points (~200–500 m spacing) in polar
   stereographic projection (EPSG:3031 Antarctica, EPSG:3413 Greenland).
3. Build a KDTree over all densified points (tagged by item ID).
4. For each line, compute the fraction of its points within `tol` (default 500 m)
   of a point from a *different* item. Pairwise: count overlap points per (A, B)
   pair, normalize by length.
5. Build similarity graph: edge(A, B) if overlap length > `min_shared_km`
   (default 20 km).
6. Cluster via connected components.

Output: `outputs/clusters.parquet` with columns
`{item_id, cluster_id, overlap_length_km, ...}` plus the densified geometries
keyed by item_id.

### Stage 2 — Extract altitudes for matched frames

Only for frames participating in a cluster:

1. Load each frame via xOPR's frame loader (e.g., `OPRConnection.load_frame_url`).
2. Extract `Longitude`, `Latitude`, `Elevation` (or aircraft altitude variable),
   and `slow_time` at trace resolution.
3. Cache to `outputs/altitudes.parquet` keyed by item_id.

### Stage 3 — Interactive viewer

GeoViews + Bokeh + Panel. Three coordinated panels.

**Panel A — Overview map**
- Polar stereographic basemap (cartopy), coastline + grounding line.
- All cluster lines plotted, colored by cluster_id.
- Faint cluster hull underneath (convex hull or buffered union).
- Hover: cluster_id, # tracks in cluster, total repeat length.
- Click cluster → drives Panels B and C.

**Panel B — Selected-cluster detail map**
- Zoomed to selected cluster bounds.
- Each track drawn as its own colored line (consistent colors used in Panel C).
- Hover tooltip per track: date, segment id, season.

**Panel C — Altitude along corridor**
- For the selected cluster:
  - Pick a reference line (longest member or medoid).
  - For each member track, restrict to points within `tol` of the reference,
    project them onto the reference to get a 1D arc-length coordinate.
  - Plot altitude vs arc-length, one curve per track, colors matching Panel B.
- Lets the user see e.g. low-altitude vs high-altitude survey passes over the
  same corridor.

Optional Panel sidebar: cluster selector (sorted by # tracks or repeat length),
parameter sliders for `tol` and `min_shared_km` if we want re-clustering live
(probably not — keep clustering offline).

## Parameters (defaults)

- `tol` = 500 m (point-to-point match tolerance in polar projection)
- `min_shared_km` = 20 km (minimum overlap length for an edge in the graph)
- `densify_spacing` = 250 m
- Region scope: start with one ice sheet (Antarctica) and a small year range to
  iterate.

## Files

- `repeat_pass_map/match_flights.py` — Stage 1; writes `outputs/{flight_points,subpaths,corridors,flights}.parquet`.
- `repeat_pass_map/extract_altitudes.py` — Stage 2; CSARP_layer only, writes `outputs/altitudes/*.parquet` cache + `outputs/altitudes.parquet`.
- `repeat_pass_map/viewer.py` — Stage 3; `uv run panel serve repeat_pass_map/viewer.py --show`.

## Tradeoffs / open questions

- Frechet/Hausdorff distance rejected: too sensitive to differing line lengths.
- Densification spacing vs memory: at 250 m a full Antarctic survey is large but
  tractable. May need to chunk by ice sheet or year.
- Direction-agnostic matching falls out naturally because we use point-set
  overlap rather than ordered path comparison.
- "Reference line" choice for Panel C: longest member is simplest; medoid is
  more representative but pricier. Start with longest.
