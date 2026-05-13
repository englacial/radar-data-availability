#!/usr/bin/env python3
"""Match repeated Greenland flight tracks via sub-path clustering.

A sub-path is a contiguous run along one flight where it travels alongside
(within tol, with aligned direction) at least one other flight. Sub-paths longer
than max_len_km are split into fixed-length chunks. Pairs of sub-paths from
different flights are clustered into corridors based on mutual spatial coverage.

Outputs:
  outputs/flights.parquet         one row per flight (segment) with metadata
  outputs/flight_points.parquet   densified points in EPSG:3413 with sub_id
  outputs/subpaths.parquet        one row per sub-path (incl. corridor_id)
  outputs/corridors.parquet       one row per corridor (>=2 sub-paths from >=2 flights)
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely
import xopr
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely.ops import transform

SCRIPT_DIR = Path(__file__).parent
OUT_DIR = SCRIPT_DIR.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

REGION = "Greenland"
EPSG = 3413  # NSIDC Sea Ice Polar Stereographic North

DENSIFY_SPACING_M = 250.0
TOL_M = 500.0
COS_THR = 0.7   # |cos(angle)| > 0.7  ⇒ within ~45° of parallel/anti-parallel
SUBPATH_MAX_LEN_M = 30_000.0   # split long sub-paths so one can't span 2 corridors
SUBPATH_MIN_LEN_M = 5_000.0    # drop very short sub-paths
SUBPATH_MATCH_FRAC = 0.5       # min(frac_A_on_B, frac_B_on_A) >= this  ⇒ edge


# --------------------------------------------------------------------------
# Data loading and densification
# --------------------------------------------------------------------------

def load_region_items(region: str) -> gpd.GeoDataFrame:
    opr = xopr.OPRConnection()
    cols = sorted(c["id"] for c in opr.get_collections() if region in c["id"])
    frames = []
    for cid in cols:
        items = opr.query_frames(collections=[cid], exclude_geometry=False)
        if items is None or len(items) == 0:
            continue
        items = items.copy()
        items["collection"] = cid
        items["segment"] = items["properties"].apply(lambda p: p.get("opr:segment"))
        items["date"] = items["properties"].apply(lambda p: p.get("opr:date"))
        # A flight = single continuous track on a single day (one (date, segment) pair)
        items["flight_id"] = (
            items["collection"]
            + "/"
            + items["date"].astype(str)
            + "_"
            + items["segment"].astype(str)
        )
        frames.append(items)
        print(f"  loaded {cid}: {len(items)} frames")
    gdf = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")


def densify_line(line: shapely.geometry.LineString, spacing: float) -> np.ndarray:
    length = line.length
    if length == 0:
        return np.array([line.coords[0]])
    n = max(2, int(np.ceil(length / spacing)) + 1)
    distances = np.linspace(0, length, n)
    pts = [line.interpolate(d) for d in distances]
    return np.array([(p.x, p.y) for p in pts])


def build_point_table(gdf: gpd.GeoDataFrame, spacing: float) -> pd.DataFrame:
    transformer = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}", always_xy=True)
    rows = []
    for _, item in gdf.iterrows():
        line_wgs84 = item.geometry
        if line_wgs84 is None or line_wgs84.is_empty:
            continue
        line_proj = transform(transformer.transform, line_wgs84)
        pts = densify_line(line_proj, spacing)
        for i, (x, y) in enumerate(pts):
            rows.append({
                "flight_id": item["flight_id"],
                "item_id": item["id"],
                "point_idx_frame": i,
                "x": x,
                "y": y,
            })
    df = pd.DataFrame(rows)
    # Renumber point_idx within each flight (concatenating frames in order)
    df = df.sort_values(["flight_id", "item_id", "point_idx_frame"]).reset_index(drop=True)
    df["point_idx"] = df.groupby("flight_id").cumcount()
    return df


# --------------------------------------------------------------------------
# Direction-aware in-corridor detection
# --------------------------------------------------------------------------

def compute_tangents(points: pd.DataFrame) -> pd.DataFrame:
    """Add tx, ty unit-tangent columns using central differences within each flight."""
    points = points.sort_values(["flight_id", "point_idx"]).reset_index(drop=True)
    xy = points[["x", "y"]].to_numpy()
    flight_ids = points["flight_id"].to_numpy()

    xy_p = np.roll(xy, -1, axis=0); fp = np.roll(flight_ids, -1)
    xy_m = np.roll(xy, 1, axis=0);  fm = np.roll(flight_ids, 1)
    fwd = (fp == flight_ids)
    bwd = (fm == flight_ids)

    dx = np.where(fwd & bwd, xy_p[:, 0] - xy_m[:, 0],
         np.where(fwd, xy_p[:, 0] - xy[:, 0], xy[:, 0] - xy_m[:, 0]))
    dy = np.where(fwd & bwd, xy_p[:, 1] - xy_m[:, 1],
         np.where(fwd, xy_p[:, 1] - xy[:, 1], xy[:, 1] - xy_m[:, 1]))
    norm = np.sqrt(dx * dx + dy * dy) + 1e-9
    points["tx"] = dx / norm
    points["ty"] = dy / norm
    return points


def flag_direction_aware_in_corridor(
    points: pd.DataFrame, tol: float, cos_thr: float
) -> pd.DataFrame:
    """Add in_corridor_dir bool: point has a neighbor on a different flight
    within tol whose tangent is parallel or anti-parallel (|cos| > cos_thr)."""
    xy = points[["x", "y"]].to_numpy()
    flight_ids = points["flight_id"].to_numpy()
    tx = points["tx"].to_numpy()
    ty = points["ty"].to_numpy()
    tree = cKDTree(xy)
    nbrs = tree.query_ball_point(xy, r=tol, workers=-1)
    in_c = np.zeros(len(points), dtype=bool)
    for i, nb in enumerate(nbrs):
        fi = flight_ids[i]
        for j in nb:
            if flight_ids[j] == fi:
                continue
            if abs(tx[i] * tx[j] + ty[i] * ty[j]) > cos_thr:
                in_c[i] = True
                break
    points["in_corridor_dir"] = in_c
    return points


# --------------------------------------------------------------------------
# Sub-path extraction
# --------------------------------------------------------------------------

def build_subpaths(
    points: pd.DataFrame,
    spacing: float,
    max_len_m: float,
    min_len_m: float,
) -> pd.DataFrame:
    """For each flight, find contiguous in_corridor_dir=True runs along the
    track. Split long runs into chunks of <= max_len_m. Drop short runs.

    Returns DataFrame with columns:
      sub_id, flight_id, start_idx, end_idx (exclusive), length_m
    Also mutates `points` in place to add `sub_id` column (-1 where unassigned).
    """
    points["sub_id"] = -1
    rows = []
    sub_id = 0
    max_len_pts = int(np.ceil(max_len_m / spacing))
    min_len_pts = int(np.ceil(min_len_m / spacing))

    points_sorted = points.sort_values(["flight_id", "point_idx"]).reset_index()
    # 'index' column gives back original row in `points`
    for fid, grp in points_sorted.groupby("flight_id", sort=False):
        flag = grp["in_corridor_dir"].to_numpy().astype(np.int8)
        orig_idx = grp["index"].to_numpy()
        # Run starts: 0 -> 1 transitions; ends: 1 -> 0 transitions
        diffs = np.diff(flag, prepend=0, append=0)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        for s, e in zip(starts, ends):
            run_len_pts = e - s
            if run_len_pts < min_len_pts:
                continue
            # Split into chunks of max_len_pts
            cur = s
            while cur < e:
                chunk_end = min(cur + max_len_pts, e)
                # If the *remaining tail* is shorter than min, fold it into
                # this chunk rather than leaving a stub.
                if e - chunk_end < min_len_pts and (chunk_end - cur) + (e - chunk_end) <= max_len_pts * 1.5:
                    chunk_end = e
                length_m = (chunk_end - cur) * spacing
                if chunk_end - cur < min_len_pts:
                    break
                # Assign sub_id to underlying points
                assign_orig = orig_idx[cur:chunk_end]
                points.loc[assign_orig, "sub_id"] = sub_id
                rows.append({
                    "sub_id": sub_id,
                    "flight_id": fid,
                    "start_idx": int(grp.iloc[cur]["point_idx"]),
                    "end_idx": int(grp.iloc[chunk_end - 1]["point_idx"]) + 1,
                    "n_points": int(chunk_end - cur),
                    "length_m": float(length_m),
                })
                sub_id += 1
                cur = chunk_end
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Sub-path clustering into corridors
# --------------------------------------------------------------------------

def cluster_subpaths(
    subpaths: pd.DataFrame,
    points: pd.DataFrame,
    tol: float,
    cos_thr: float,
    min_frac: float,
) -> dict[int, int]:
    """Pairwise compare sub-paths (within bbox-prefiltered candidates), build
    edges where min(frac_A_on_B, frac_B_on_A) >= min_frac, return component IDs.

    A 'frac' is the fraction of A's points that have a B point within tol AND
    direction-aligned. min() makes the match symmetric and requires both to
    largely overlap each other.
    """
    # Pre-extract per-subpath arrays
    sub_data: dict[int, dict] = {}
    for _, row in subpaths.iterrows():
        sid = int(row["sub_id"])
        mask = points["sub_id"] == sid
        sub_pts = points[mask].sort_values("point_idx")
        sub_data[sid] = {
            "xy": sub_pts[["x", "y"]].to_numpy(),
            "tx": sub_pts["tx"].to_numpy(),
            "ty": sub_pts["ty"].to_numpy(),
            "flight_id": row["flight_id"],
        }

    # Bounding boxes for candidate prefilter
    sub_ids = subpaths["sub_id"].to_numpy()
    bboxes = np.zeros((len(sub_ids), 4))
    centroids = np.zeros((len(sub_ids), 2))
    for k, sid in enumerate(sub_ids):
        xy = sub_data[int(sid)]["xy"]
        bboxes[k] = [xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()]
        centroids[k] = xy.mean(axis=0)

    # KDTree over sub-path centroids for fast candidate lookup
    # A sub-path's "radius" = max distance from centroid to one of its endpoints
    radii = np.zeros(len(sub_ids))
    for k, sid in enumerate(sub_ids):
        xy = sub_data[int(sid)]["xy"]
        d = np.linalg.norm(xy - centroids[k], axis=1)
        radii[k] = d.max()
    centroid_tree = cKDTree(centroids)
    max_radius = radii.max()

    print(f"    Pairwise matching {len(sub_ids):,} sub-paths "
          f"(max_radius={max_radius/1000:.0f} km)...")

    edges = []
    for k, sid_a in enumerate(sub_ids):
        # Candidate centroids within radii[k] + max_radius + tol
        search_r = radii[k] + max_radius + tol
        candidate_idxs = centroid_tree.query_ball_point(centroids[k], r=search_r)
        a = sub_data[int(sid_a)]
        n_a = len(a["xy"])
        tree_a = cKDTree(a["xy"])
        for kb in candidate_idxs:
            if kb <= k:
                continue
            sid_b = sub_ids[kb]
            b = sub_data[int(sid_b)]
            if a["flight_id"] == b["flight_id"]:
                continue
            # Tighter bbox prefilter
            if (bboxes[k, 0] - bboxes[kb, 2] > tol or bboxes[kb, 0] - bboxes[k, 2] > tol
                or bboxes[k, 1] - bboxes[kb, 3] > tol or bboxes[kb, 1] - bboxes[k, 3] > tol):
                continue
            n_b = len(b["xy"])
            # frac_A_on_B: # of A points with a B-neighbor within tol AND aligned
            tree_b = cKDTree(b["xy"])
            nbrs_a = tree_b.query_ball_point(a["xy"], r=tol)
            matches_a = 0
            for i, nb in enumerate(nbrs_a):
                for j in nb:
                    if abs(a["tx"][i] * b["tx"][j] + a["ty"][i] * b["ty"][j]) > cos_thr:
                        matches_a += 1
                        break
            frac_a = matches_a / n_a
            if frac_a < min_frac:
                continue
            # frac_B_on_A
            nbrs_b = tree_a.query_ball_point(b["xy"], r=tol)
            matches_b = 0
            for i, nb in enumerate(nbrs_b):
                for j in nb:
                    if abs(b["tx"][i] * a["tx"][j] + b["ty"][i] * a["ty"][j]) > cos_thr:
                        matches_b += 1
                        break
            frac_b = matches_b / n_b
            if min(frac_a, frac_b) >= min_frac:
                edges.append((k, kb))

    print(f"    {len(edges):,} sub-path match edges")

    # Connected components
    n = len(sub_ids)
    if edges:
        edges_arr = np.array(edges)
        rows = np.concatenate([edges_arr[:, 0], edges_arr[:, 1]])
        cols = np.concatenate([edges_arr[:, 1], edges_arr[:, 0]])
        data = np.ones(len(rows), dtype=np.int8)
        g = csr_matrix((data, (rows, cols)), shape=(n, n))
    else:
        g = csr_matrix((n, n))
    _, labels = connected_components(g, directed=False)

    # Reassign so only multi-sub-path / multi-flight corridors get an id
    sid_to_label = {int(sub_ids[i]): int(labels[i]) for i in range(n)}
    # Determine which labels have >=2 distinct flights
    sub_flight = subpaths.set_index("sub_id")["flight_id"].to_dict()
    label_flights: dict[int, set] = {}
    for sid, lab in sid_to_label.items():
        label_flights.setdefault(lab, set()).add(sub_flight[sid])
    valid = {lab for lab, flights in label_flights.items() if len(flights) >= 2}

    label_remap: dict[int, int] = {}
    out: dict[int, int] = {}
    new_id = 0
    for sid, lab in sid_to_label.items():
        if lab in valid:
            if lab not in label_remap:
                label_remap[lab] = new_id
                new_id += 1
            out[sid] = label_remap[lab]
        else:
            out[sid] = -1
    return out


# --------------------------------------------------------------------------
# Post-processing: stitch corridors fragmented by chunk boundaries
# --------------------------------------------------------------------------

def stitch_corridors(subpaths: pd.DataFrame, min_bridges: int = 2,
                     min_frac_smaller: float = 0.6, min_frac_larger: float = 0.3,
                     max_gap: int = 5) -> dict[int, int]:
    """Merge corridors that are consecutive along single flights.

    The 30-km sub-path cap causes corridors to split at chunk boundaries: a long
    physical line is matched as 5+ separate corridors, each covering one 30-km
    stretch. We detect this by looking for single flights whose consecutive
    sub-paths fall in different corridors — those corridors should merge.

    To avoid merging hubs with their spokes (which all share the hub's flights),
    we require the bridging-flight count to clear two thresholds:
      - >= min_frac_smaller * size of the *smaller* corridor
      - >= min_frac_larger  * size of the *larger* corridor
    This passes for genuine fragmentation (most of both corridors' members bridge)
    but fails for hub-spoke situations (only a small fraction of the hub bridges
    to each spoke).
    """
    from collections import defaultdict

    bridges = defaultdict(set)  # (c_lo, c_hi) -> set of flight_ids that bridge them
    for fid, grp in subpaths.groupby("flight_id", sort=False):
        grp = grp.sort_values("start_idx")
        rows = grp[["corridor_id", "start_idx", "end_idx"]].to_numpy()
        for i in range(len(rows) - 1):
            c1, _, e1 = rows[i]
            c2, s2, _ = rows[i + 1]
            c1, c2 = int(c1), int(c2)
            if c1 < 0 or c2 < 0 or c1 == c2:
                continue
            if int(s2) - int(e1) > max_gap:
                continue
            bridges[(min(c1, c2), max(c1, c2))].add(fid)

    corridor_flights = subpaths.groupby("corridor_id")["flight_id"].apply(set).to_dict()

    edges = []
    for (c1, c2), bridging_flights in bridges.items():
        n = len(bridging_flights)
        if n < min_bridges:
            continue
        if c1 not in corridor_flights or c2 not in corridor_flights:
            continue
        size_a = len(corridor_flights[c1])
        size_b = len(corridor_flights[c2])
        sm = min(size_a, size_b)
        lg = max(size_a, size_b)
        if n >= max(min_bridges, int(np.ceil(min_frac_smaller * sm))) \
           and n >= int(np.ceil(min_frac_larger * lg)):
            edges.append((c1, c2))

    # Union-find
    parent: dict[int, int] = {}
    def find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x
    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for c in corridor_flights:
        parent[c] = c
    for c1, c2 in edges:
        union(c1, c2)

    roots = {c: find(c) for c in corridor_flights}
    new_ids = {r: i for i, r in enumerate(sorted(set(roots.values())))}
    return {c: new_ids[r] for c, r in roots.items()}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    print(f"Loading {REGION} items...")
    gdf = load_region_items(REGION)
    print(f"Loaded {len(gdf)} frames across {gdf['flight_id'].nunique()} flights")

    print(f"Densifying lines at {DENSIFY_SPACING_M} m in EPSG:{EPSG}...")
    points = build_point_table(gdf, DENSIFY_SPACING_M)
    print(f"  {len(points):,} points total")

    print("Computing tangents...")
    points = compute_tangents(points)

    print(f"Flagging direction-aware in-corridor points (tol={TOL_M}, |cos|>{COS_THR})...")
    points = flag_direction_aware_in_corridor(points, TOL_M, COS_THR)
    n_in = int(points["in_corridor_dir"].sum())
    print(f"  {n_in:,} / {len(points):,} ({100*n_in/len(points):.1f}%) in-corridor")

    print(f"Building sub-paths (max {SUBPATH_MAX_LEN_M/1000:.0f} km, "
          f"min {SUBPATH_MIN_LEN_M/1000:.0f} km)...")
    subpaths = build_subpaths(points, DENSIFY_SPACING_M, SUBPATH_MAX_LEN_M, SUBPATH_MIN_LEN_M)
    print(f"  {len(subpaths):,} sub-paths from {subpaths['flight_id'].nunique()} flights")
    print(f"  total sub-path length: {subpaths['length_m'].sum()/1e3:.0f} km")

    print(f"Clustering sub-paths (tol={TOL_M}, min_frac={SUBPATH_MATCH_FRAC})...")
    corr_map = cluster_subpaths(subpaths, points, TOL_M, COS_THR, SUBPATH_MATCH_FRAC)
    subpaths["corridor_id"] = subpaths["sub_id"].map(corr_map).fillna(-1).astype(int)
    n_corr = subpaths.loc[subpaths["corridor_id"] >= 0, "corridor_id"].nunique()
    n_assigned = (subpaths["corridor_id"] >= 0).sum()
    print(f"  {n_corr} initial corridors covering {n_assigned}/{len(subpaths)} sub-paths")

    print("Stitching chunk-boundary fragments...")
    stitch_map = stitch_corridors(subpaths)
    subpaths["corridor_id"] = subpaths["corridor_id"].map(
        lambda c: stitch_map.get(int(c), -1) if c >= 0 else -1
    )
    n_corr_after = subpaths.loc[subpaths["corridor_id"] >= 0, "corridor_id"].nunique()
    print(f"  {n_corr} -> {n_corr_after} corridors after stitching "
          f"({n_corr - n_corr_after} merged away)")

    # Corridor summary
    corr_rows = []
    for cid, grp in subpaths[subpaths["corridor_id"] >= 0].groupby("corridor_id"):
        corr_rows.append({
            "corridor_id": int(cid),
            "n_subpaths": len(grp),
            "n_flights": grp["flight_id"].nunique(),
            "total_length_km": grp["length_m"].sum() / 1000.0,
            "mean_length_km": grp["length_m"].mean() / 1000.0,
            "flights": sorted(grp["flight_id"].unique().tolist()),
        })
    corridors = pd.DataFrame(corr_rows).sort_values("n_flights", ascending=False)

    # Flight-level table
    flight_rows = []
    for fid, grp in gdf.groupby("flight_id"):
        sub_grp = subpaths[(subpaths["flight_id"] == fid) & (subpaths["corridor_id"] >= 0)]
        flight_rows.append({
            "flight_id": fid,
            "collection": grp["collection"].iloc[0],
            "segment": grp["segment"].iloc[0],
            "date": grp["date"].iloc[0],
            "n_frames": len(grp),
            "item_ids": grp["id"].tolist(),
            "n_subpaths_in_corridors": len(sub_grp),
            "corridors": sorted(sub_grp["corridor_id"].unique().tolist()),
        })
    flights_df = pd.DataFrame(flight_rows)

    points.to_parquet(OUT_DIR / "flight_points.parquet", index=False)
    subpaths.to_parquet(OUT_DIR / "subpaths.parquet", index=False)
    corridors.to_parquet(OUT_DIR / "corridors.parquet", index=False)
    flights_df.to_parquet(OUT_DIR / "flights.parquet", index=False)
    print(f"Wrote outputs/{{flight_points,subpaths,corridors,flights}}.parquet")
    print()
    print("Top corridors:")
    print(corridors[["corridor_id", "n_subpaths", "n_flights", "total_length_km", "mean_length_km"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
