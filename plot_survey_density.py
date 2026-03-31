#!/usr/bin/env python3
"""Survey density map from BedMap or xOPR catalogs.

Configuration via command-line arguments:
  --source: "bedmap", "bedmap_local", or "xopr" (default: bedmap)
  --region: "antarctica" or "greenland" (default: antarctica)
  --zoom: zoom to a named sub-region (ase, wilkes, aurora)
  --grid-km: grid cell size in km (default: 30; 0 for raw survey lines)
  --target-spacing: target survey spacing in km; uses diverging PiYG colorscale
  --vmin/--vmax: colorbar limits (default: 0.1/10)
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer

SCRIPT_DIR = Path(__file__).parent
OUT_DIR = SCRIPT_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

SUBDIV = 5_000  # subdivide segments > 5 km for accurate binning

REGIONS = {
    "antarctica": {
        "epsg": "EPSG:3031",
        "crs": lambda: ccrs.Stereographic(central_latitude=-90,
                                           true_scale_latitude=-71),
        "max_extent": 3_333_000,
        "xlim": (-2.8e6, 2.8e6),
        "ylim": (-2.8e6, 2.8e6),
    },
    "greenland": {
        "epsg": "EPSG:3413",
        "crs": lambda: ccrs.Stereographic(central_latitude=90,
                                           true_scale_latitude=70),
        "max_extent": 3_500_000,
        "xlim": (-900_000, 500_000),
        "ylim": (-3_500_000, -500_000),
    },
}

# Zoom regions: override xlim/ylim for a parent region
ZOOM_REGIONS = {
    "ase": {
        "parent": "antarctica",
        "xlim": (-1.75e6, -1.25e6),
        "ylim": (-800e3, 0),
    },
    "wilkes": {
        "parent": "antarctica",
        "xlim": (0.8e6, 1.7e6),
        "ylim": (-2.4e6, -1.5e6),
    },
    "aurora": {
        "parent": "antarctica",
        "xlim": (1.8e6, 3.0e6),
        "ylim": (-0.9e6, 1.3e6),
    },
}


def load_bedmap(epsg, max_point_spacing_m=1000, local_cache=True):
    """Load BedMap2+3 point data and return projected segment endpoints.

    Parameters
    ----------
    epsg : str
        Target CRS (e.g. "EPSG:3031").
    max_point_spacing_m : float
        Discard segments longer than this (gap between survey points).

    Returns
    -------
    x1, y1, x2, y2 : np.ndarray
        Segment endpoint arrays in the target CRS.
    """
    from xopr.bedmap import query_bedmap, fetch_bedmap
    tf = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    fetch_bedmap()
    df = query_bedmap(collections=["bedmap1", "bedmap2", "bedmap3"],
                      columns=["lon", "lat", "source_file", "row"],
                      local_cache=local_cache, show_progress=True)
    print(f"  {len(df)} BedMap points from {df['source_file'].nunique()} files")
    df = df.sort_values(["source_file", "row"])
    xs, ys = tf.transform(df["lon"].values, df["lat"].values)
    # Mask transitions between files so we don't connect unrelated points
    same_file = df["source_file"].values[:-1] == df["source_file"].values[1:]
    dists = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
    ok = same_file & (dists < max_point_spacing_m) & (dists > 0)
    return xs[:-1][ok], ys[:-1][ok], xs[1:][ok], ys[1:][ok]

def load_xopr(region_filter=None):
    """Load geometries from xOPR STAC catalog.

    Parameters
    ----------
    region_filter : str, optional
        Filter collections by name substring ("Antarctica" or "Greenland").
    """
    import xopr
    conn = xopr.OPRConnection()
    geoms = []
    for c in sorted(conn.get_collections(), key=lambda c: c["id"]):
        if region_filter and region_filter not in c["id"]:
            continue
        items = conn.query_frames(collections=[c["id"]], exclude_geometry=False)
        if items is not None and len(items) > 0:
            geoms.extend(items["geometry"].dropna().tolist())
            print(f"  {c['id']}: {len(items)} frames")
    print(f"  {len(geoms)} total frames")
    return geoms


def extract_segments(geometries, epsg):
    """Project geometries to target CRS, return segment endpoints."""
    tf = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    x1s, y1s, x2s, y2s = [], [], [], []
    for geom in geometries:
        if geom is None or geom.is_empty:
            continue
        for line in (geom.geoms if hasattr(geom, "geoms") else [geom]):
            coords = np.array(line.coords)
            if len(coords) < 2:
                continue
            xs, ys = tf.transform(coords[:, 0], coords[:, 1])
            x1s.append(xs[:-1]); y1s.append(ys[:-1])
            x2s.append(xs[1:]); y2s.append(ys[1:])
    return (np.concatenate(x1s), np.concatenate(y1s),
            np.concatenate(x2s), np.concatenate(y2s))


def bin_line_km(x1, y1, x2, y2, grid_m, max_extent):
    """Bin segment lengths into grid cells, subdividing long segments."""
    dists = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    ok = dists > 0
    x1, y1, x2, y2, dists = x1[ok], y1[ok], x2[ok], y2[ok], dists[ok]

    n_sub = np.maximum(1, np.ceil(dists / SUBDIV).astype(int))
    total = n_sub.sum()
    offsets = np.repeat(np.cumsum(n_sub) - n_sub, n_sub)
    idx = np.arange(total) - offsets
    rn = np.repeat(n_sub, n_sub).astype(float)
    mx = np.repeat(x1, n_sub) + np.repeat(x2 - x1, n_sub) * (idx + 0.5) / rn
    my = np.repeat(y1, n_sub) + np.repeat(y2 - y1, n_sub) * (idx + 0.5) / rn
    sl = np.repeat(dists, n_sub) / rn

    nx = int(2 * max_extent / grid_m)
    grid = np.zeros((nx, nx))
    ix = ((mx + max_extent) / grid_m).astype(int)
    iy = ((my + max_extent) / grid_m).astype(int)
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < nx)
    np.add.at(grid, (ix[valid], iy[valid]), sl[valid])
    return grid / 1000.0  # m → km


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["bedmap", "bedmap_local", "xopr"],
                   default="bedmap")
    p.add_argument("--region", choices=["antarctica", "greenland"],
                   default="antarctica")
    p.add_argument("--zoom", choices=list(ZOOM_REGIONS.keys()),
                   help="Zoom to a named sub-region")
    p.add_argument("--grid-km", type=float, default=30)
    p.add_argument("--target-spacing", type=float, default=None,
                   help="Target survey spacing [km]; uses diverging PiYG colorscale")
    p.add_argument("--vmin", type=float, default=0.1)
    p.add_argument("--vmax", type=float, default=10)
    args = p.parse_args()
    grid_m = int(args.grid_km * 1000)
    if args.zoom:
        zr = ZOOM_REGIONS[args.zoom]
        args.region = zr["parent"]
    reg = REGIONS[args.region]

    print(f"Loading {args.source} data ({args.region})...")
    if args.source == "bedmap":
        if args.region == "greenland":
            print("  Warning: BedMap is Antarctic-only, map will be empty.")
        x1, y1, x2, y2 = load_bedmap(reg["epsg"], local_cache=False)
    elif args.source == "bedmap_local":
        if args.region == "greenland":
            print("  Warning: BedMap is Antarctic-only, map will be empty.")
        x1, y1, x2, y2 = load_bedmap(reg["epsg"], local_cache=True)
    else:
        region_filter = {"antarctica": "Antarctica",
                         "greenland": "Greenland"}[args.region]
        geoms = load_xopr(region_filter=region_filter)
        x1, y1, x2, y2 = extract_segments(geoms, reg["epsg"])
    print(f"  {len(x1)} segments")

    xlim = ZOOM_REGIONS[args.zoom]["xlim"] if args.zoom else reg["xlim"]
    ylim = ZOOM_REGIONS[args.zoom]["ylim"] if args.zoom else reg["ylim"]

    # Plot
    proj = reg["crs"]()
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection=proj))
    ax.coastlines(resolution="10m", color="gray")

    if grid_m == 0:
        # Direct survey line plot — clip to view bounds first
        from matplotlib.collections import LineCollection
        x1m, x2m = xlim
        y1m, y2m = ylim
        pad = max(x2m - x1m, y2m - y1m) * 0.1
        in_view = (
            (np.maximum(x1, x2) >= x1m - pad) & (np.minimum(x1, x2) <= x2m + pad) &
            (np.maximum(y1, y2) >= y1m - pad) & (np.minimum(y1, y2) <= y2m + pad)
        )
        print(f"  {in_view.sum()} segments in view (of {len(x1)})")
        segs = np.stack([np.column_stack([x1[in_view], y1[in_view]]),
                         np.column_stack([x2[in_view], y2[in_view]])], axis=1)
        lc = LineCollection(segs, linewidths=0.3, colors="blue", alpha=0.5,
                            transform=proj)
        ax.add_collection(lc)
    else:
        line_km = bin_line_km(x1, y1, x2, y2, grid_m, reg["max_extent"])
        grid_km_size = grid_m / 1000.0
        with np.errstate(divide="ignore"):
            density = 2 * grid_km_size**2 / line_km
        density[line_km == 0] = np.nan

        nx = line_km.shape[0]
        edges = np.linspace(-reg["max_extent"],
                            -reg["max_extent"] + nx * grid_m, nx + 1)
        if args.target_spacing is not None:
            tc = args.target_spacing
            norm = mcolors.TwoSlopeNorm(vcenter=tc, vmin=args.vmin, vmax=args.vmax)
            cmap = plt.cm.PiYG_r.copy()
        else:
            norm = mcolors.Normalize(vmin=args.vmin, vmax=args.vmax)
            cmap = plt.cm.magma.copy()
            cmap.set_over("white")
        im = ax.pcolormesh(edges, edges, density.T, norm=norm, cmap=cmap,
                           transform=proj)
        finite = density[np.isfinite(density)]
        if len(finite) == 0:
            extend = "neither"
        else:
            lo = np.nanmin(finite) < args.vmin
            hi = np.nanmax(finite) > args.vmax
            extend = "both" if lo and hi else ("min" if lo else ("max" if hi else "neither"))
        cbar = fig.colorbar(im, ax=ax, shrink=0.7, extend=extend)
        cbar.set_label("Equivalent gridded survey spacing [km]", fontsize=14)
        cbar.ax.invert_yaxis()

    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    ax.set_title(" ")

    # Scale bar: pick a round length ~15% of view width, placed lower-left on land
    view_width_km = (xlim[1] - xlim[0]) / 1000
    for bar_km in [1, 2, 5, 10, 20, 50, 100, 200, 500]:
        if bar_km >= view_width_km * 0.1:
            break
    bar_m = bar_km * 1000
    view_h = ylim[1] - ylim[0]
    x0 = xlim[0] + (xlim[1] - xlim[0]) * 0.05
    y0 = ylim[0] + view_h * 0.20  # above coastline
    ax.plot([x0, x0 + bar_m], [y0, y0], color="black", linewidth=4,
            solid_capstyle="butt", transform=proj,
            path_effects=[__import__("matplotlib.patheffects", fromlist=["withStroke"])
                          .withStroke(linewidth=6, foreground="white")])
    ax.text(x0 + bar_m / 2, y0 + view_h * 0.015,
            f"{bar_km} km", color="black", ha="center", va="bottom",
            fontsize=11, fontweight="bold", transform=proj,
            path_effects=[__import__("matplotlib.patheffects", fromlist=["withStroke"])
                          .withStroke(linewidth=3, foreground="white")])

    plt.tight_layout()
    suffix = f"_{args.zoom}" if args.zoom else ""
    mode = "lines" if grid_m == 0 else "density"
    out_path = OUT_DIR / f"survey_{mode}_{args.source}_{args.region}{suffix}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved to {out_path}")
