#!/usr/bin/env python3
"""Coastal survey density analysis by IMBIE drainage basin.

Computes survey density only within a configurable distance of the coastline
in each Antarctic drainage basin (IMBIE SUBREGION). Outputs a map and a table
of additional line-km needed to reach a target equivalent resolution.

Usage:
  uv run coastal_survey_gaps.py [--source bedmap_local] [--grid-km 30]
      [--coast-dist-km 20] [--target-km 2]
"""

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import box

import rioxarray
import xopr.geometry
from rasterio.enums import Resampling
from rasterio.transform import from_bounds

from plot_survey_density import (
    REGIONS, OUT_DIR, bin_line_km, load_bedmap,
    load_xopr, extract_segments,
)


def get_imbie_basins():
    """Load Antarctic grounded regions dissolved by SUBREGION (IMBIE basins)."""
    regions = xopr.geometry.get_antarctic_regions(
        merge_regions=False, type='GR'
    ).to_crs('EPSG:3031')
    return regions.dissolve(by='SUBREGION').reset_index()


def get_coastline_3031():
    """Load Natural Earth coastline projected to EPSG:3031, clipped to Antarctica."""
    coast_shp = shpreader.natural_earth(
        resolution='10m', category='physical', name='coastline'
    )
    coast = gpd.read_file(coast_shp).to_crs('EPSG:3031')
    return coast.clip(box(-3.5e6, -3.5e6, 3.5e6, 3.5e6)).union_all()


def make_coastal_mask(basins, coastline, grid_m, max_extent, coast_dist_m):
    """Build per-basin coastal masks for grid cells.

    Parameters
    ----------
    basins : GeoDataFrame
        IMBIE basins in EPSG:3031.
    coastline : shapely geometry
        Merged coastline in EPSG:3031.
    grid_m : float
        Grid cell size in meters.
    max_extent : float
        Half-width of the grid domain in meters.
    coast_dist_m : float
        Maximum distance from coastline to include.

    Returns
    -------
    mask : np.ndarray, shape (nx, nx)
        Integer basin index (0-based) or -1 for excluded cells.
    basin_labels : list of str
        SUBREGION label for each basin index.
    """
    nx = int(2 * max_extent / grid_m)
    # Grid cell centers
    centers = np.linspace(-max_extent + grid_m / 2, max_extent - grid_m / 2, nx)
    cx, cy = np.meshgrid(centers, centers, indexing='ij')

    # Coastal buffer
    coast_buf = coastline.buffer(coast_dist_m)

    mask = np.full((nx, nx), -1, dtype=int)
    basin_labels = basins['SUBREGION'].tolist()

    from shapely import contains
    from shapely.prepared import prep

    coast_prep = prep(coast_buf)
    pts_flat = gpd.points_from_xy(cx.ravel(), cy.ravel())

    # First filter: within coastal buffer
    in_coast = np.array([coast_prep.contains(p) for p in pts_flat]).reshape(nx, nx)

    for i, (_, row) in enumerate(basins.iterrows()):
        basin_prep = prep(row.geometry)
        in_basin = np.array(
            [basin_prep.contains(p) for p in pts_flat]
        ).reshape(nx, nx)
        mask[in_basin & in_coast] = i

    return mask, basin_labels


def load_velocity_grid(grid_m, max_extent, overview_level=4):
    """Load ITS-LIVE velocity mosaic and compute 90th-percentile velocity per grid cell.

    Returns 2D array (nx, nx) of velocity in m/year, aligned with the
    analysis grid (first axis = x, second = y, origin at -max_extent).
    """
    url = ("https://its-live-data.s3.amazonaws.com/velocity_mosaic/v2/"
           "static/cog/ITS_LIVE_velocity_120m_RGI19A_0000_v02_v.tif")
    vel = rioxarray.open_rasterio(
        url, chunks='auto', overview_level=overview_level, cache=False,
    ).squeeze().drop_vars('band')

    # Reproject at overview resolution (not grid resolution) to preserve detail
    vel_reproj = vel.rio.reproject('EPSG:3031').compute()
    vals = vel_reproj.values  # (rows=y_desc, cols=x_asc)
    ys = vel_reproj.y.values  # descending
    xs = vel_reproj.x.values  # ascending

    # Map each velocity pixel to a grid cell index
    nx = int(2 * max_extent / grid_m)
    ix = ((xs + max_extent) / grid_m).astype(int)
    iy = ((ys + max_extent) / grid_m).astype(int)

    # Build flat arrays of (grid_cell_flat_index, value) for valid pixels
    ix_grid, iy_grid = np.meshgrid(ix, iy)  # both shape (nrows, ncols)
    valid = (
        np.isfinite(vals)
        & (ix_grid >= 0) & (ix_grid < nx)
        & (iy_grid >= 0) & (iy_grid < nx)
    )
    cell_idx = ix_grid[valid] * nx + iy_grid[valid]
    cell_vals = vals[valid]

    # Sort by cell index, then compute 90th percentile per cell
    order = np.argsort(cell_idx)
    cell_idx = cell_idx[order]
    cell_vals = cell_vals[order]
    splits = np.searchsorted(cell_idx, np.arange(nx * nx), side='left')
    splits = np.append(splits, len(cell_idx))

    result = np.full(nx * nx, np.nan)
    for c in range(nx * nx):
        s, e = splits[c], splits[c + 1]
        if e > s:
            result[c] = np.percentile(cell_vals[s:e], 90)

    return result.reshape(nx, nx)


def compute_gap_table(line_km, mask, basin_labels, grid_km, target_km):
    """Compute additional line-km needed per basin.

    The equivalent resolution for a cell is: spacing = 2 * cell_area / line_km.
    To achieve target_km spacing, a cell needs: line_km_needed = 2 * grid_km^2 / target_km.

    Returns list of dicts with basin stats.
    """
    needed_per_cell = 2 * grid_km**2 / target_km
    rows = []
    for i, label in enumerate(basin_labels):
        sel = mask == i
        n_cells = sel.sum()
        if n_cells == 0:
            continue
        cell_km = line_km[sel]
        deficit = np.maximum(0, needed_per_cell - cell_km)
        # Equivalent spacing per cell: 2 * grid_km^2 / line_km (inf where line_km == 0)
        with np.errstate(divide='ignore'):
            cell_spacing = 2 * grid_km**2 / cell_km
        cell_spacing[cell_km == 0] = np.inf
        rows.append({
            'basin': label,
            'n_cells': int(n_cells),
            'area_km2': int(n_cells * grid_km**2),
            'total_line_km': float(cell_km.sum()),
            'median_spacing_km': float(np.median(cell_spacing)),
            'additional_line_km': float(deficit.sum()),
            'cells_below_target': int((cell_km < needed_per_cell).sum()),
            'pct_below_target': float((cell_km < needed_per_cell).sum() / n_cells * 100),
        })
    return sorted(rows, key=lambda r: -r['additional_line_km'])


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--source', choices=['bedmap', 'bedmap_local', 'xopr'],
                   default='bedmap_local')
    p.add_argument('--grid-km', type=float, default=30)
    p.add_argument('--coast-dist-km', type=float, default=20,
                   help='Max distance from coastline [km]')
    p.add_argument('--target-km', type=float, default=2,
                   help='Target equivalent resolution [km]')
    p.add_argument('--vmin', type=float, default=0.1)
    p.add_argument('--vmax', type=float, default=10)
    p.add_argument('--min-velocity', type=float, default=None,
                   help='Only include cells where max surface velocity >= this value [m/year]')
    basin_filter = p.add_mutually_exclusive_group()
    basin_filter.add_argument('--exclude-basins', type=str, default=None,
                   help='Comma-separated list of basins to exclude (e.g. "H-Hp,Ipp-J")')
    basin_filter.add_argument('--include-basins', type=str, default=None,
                   help='Comma-separated list of basins to include (whitelist, e.g. "A-Ap,Ep-F")')
    p.add_argument('--rdgn', nargs=3, type=float, default=None, metavar=('GREEN', 'MID', 'RED'),
                   help='Use red-green diverging colorscale (green=good, mid=center, red=bad)')
    args = p.parse_args()

    grid_m = int(args.grid_km * 1000)
    reg = REGIONS['antarctica']

    # Load survey data
    print(f'Loading {args.source} data...')
    if args.source == 'bedmap':
        x1, y1, x2, y2 = load_bedmap(reg['epsg'], local_cache=False)
    elif args.source == 'bedmap_local':
        x1, y1, x2, y2 = load_bedmap(reg['epsg'], local_cache=True)
    else:
        geoms = load_xopr(region_filter='Antarctica')
        x1, y1, x2, y2 = extract_segments(geoms, reg['epsg'])
    print(f'  {len(x1)} segments')

    # Bin into grid
    line_km = bin_line_km(x1, y1, x2, y2, grid_m, reg['max_extent'])

    # Build coastal basin mask
    print('Loading basins and coastline...')
    basins = get_imbie_basins()
    coastline = get_coastline_3031()
    print(f'Building coastal mask ({args.coast_dist_km} km buffer)...')
    mask, basin_labels = make_coastal_mask(
        basins, coastline, grid_m, reg['max_extent'],
        args.coast_dist_km * 1000,
    )
    # Filter basins if requested
    if args.exclude_basins is not None:
        excluded = {b.strip() for b in args.exclude_basins.split(',')}
        for i, label in enumerate(basin_labels):
            if label in excluded:
                mask[mask == i] = -1
        print(f'  Excluded basins: {", ".join(sorted(excluded))}')
    elif args.include_basins is not None:
        included = {b.strip() for b in args.include_basins.split(',')}
        for i, label in enumerate(basin_labels):
            if label not in included:
                mask[mask == i] = -1
        print(f'  Included basins: {", ".join(sorted(included))}')

    n_coastal = (mask >= 0).sum()
    print(f'  {n_coastal} coastal grid cells across {len(basin_labels)} basins')

    # Optional velocity filter
    if args.min_velocity is not None:
        print(f'Loading velocity data (threshold {args.min_velocity} m/yr)...')
        vel_grid = load_velocity_grid(grid_m, reg['max_extent'])
        slow = np.isnan(vel_grid) | (vel_grid < args.min_velocity)
        n_removed = ((mask >= 0) & slow).sum()
        mask[slow] = -1
        print(f'  Removed {n_removed} cells below {args.min_velocity} m/yr')

    # Compute equivalent spacing (only for coastal cells)
    grid_km_size = args.grid_km
    with np.errstate(divide='ignore'):
        density = 2 * grid_km_size**2 / line_km
    density[line_km == 0] = np.nan
    density[mask < 0] = np.nan  # mask out non-coastal cells

    # Gap analysis table
    gap_table = compute_gap_table(
        line_km, mask, basin_labels, grid_km_size, args.target_km,
    )
    total_additional = sum(r['additional_line_km'] for r in gap_table)

    print(f'\n{"Basin":<10} {"Cells":>6} {"Area km²":>10} {"Exist. km":>12} '
          f'{"Med. sp. km":>12} {"Need. km":>12} {"Below %":>8}')
    print('-' * 78)
    for r in gap_table:
        med_sp = r['median_spacing_km']
        med_str = f'{med_sp:>12.1f}' if np.isfinite(med_sp) else f'{"inf":>12}'
        print(f'{r["basin"]:<10} {r["n_cells"]:>6} {r["area_km2"]:>10} '
              f'{r["total_line_km"]:>12.0f} {med_str} '
              f'{r["additional_line_km"]:>12.0f} '
              f'{r["pct_below_target"]:>7.1f}%')
    print('-' * 78)
    print(f'{"TOTAL":<10} {sum(r["n_cells"] for r in gap_table):>6} '
          f'{sum(r["area_km2"] for r in gap_table):>10} '
          f'{sum(r["total_line_km"] for r in gap_table):>12.0f} '
          f'{"":>12} '
          f'{total_additional:>12.0f}')
    print(f'\nTarget resolution: {args.target_km} km')
    print(f'Total additional line-km needed: {total_additional:,.0f}')

    # Plot
    proj = reg['crs']()
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection=proj))
    ax.coastlines(resolution='10m', color='gray')

    nx = line_km.shape[0]
    edges = np.linspace(-reg['max_extent'],
                        -reg['max_extent'] + nx * grid_m, nx + 1)

    if args.rdgn is not None:
        from matplotlib.colors import LinearSegmentedColormap
        green_val, mid_val, red_val = args.rdgn
        norm = mcolors.Normalize(vmin=green_val, vmax=red_val)
        # Resample RdYlGn_r so midpoint color lands at mid_val in linear scale
        base = plt.cm.RdYlGn_r
        mid_frac = (mid_val - green_val) / (red_val - green_val)
        n = 256
        # Map linear [0,1] positions to base colormap positions via piecewise linear
        # [0, mid_frac] in linear -> [0, 0.5] in base; [mid_frac, 1] -> [0.5, 1]
        positions = np.linspace(0, 1, n)
        base_positions = np.where(
            positions <= mid_frac,
            0.5 * positions / mid_frac,
            0.5 + 0.5 * (positions - mid_frac) / (1 - mid_frac),
        )
        cmap = LinearSegmentedColormap.from_list(
            'rdgn_recentered', base(base_positions), N=n,
        )
    else:
        norm = mcolors.Normalize(vmin=args.vmin, vmax=args.vmax)
        cmap = plt.cm.inferno.copy()
        cmap.set_over('white')
    im = ax.pcolormesh(edges, edges, density.T, norm=norm, cmap=cmap,
                       transform=proj)

    finite = density[np.isfinite(density)]
    vlo, vhi = norm.vmin, norm.vmax
    if len(finite) == 0:
        extend = 'neither'
    else:
        lo = np.nanmin(finite) < vlo
        hi = np.nanmax(finite) > vhi
        extend = ('both' if lo and hi else
                  ('min' if lo else ('max' if hi else 'neither')))
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, extend=extend)
    cbar.set_label('Equivalent gridded survey spacing [km]', fontsize=14)
    cbar.ax.invert_yaxis()

    # Draw basin boundaries
    for _, row in basins.iterrows():
        geom = row.geometry
        polys = geom.geoms if hasattr(geom, 'geoms') else [geom]
        for poly in polys:
            xs, ys = poly.exterior.coords.xy
            ax.plot(xs, ys, color='#555555', linewidth=0.6, transform=proj)

    ax.axis('off')
    ax.set_xlim(*reg['xlim'])
    ax.set_ylim(*reg['ylim'])
    ax.set_aspect('equal')
    title = f'Coastal survey density ({args.coast_dist_km:.0f} km from coast)'
    if args.min_velocity is not None:
        title += f', velocity ≥ {args.min_velocity} m/yr'
    ax.set_title(title, fontsize=14)

    # Legend for pie colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca02c', edgecolor='white', label='Collected'),
        Patch(facecolor='#d62728', edgecolor='white', label='Additional needed'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
              framealpha=0.9)

    # Pie charts drawn as wedge patches directly on the map axes
    from matplotlib.patches import Wedge
    gap_lookup = {r['basin']: r for r in gap_table}
    pie_radius = 120_000  # meters in data coords
    for _, row in basins.iterrows():
        label = row['SUBREGION']
        if label not in gap_lookup:
            continue
        stats = gap_lookup[label]
        collected = stats['total_line_km']
        additional = stats['additional_line_km']
        total = collected + additional
        if total == 0:
            continue

        pt = row.geometry.centroid
        cx, cy = pt.x, pt.y
        # Collected wedge (green), then additional (red)
        # Wedge sweeps CCW from theta1 to theta2
        angle_collected = 360 * collected / total
        wedge1 = Wedge((cx, cy), pie_radius, 90 - angle_collected, 90,
                        facecolor='#2ca02c', edgecolor='white', linewidth=0.4,
                        transform=proj, zorder=5)
        wedge2 = Wedge((cx, cy), pie_radius, 90 - 360, 90 - angle_collected,
                        facecolor='#d62728', edgecolor='white', linewidth=0.4,
                        transform=proj, zorder=5)
        ax.add_patch(wedge1)
        ax.add_patch(wedge2)
        # Basin label below pie
        ax.text(cx, cy - pie_radius * 1.3, label, transform=proj,
                ha='center', va='top', fontsize=5, color='#333333', zorder=6)

    plt.tight_layout()
    vel_suffix = f'_vel{args.min_velocity:.0f}' if args.min_velocity is not None else ''
    out_path = OUT_DIR / f'coastal_density_{args.source}_{args.coast_dist_km:.0f}km{vel_suffix}.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'\nSaved map to {out_path}')


if __name__ == '__main__':
    main()
