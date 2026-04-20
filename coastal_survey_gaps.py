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
    local_path = Path(__file__).parent / "radar_cache" / "velocity" / "ITS_LIVE_velocity_120m_RGI19A_0000_v02_v.tif"
    source = str(local_path) if local_path.exists() else url
    open_kw = dict(chunks='auto', overview_level=overview_level)
    if source == url:
        open_kw['cache'] = False
    vel = rioxarray.open_rasterio(source, **open_kw).squeeze().drop_vars('band')

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


def _plot_geometry(ax, geom, **kwargs):
    """Plot a shapely geometry's outline on matplotlib axes."""
    from shapely.geometry import LineString, Polygon, MultiLineString, MultiPolygon, GeometryCollection
    if isinstance(geom, LineString):
        xs, ys = geom.coords.xy
        ax.plot(xs, ys, **kwargs)
    elif isinstance(geom, Polygon):
        xs, ys = geom.exterior.coords.xy
        ax.plot(xs, ys, **kwargs)
    elif isinstance(geom, (MultiLineString, MultiPolygon, GeometryCollection)):
        for g in geom.geoms:
            _plot_geometry(ax, g, **kwargs)


def plot_side_by_side(density, mask, basin_labels, gap_table, basins,
                      coastline_geom, grid_m, max_extent, args):
    """Plot each active basin in its own panel, rotated so coast faces down."""
    from shapely.affinity import rotate as shapely_rotate
    from matplotlib.patches import Wedge, Patch
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.gridspec import GridSpec
    import pyproj

    gap_lookup = {r['basin']: r for r in gap_table}
    transformer = pyproj.Transformer.from_crs('EPSG:3031', 'EPSG:4326', always_xy=True)

    nx_grid = density.shape[0]
    centers = np.linspace(-max_extent + grid_m / 2, max_extent - grid_m / 2, nx_grid)
    edges = np.linspace(-max_extent, -max_extent + nx_grid * grid_m, nx_grid + 1)
    ex, ey = np.meshgrid(edges, edges, indexing='ij')

    # Find active basins and compute rotation angles
    active = []
    for i, label in enumerate(basin_labels):
        n_cells = (mask == i).sum()
        if n_cells == 0 or label not in gap_lookup:
            continue
        basin_geom = basins[basins['SUBREGION'] == label].iloc[0].geometry
        cx, cy = basin_geom.centroid.x, basin_geom.centroid.y
        lon, _ = transformer.transform(cx, cy)
        theta = np.arctan2(cy, cx)
        # Rotate so coast (radially outward) points downward
        rot = -np.pi / 2 - theta

        # Compute bounds from coastal cells in rotated frame
        cell_ij = np.argwhere(mask == i)
        cell_x = centers[cell_ij[:, 0]]
        cell_y = centers[cell_ij[:, 1]]
        cos_a, sin_a = np.cos(rot), np.sin(rot)
        rx = cell_x * cos_a - cell_y * sin_a
        ry = cell_x * sin_a + cell_y * cos_a
        ext = max(rx.max() - rx.min(), ry.max() - ry.min())
        pad = max(ext * 0.4, grid_m * 5)

        active.append({
            'idx': i, 'label': label, 'geom': basin_geom,
            'lon': lon, 'rot': rot,
            'xlim': (rx.min() - pad, rx.max() + pad),
            'ylim': (ry.min() - pad, ry.max() + pad),
            'stats': gap_lookup[label],
        })

    active.sort(key=lambda b: b['lon'])
    n = len(active)
    if n == 0:
        print('No active basins to plot in side-by-side mode.')
        return None

    # Colormap / norm (same logic as main plot)
    if args.rdgn is not None:
        from matplotlib.colors import LinearSegmentedColormap
        green_val, mid_val, red_val = args.rdgn
        norm = mcolors.Normalize(vmin=green_val, vmax=red_val)
        base = plt.cm.PiYG_r
        mid_frac = (mid_val - green_val) / (red_val - green_val)
        nn = 256
        positions = np.linspace(0, 1, nn)
        base_positions = np.where(
            positions <= mid_frac,
            0.5 * positions / mid_frac,
            0.5 + 0.5 * (positions - mid_frac) / (1 - mid_frac),
        )
        cmap = LinearSegmentedColormap.from_list(
            'rdgn_recentered', base(base_positions), N=nn,
        )
    else:
        norm = mcolors.Normalize(vmin=args.vmin, vmax=args.vmax)
        cmap = plt.cm.inferno.copy()
        cmap.set_over('white')

    # Panel colors for linking basins to overview (cycle tab10)
    panel_colors = [plt.cm.tab10(i % 10) for i in range(n)]

    # Grid layout: small overview first, then basin panels
    n_cols = n + 2  # overview + basins + colorbar
    cell_size = 3.2  # inches per basin panel
    ov_ratio = 0.45  # overview width relative to a basin panel

    # Compute width ratios: overview, basin panels, colorbar
    width_ratios = [ov_ratio] + [1] * n + [0.04]
    fig_w = (ov_ratio + n) * cell_size + 1.0
    fig_h = cell_size + 1.0
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = GridSpec(1, n_cols, figure=fig, width_ratios=width_ratios,
                  left=0.02, right=0.94, bottom=0.04, top=0.92,
                  wspace=0.15)

    # Pre-rotate coastline and basin boundaries once per unique rotation
    basin_subregions = basins['SUBREGION'].tolist()
    rot_cache = {}
    for b in active:
        rot_deg = np.degrees(b['rot'])
        if rot_deg not in rot_cache:
            rot_coast = shapely_rotate(coastline_geom, rot_deg, origin=(0, 0))
            rot_basins = [
                (basin_subregions[i],
                 shapely_rotate(row.geometry, rot_deg, origin=(0, 0)))
                for i, (_, row) in enumerate(basins.iterrows())
            ]
            rot_cache[rot_deg] = (rot_coast, rot_basins)

    # Overview first (column 0)
    proj = ccrs.Stereographic(central_latitude=-90, true_scale_latitude=-71)
    ax_ov = fig.add_subplot(gs[0, 0], projection=proj)
    ax_ov.coastlines(resolution='50m', color='gray', linewidth=0.5)
    ax_ov.set_xlim(-2.8e6, 2.8e6)
    ax_ov.set_ylim(-2.8e6, 2.8e6)
    for _, row in basins.iterrows():
        polys = row.geometry.geoms if hasattr(row.geometry, 'geoms') else [row.geometry]
        for poly in polys:
            if hasattr(poly, 'exterior'):
                bxs, bys = poly.exterior.coords.xy
                ax_ov.plot(bxs, bys, color='#aaaaaa', linewidth=0.3,
                           transform=proj)
    for panel_idx, b in enumerate(active):
        polys = b['geom'].geoms if hasattr(b['geom'], 'geoms') else [b['geom']]
        for poly in polys:
            if hasattr(poly, 'exterior'):
                coords = np.array(poly.exterior.coords)
                patch = MplPolygon(coords, facecolor=panel_colors[panel_idx],
                                   alpha=0.4, edgecolor=panel_colors[panel_idx],
                                   linewidth=1.5, transform=proj, zorder=3)
                ax_ov.add_patch(patch)
    ax_ov.set_aspect('equal')
    ax_ov.axis('off')

    # Plot each basin panel (columns 1..n)
    bar_info = []  # (ax, pct) tuples for deferred progress bar drawing
    for panel_idx, b in enumerate(active):
        ax = fig.add_subplot(gs[0, panel_idx + 1])
        cos_a, sin_a = np.cos(b['rot']), np.sin(b['rot'])

        # Rotate grid edges
        rx = ex * cos_a - ey * sin_a
        ry = ex * sin_a + ey * cos_a

        # Mask density to this basin only
        d = density.copy()
        d[mask != b['idx']] = np.nan

        ax.pcolormesh(rx, ry, d, norm=norm, cmap=cmap, shading='flat')

        # Draw rotated coastline and basin boundaries
        rot_deg = np.degrees(b['rot'])
        rot_coast, rot_basins_list = rot_cache[rot_deg]
        _plot_geometry(ax, rot_coast, color='gray', linewidth=0.5)
        for blabel, rot_bnd in rot_basins_list:
            is_active = (blabel == b['label'])
            color = panel_colors[panel_idx] if is_active else '#555555'
            lw = 0.9 if is_active else 0.2
            polys = rot_bnd.geoms if hasattr(rot_bnd, 'geoms') else [rot_bnd]
            for poly in polys:
                if hasattr(poly, 'exterior'):
                    bxs, bys = poly.exterior.coords.xy
                    ax.plot(bxs, bys, color=color, linewidth=lw,
                            zorder=4 if is_active else 2)

        ax.set_xlim(*b['xlim'])
        ax.set_ylim(*b['ylim'])
        ax.set_aspect('equal')
        ax.axis('off')

        # Collect info for progress bars (drawn after layout is resolved)
        stats = b['stats']
        collected = stats['total_line_km']
        additional = stats['additional_line_km']
        total = collected + additional
        if total > 0:
            bar_info.append((ax, collected / total))

    # Colorbar (last column)
    import matplotlib.cm as mcm
    cbar_ax = fig.add_subplot(gs[0, n + 1])
    sm = mcm.ScalarMappable(norm=norm, cmap=cmap)
    finite = density[np.isfinite(density) & (mask >= 0)]
    vlo, vhi = norm.vmin, norm.vmax
    if len(finite) == 0:
        extend = 'neither'
    else:
        lo = np.nanmin(finite) < vlo
        hi = np.nanmax(finite) > vhi
        extend = ('both' if lo and hi else
                  ('min' if lo else ('max' if hi else 'neither')))
    cbar = fig.colorbar(sm, cax=cbar_ax, extend=extend)
    cbar.set_label('Equivalent gridded survey spacing [km]', fontsize=10)
    cbar.ax.invert_yaxis()

    # Draw progress bars at a consistent figure-Y position
    if bar_info:
        from matplotlib.patches import Rectangle
        fig.canvas.draw_idle()
        renderer = fig.canvas.get_renderer()
        fig_h = fig.get_size_inches()[1] * fig.dpi
        target_h_pts = 15  # fixed bar height in points
        bar_w_frac = 0.35
        # Use top of the tallest panel as reference Y (in figure fraction)
        bar_top_fig = max(
            ax.get_position().y1 for ax, _ in bar_info
        ) - 0.01  # small margin below top
        bar_h_fig = target_h_pts / fig_h
        bar_y_fig = bar_top_fig - bar_h_fig
        for ax, pct in bar_info:
            pos = ax.get_position()
            # Place bar in figure coords: right-aligned within each panel
            bar_x_fig = pos.x1 - bar_w_frac * pos.width - 0.02 * pos.width
            bar_w_fig = bar_w_frac * pos.width
            bar_ax = fig.add_axes([bar_x_fig, bar_y_fig, bar_w_fig, bar_h_fig])
            bar_ax.set_xlim(0, 1)
            bar_ax.set_ylim(0, 1)
            bar_ax.add_patch(Rectangle((0, 0), 1, 1,
                                       facecolor='white', edgecolor='#333333',
                                       linewidth=0.6))
            bar_ax.add_patch(Rectangle((0, 0), pct, 1,
                                       facecolor='#1f77b4', edgecolor='none'))
            bar_ax.add_patch(Rectangle((0, 0), 1, 1,
                                       facecolor='none', edgecolor='#333333',
                                       linewidth=0.6))
            bar_ax.text(0.5, 0.5, f'{pct:.0%}', ha='center', va='center',
                        fontsize=7, fontweight='bold', color='black',
                        transform=bar_ax.transAxes)
            bar_ax.axis('off')

    # Title
    title = f'Coastal survey density ({args.coast_dist_km:.0f} km from coast)'
    if args.min_velocity is not None:
        title += f', velocity \u2265 {args.min_velocity} m/yr'
    fig.suptitle(title, fontsize=14)

    return fig


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
    p.add_argument('--side-by-side', action='store_true',
                   help='Show each basin separately side by side with coast at bottom')
    p.add_argument('--set-resolution', type=float, default=None,
                   help='Override all coastal cells to this equivalent spacing [km]')
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

    # Override density if --set-resolution is given
    if args.set_resolution is not None:
        density[mask >= 0] = args.set_resolution
        # Also override line_km so gap table reflects the set resolution
        line_km[mask >= 0] = 2 * grid_km_size**2 / args.set_resolution
        print(f'  Overriding all coastal cells to {args.set_resolution} km spacing')

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

    # Side-by-side mode
    if args.side_by_side:
        fig = plot_side_by_side(
            density, mask, basin_labels, gap_table, basins,
            coastline, grid_m, reg['max_extent'], args,
        )
        if fig is not None:
            vel_suffix = (f'_vel{args.min_velocity:.0f}'
                          if args.min_velocity is not None else '')
            out_path = (OUT_DIR /
                        f'coastal_density_{args.source}_{args.coast_dist_km:.0f}km'
                        f'{vel_suffix}_sidebyside.png')
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f'\nSaved side-by-side map to {out_path}')
        return

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
        # Resample PiYG so midpoint color lands at mid_val in linear scale
        base = plt.cm.PiYG_r
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
