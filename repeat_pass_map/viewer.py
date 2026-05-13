#!/usr/bin/env python3
"""Interactive viewer for repeated flight-track corridors.

Three coordinated panels:
  A. Overview map: all corridors with >= min_flights, colored by corridor id
  B. Detail map: selected corridor, each member flight its own color
  C. Altitude profile: altitude vs arc-length along the corridor's reference line

Run with:  uv run panel serve viewer.py --show
"""

import io
from functools import lru_cache
from pathlib import Path

import cartopy.feature as cfeature
import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import pyproj
from scipy.spatial import cKDTree
from shapely.geometry import box
from shapely.ops import transform

hv.extension("bokeh")
pn.extension(loading_indicator=True)
pn.config.loading_spinner = "arc"
pn.config.loading_color = "#1976d2"

SCRIPT_DIR = Path(__file__).parent
OUT_DIR = SCRIPT_DIR.parent / "outputs"

PROJ_EPSG = 3413
TOL_M = 1000.0  # for projecting altitude traces onto corridor reference
MAP_SIZE = 560  # frame width = frame height (square plot area)
MAP_WIDTH = MAP_SIZE
MAP_HEIGHT = MAP_SIZE
GREENLAND_BBOX_WGS84 = (-75, 58, -10, 84)

# Square data bounding box (EPSG:3413) used by the overview map. Width and height
# spans are equal so that with the same pixel width/height the plot is square
# and matches the detail map exactly.
_OVERVIEW_CX, _OVERVIEW_CY = -250_000, -2_000_000
_OVERVIEW_HALF = 1_550_000
OVERVIEW_XLIM = (_OVERVIEW_CX - _OVERVIEW_HALF, _OVERVIEW_CX + _OVERVIEW_HALF)
OVERVIEW_YLIM = (_OVERVIEW_CY - _OVERVIEW_HALF, _OVERVIEW_CY + _OVERVIEW_HALF)


@lru_cache(maxsize=1)
def greenland_coastline_segments() -> list[dict]:
    """Return list of dicts with x/y arrays of Greenland coastline in EPSG:3413."""
    tx = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{PROJ_EPSG}", always_xy=True)
    clip = box(*GREENLAND_BBOX_WGS84)
    coast = cfeature.NaturalEarthFeature("physical", "coastline", "50m")
    segs = []
    for geom in coast.geometries():
        if not geom.intersects(clip):
            continue
        clipped = geom.intersection(clip)
        if clipped.is_empty:
            continue
        # clipped may be a LineString or MultiLineString
        if clipped.geom_type == "MultiLineString":
            lines = list(clipped.geoms)
        else:
            lines = [clipped]
        for line in lines:
            proj = transform(tx.transform, line)
            xs, ys = proj.xy
            segs.append({"x": np.asarray(xs), "y": np.asarray(ys)})
    return segs


def coastline_overlay():
    segs = greenland_coastline_segments()
    return hv.Path(segs, kdims=["x", "y"]).opts(
        color="black", line_width=0.5, alpha=0.7,
    )

# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_all():
    points = pd.read_parquet(OUT_DIR / "flight_points.parquet")
    subpaths = pd.read_parquet(OUT_DIR / "subpaths.parquet")
    corridors = pd.read_parquet(OUT_DIR / "corridors.parquet")
    sid_to_corr = dict(zip(subpaths["sub_id"], subpaths["corridor_id"]))
    points["corridor_id"] = points["sub_id"].map(sid_to_corr).fillna(-2).astype(int)
    alt_path = OUT_DIR / "altitudes.parquet"
    altitudes = pd.read_parquet(alt_path) if alt_path.exists() else pd.DataFrame()
    return points, subpaths, corridors, altitudes


POINTS, SUBPATHS, CORRIDORS, ALTITUDES = load_all()

# Pre-project altitudes (lon/lat → EPSG:3413) and tag with corridor_id via item lookup.
if len(ALTITUDES):
    _tx = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{PROJ_EPSG}", always_xy=True)
    ax, ay = _tx.transform(ALTITUDES["longitude"].values, ALTITUDES["latitude"].values)
    ALTITUDES["x"] = ax
    ALTITUDES["y"] = ay
    ITEM_TO_FLIGHT = dict(zip(POINTS["item_id"], POINTS["flight_id"]))
    ALTITUDES["flight_id"] = ALTITUDES["item_id"].map(ITEM_TO_FLIGHT)


# --------------------------------------------------------------------------
# Per-corridor data prep
# --------------------------------------------------------------------------

@lru_cache(maxsize=64)
def corridor_data(corr_id: int):
    """Return (tracks_df, ref_xy, ref_flight_id, altitude_df) for one corridor.

    For each flight that contributes to the corridor, we find its longest run
    of *consecutive* in-corridor points along its own track. Those per-flight
    runs are the curves drawn in the detail map. The longest such run across
    all flights becomes the corridor's reference line — so the reference can be
    much longer than the 30 km sub-path cap used during matching.
    """
    sub = SUBPATHS[SUBPATHS["corridor_id"] == corr_id]
    if len(sub) == 0:
        return None, None, None, None

    member_flights = sub["flight_id"].unique().tolist()
    pts = POINTS[POINTS["corridor_id"] == corr_id]

    tracks = {}
    longest_per_flight: dict[str, np.ndarray] = {}
    for fid in member_flights:
        f_pts = pts[pts["flight_id"] == fid].sort_values("point_idx")
        if len(f_pts) == 0:
            continue
        tracks[fid] = (f_pts["x"].to_numpy(), f_pts["y"].to_numpy())

        # Longest contiguous run of corridor points for this flight
        idxs = f_pts["point_idx"].to_numpy()
        if len(idxs) == 0:
            continue
        breaks = np.where(np.diff(idxs) > 1)[0]
        starts = np.concatenate([[0], breaks + 1])
        ends = np.concatenate([breaks + 1, [len(idxs)]])
        lengths = ends - starts
        best = int(np.argmax(lengths))
        s, e = starts[best], ends[best]
        run = f_pts.iloc[s:e][["x", "y"]].to_numpy()
        longest_per_flight[fid] = run

    if not longest_per_flight:
        return None, None, None, None

    ref_flight = max(longest_per_flight, key=lambda f: len(longest_per_flight[f]))
    ref_xy = longest_per_flight[ref_flight]
    ref_arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ref_xy, axis=0), axis=1))])

    # Project altitudes onto reference
    alt_rows = []
    if len(ALTITUDES):
        ref_tree = cKDTree(ref_xy)
        for fid in member_flights:
            a = ALTITUDES[ALTITUDES["flight_id"] == fid]
            if len(a) == 0:
                continue
            dists, idxs = ref_tree.query(a[["x", "y"]].to_numpy(), k=1)
            mask = dists <= TOL_M
            if not mask.any():
                continue
            alt_rows.append(pd.DataFrame({
                "flight_id": fid,
                "arc_length_m": ref_arc[idxs[mask]],
                "elevation": a["elevation"].values[mask],
            }))
    alt_df = pd.concat(alt_rows, ignore_index=True) if alt_rows else pd.DataFrame(
        columns=["flight_id", "arc_length_m", "elevation"]
    )

    return tracks, ref_xy, ref_flight, alt_df


# --------------------------------------------------------------------------
# Plot builders
# --------------------------------------------------------------------------

def overview_map(min_flights: int, highlight_corr: int | None,
                 min_length_km: float = 0.0):
    big = CORRIDORS[
        (CORRIDORS["n_flights"] >= min_flights)
        & (CORRIDORS["total_length_km"] >= min_length_km)
    ].sort_values("n_flights", ascending=False)
    big_ids = set(big["corridor_id"].tolist())
    in_corr_pts = POINTS[POINTS["corridor_id"].isin(big_ids)]
    # Use sample for speed if huge
    if len(in_corr_pts) > 200_000:
        in_corr_pts = in_corr_pts.sample(200_000, random_state=0)
    other = POINTS[~POINTS["corridor_id"].isin(big_ids)].sample(
        min(50_000, (~POINTS["corridor_id"].isin(big_ids)).sum()), random_state=0
    )

    bg = hv.Points(other, kdims=["x", "y"]).opts(
        size=1, color="lightgray", alpha=0.25, tools=[], default_tools=["pan", "wheel_zoom", "reset"]
    )
    fg = hv.Points(in_corr_pts, kdims=["x", "y"], vdims=["corridor_id"]).opts(
        size=3, color="corridor_id", cmap="glasbey_dark", alpha=0.7,
        tools=["hover", "tap"], active_tools=["wheel_zoom", "tap"],
        nonselection_alpha=0.7,
    )
    coast = coastline_overlay()
    plot = coast * bg * fg
    if highlight_corr is not None and highlight_corr >= 0:
        hpts = POINTS[POINTS["corridor_id"] == highlight_corr]
        plot = plot * hv.Points(hpts, kdims=["x", "y"]).opts(
            size=4, color="red", alpha=0.85
        )
    return plot.opts(
        frame_width=MAP_WIDTH, frame_height=MAP_HEIGHT, data_aspect=1,
        xlim=OVERVIEW_XLIM, ylim=OVERVIEW_YLIM,
        title=f"Corridors with ≥{min_flights} flights ({len(big_ids)})",
        xlabel="x (m, EPSG:3413)", ylabel="y (m, EPSG:3413)",
        show_legend=False,
    )


def detail_map(corr_id: int):
    tracks, ref_xy, ref_flight, _alts = corridor_data(corr_id)
    if not tracks:
        return hv.Text(0, 0, "no data").opts(
            frame_width=MAP_WIDTH, frame_height=MAP_HEIGHT
        )
    overlays = [coastline_overlay()]
    palette = hv.plotting.util.process_cmap("glasbey_dark", len(tracks))
    for color, (fid, (x, y)) in zip(palette, sorted(tracks.items())):
        overlays.append(hv.Curve(pd.DataFrame({"x": x, "y": y, "flight_id": fid}), "x", ["y", "flight_id"])
                        .opts(color=color, line_width=1.5, tools=["hover"]))
    # Highlight reference flight line on top
    if ref_xy is not None and len(ref_xy):
        overlays.append(
            hv.Curve(pd.DataFrame({"x": ref_xy[:, 0], "y": ref_xy[:, 1], "flight_id": f"REF: {ref_flight}"}),
                     "x", ["y", "flight_id"])
            .opts(color="black", line_width=3.5, line_dash="solid", tools=["hover"], alpha=0.9)
        )
    # Square xlim/ylim centered on corridor, with at least 50 km padding.
    all_x = np.concatenate([t[0] for t in tracks.values()])
    all_y = np.concatenate([t[1] for t in tracks.values()])
    cx, cy = (all_x.min() + all_x.max()) / 2, (all_y.min() + all_y.max()) / 2
    span = max(all_x.max() - all_x.min(), all_y.max() - all_y.min(), 50_000)
    half = span / 2 * 1.15 + 25_000
    xlim = (cx - half, cx + half)
    ylim = (cy - half, cy + half)
    return hv.Overlay(overlays).opts(
        frame_width=MAP_WIDTH, frame_height=MAP_HEIGHT, data_aspect=1,
        xlim=xlim, ylim=ylim,
        title=f"Corridor {corr_id} — {len(tracks)} flights",
        xlabel="x (m, EPSG:3413)", ylabel="y (m, EPSG:3413)",
        show_legend=False,
    )


def _kml_placemark(corr_id: int, ref_xy: np.ndarray, ref_flight: str,
                   transformer: pyproj.Transformer) -> str:
    lons, lats = transformer.transform(ref_xy[:, 0], ref_xy[:, 1])
    coords = " ".join(f"{lon:.6f},{lat:.6f},0" for lon, lat in zip(lons, lats))
    return (
        '    <Placemark>\n'
        f'      <name>corridor_{corr_id}_reference (from {ref_flight})</name>\n'
        '      <Style><LineStyle><color>ff0000ff</color><width>3</width></LineStyle></Style>\n'
        '      <LineString>\n'
        '        <tessellate>1</tessellate>\n'
        f'        <coordinates>{coords}</coordinates>\n'
        '      </LineString>\n'
        '    </Placemark>\n'
    )


def reference_kml(corr_id: int) -> bytes:
    """Return a KML LineString of the corridor's reference line in WGS84."""
    _, ref_xy, ref_flight, _ = corridor_data(corr_id)
    if ref_xy is None or len(ref_xy) == 0:
        return b""
    tx = pyproj.Transformer.from_crs(f"EPSG:{PROJ_EPSG}", "EPSG:4326", always_xy=True)
    body = _kml_placemark(corr_id, ref_xy, ref_flight, tx)
    kml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
        '  <Document>\n'
        f'    <name>Corridor {corr_id} reference</name>\n'
        f'{body}'
        '  </Document>\n'
        '</kml>\n'
    )
    return kml.encode("utf-8")


def bulk_reference_kml(min_flights: int, min_length_km: float) -> bytes:
    """Return a KML with one Placemark per filtered corridor's reference line."""
    tx = pyproj.Transformer.from_crs(f"EPSG:{PROJ_EPSG}", "EPSG:4326", always_xy=True)
    filt = CORRIDORS[
        (CORRIDORS["n_flights"] >= min_flights)
        & (CORRIDORS["total_length_km"] >= min_length_km)
    ].sort_values("n_flights", ascending=False)
    parts = []
    for cid in filt["corridor_id"].astype(int):
        _, ref_xy, ref_flight, _ = corridor_data(int(cid))
        if ref_xy is None or len(ref_xy) == 0:
            continue
        parts.append(_kml_placemark(int(cid), ref_xy, ref_flight, tx))
    body = "".join(parts)
    kml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
        '  <Document>\n'
        f'    <name>Greenland corridors — ≥{min_flights} flights, ≥{int(min_length_km)} km '
        f'({len(parts)} corridors)</name>\n'
        f'{body}'
        '  </Document>\n'
        '</kml>\n'
    )
    return kml.encode("utf-8")


def altitude_plot(corr_id: int):
    tracks, ref_xy, ref_flight, alt_df = corridor_data(corr_id)
    if alt_df is None or len(alt_df) == 0:
        return hv.Text(0, 0, "no altitudes available yet").opts(width=1060, height=300)
    overlays = []
    flights_sorted = sorted(alt_df["flight_id"].unique())
    palette = hv.plotting.util.process_cmap("glasbey_dark", max(len(flights_sorted), len(tracks)))
    # Use same color order as detail_map (sorted flight_id)
    flight_to_color = {fid: c for fid, c in zip(sorted(tracks.keys()), palette)}
    for fid in flights_sorted:
        sub = alt_df[alt_df["flight_id"] == fid].sort_values("arc_length_m")
        c = flight_to_color.get(fid, "gray")
        overlays.append(hv.Scatter(sub, kdims=["arc_length_m"], vdims=["elevation", "flight_id"])
                        .opts(color=c, size=2, alpha=0.7, tools=["hover"]))
    return hv.Overlay(overlays).opts(
        width=1060, height=300,
        title=f"Altitude along corridor {corr_id} reference line",
        xlabel="along-corridor distance (m)", ylabel="elevation (m, WGS84)",
        legend_position="right",
    )


# --------------------------------------------------------------------------
# Click-to-select machinery
# --------------------------------------------------------------------------

# KDTree of all in-corridor points so we can map a tap (x, y) to a corridor id.
_corr_pts = POINTS[POINTS["corridor_id"] >= 0]
_TAP_TREE = cKDTree(_corr_pts[["x", "y"]].to_numpy()) if len(_corr_pts) else None
_TAP_CORR_IDS = _corr_pts["corridor_id"].to_numpy() if len(_corr_pts) else np.array([])


def _label_for(corr_id: int) -> str:
    row = CORRIDORS[CORRIDORS["corridor_id"] == corr_id]
    if len(row) == 0:
        return f"c{corr_id}"
    r = row.iloc[0]
    return f"c{int(corr_id)} — {int(r['n_flights'])} fl, {int(r['total_length_km'])} km"


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------

max_length = int(CORRIDORS["total_length_km"].max())
min_flights_slider = pn.widgets.IntSlider(
    name="min flights per corridor", value=5, start=2, end=20
)
min_length_slider = pn.widgets.IntSlider(
    name="min total length (km)", value=100, start=20,
    end=min(max(max_length, 100), 20_000), step=50,
)


def _filtered_corridors(min_flights: int, min_length: int) -> pd.DataFrame:
    return CORRIDORS[
        (CORRIDORS["n_flights"] >= min_flights)
        & (CORRIDORS["total_length_km"] >= min_length)
    ].sort_values("n_flights", ascending=False)


def _options_for(min_flights: int, min_length: int) -> dict[str, int]:
    filt = _filtered_corridors(min_flights, min_length).head(200)
    return {_label_for(int(r["corridor_id"])): int(r["corridor_id"]) for _, r in filt.iterrows()}


_init_options = _options_for(min_flights_slider.value, min_length_slider.value)
corridor_select = pn.widgets.Select(
    name="corridor", options=_init_options,
    value=next(iter(_init_options.values())) if _init_options else None,
)


def _refresh_options(*events) -> None:
    opts = _options_for(min_flights_slider.value, min_length_slider.value)
    if not opts:
        corridor_select.options = {}
        corridor_select.value = None
        return
    prev = corridor_select.value
    corridor_select.options = opts
    if prev not in opts.values():
        corridor_select.value = next(iter(opts.values()))


min_flights_slider.param.watch(_refresh_options, "value")
min_length_slider.param.watch(_refresh_options, "value")


def _tap_select(x, y) -> None:
    if x is None or y is None or _TAP_TREE is None:
        return
    _, idx = _TAP_TREE.query([x, y])
    cid = int(_TAP_CORR_IDS[idx])
    # If the tapped corridor isn't currently in the dropdown (e.g., it ranks
    # outside the top-200 shown there), add it without touching the filters.
    if cid not in corridor_select.options.values():
        opts = dict(corridor_select.options)
        opts[_label_for(cid)] = cid
        corridor_select.options = opts
    corridor_select.value = cid


def _make_overview(min_flights, min_length, cid, x, y):
    # x/y are pulled in as a stream so holoviews wires up the bokeh tap, but
    # they don't affect the rendering — the actual selection side-effect runs
    # in the subscriber below.
    return overview_map(min_flights, cid, min_length_km=min_length)


_flights_stream = hv.streams.Params(min_flights_slider, parameters=["value"],
                                    rename={"value": "min_flights"})
_length_stream = hv.streams.Params(min_length_slider, parameters=["value"],
                                   rename={"value": "min_length"})
_corridor_stream = hv.streams.Params(corridor_select, parameters=["value"],
                                     rename={"value": "cid"})
_tap_stream = hv.streams.SingleTap(x=None, y=None)

overview_dmap = hv.DynamicMap(
    _make_overview,
    streams=[_flights_stream, _length_stream, _corridor_stream, _tap_stream],
)


def _tap_subscriber(x, y):
    _tap_select(x, y)


_tap_stream.add_subscriber(_tap_subscriber)

_overview = overview_dmap


@pn.depends(corridor_select.param.value)
def _detail(cid):
    if cid is None:
        return hv.Text(0, 0, "select a corridor")
    return detail_map(int(cid))


@pn.depends(corridor_select.param.value)
def _altitude(cid):
    if cid is None:
        return hv.Text(0, 0, "select a corridor")
    return altitude_plot(int(cid))


@pn.depends(corridor_select.param.value)
def _kml_button(cid):
    if cid is None:
        return pn.pane.Markdown("_select a corridor for KML_")
    cid = int(cid)
    return pn.widgets.FileDownload(
        callback=lambda c=cid: io.BytesIO(reference_kml(c)),
        filename=f"corridor_{cid}_reference.kml",
        label=f"Download c{cid} reference line (KML)",
        button_type="primary",
    )


@pn.depends(min_flights_slider.param.value, min_length_slider.param.value)
def _bulk_kml_button(min_flights, min_length):
    n = len(_filtered_corridors(min_flights, min_length))
    return pn.widgets.FileDownload(
        callback=lambda mf=min_flights, ml=min_length: io.BytesIO(bulk_reference_kml(mf, ml)),
        filename=f"corridors_min{min_flights}fl_min{int(min_length)}km_references.kml",
        label=f"Download all {n} filtered reference lines (KML)",
        button_type="default",
    )


sidebar = pn.Column(
    "## Greenland repeat-flight corridors",
    min_flights_slider,
    min_length_slider,
    corridor_select,
    _kml_button,
    _bulk_kml_button,
    f"_Top 200 corridors by flight count; altitudes loaded: {len(ALTITUDES):,} rows from {ALTITUDES['item_id'].nunique() if len(ALTITUDES) else 0} items_",
    width=320,
)

app = pn.Column(
    pn.Row(sidebar, _overview, _detail),
    _altitude,
).servable()


if __name__ == "__main__":
    pn.serve(app, port=5006, show=True)
