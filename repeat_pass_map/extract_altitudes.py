#!/usr/bin/env python3
"""Fetch and cache altitudes for radar frames that participate in a corridor.

Only uses the CSARP_layer HDF5 asset (≈0.1 MB) — orders of magnitude faster
than downloading the full CSARP_standard radar file. Frames without a
CSARP_layer asset (e.g., pre-2002 collections) are skipped.

Outputs:
  outputs/altitudes/{item_id}.parquet   one file per frame (cache)
  outputs/altitudes.parquet             concatenated table
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fsspec
import h5py
import numpy as np
import pandas as pd
import xopr
from pyproj import Geod

SCRIPT_DIR = Path(__file__).parent
OUT_DIR = SCRIPT_DIR.parent / "outputs"
CACHE_DIR = OUT_DIR / "altitudes"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DOWNSAMPLE_M = 50.0
N_WORKERS = 8
GEOD = Geod(ellps="WGS84")

CANONICAL = ("latitude", "longitude", "elevation", "gps_time")


def items_in_corridors(min_n_flights: int) -> list[str]:
    points = pd.read_parquet(OUT_DIR / "flight_points.parquet")
    subpaths = pd.read_parquet(OUT_DIR / "subpaths.parquet")
    corridors = pd.read_parquet(OUT_DIR / "corridors.parquet")
    qualifying = set(corridors[corridors["n_flights"] >= min_n_flights]["corridor_id"])
    sid_to_corr = dict(zip(subpaths["sub_id"], subpaths["corridor_id"]))
    points["corridor_id"] = points["sub_id"].map(sid_to_corr).fillna(-2).astype(int)
    in_corr = points[points["corridor_id"].isin(qualifying)]
    return sorted(in_corr["item_id"].unique().tolist())


def load_region_items(region: str) -> pd.DataFrame:
    opr = xopr.OPRConnection()
    cols = sorted(c["id"] for c in opr.get_collections() if region in c["id"])
    frames = []
    for cid in cols:
        items = opr.query_frames(collections=[cid], exclude_geometry=True)
        if items is None or len(items) == 0:
            continue
        items = items.copy()
        items["collection"] = cid
        frames.append(items)
    return pd.concat(frames, ignore_index=True)


def fetch_layer_metadata(url: str) -> dict:
    """Stream-read lat/lon/elev/gps_time from a CSARP_layer HDF5 file.

    Uses h5py over an fsspec HTTP file so we issue byte-range requests for
    only the needed datasets. Layer files are ~0.1 MB vs ~5-150 MB for the
    radar file.
    """
    fs = fsspec.filesystem("https")
    with fs.open(url, "rb") as f:
        with h5py.File(f, "r") as h:
            return {
                CANONICAL[0]: np.asarray(h["lat"][...]).ravel(),
                CANONICAL[1]: np.asarray(h["lon"][...]).ravel(),
                CANONICAL[2]: np.asarray(h["elev"][...]).ravel(),
                CANONICAL[3]: np.asarray(h["gps_time"][...]).ravel(),
            }


def along_track_extract(meta: dict, downsample_m: float, item_id: str) -> pd.DataFrame:
    lat = meta["latitude"]
    lon = meta["longitude"]
    elev = meta["elevation"]
    gpst = meta["gps_time"]

    if len(lat) >= 2:
        _, _, seg = GEOD.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
        cum = np.concatenate([[0.0], np.cumsum(seg)])
    else:
        cum = np.zeros_like(lat, dtype=float)

    if cum[-1] > 0:
        targets = np.arange(0, cum[-1] + downsample_m, downsample_m)
        idx = np.searchsorted(cum, targets)
        idx = np.clip(idx, 0, len(lat) - 1)
        idx = np.unique(idx)
    else:
        idx = np.array([0])

    return pd.DataFrame({
        "item_id": item_id,
        "trace_idx": idx,
        "along_track_m": cum[idx],
        "longitude": lon[idx],
        "latitude": lat[idx],
        "elevation": elev[idx],
        "gps_time": gpst[idx],
    })


def cached_or_fetch(item, force: bool, downsample_m: float):
    """Return (item_id, status, err) where status is 'cached' | 'fetched' | 'skipped' | 'failed'."""
    item_id = item["id"]
    cache_path = CACHE_DIR / f"{item_id}.parquet"
    if cache_path.exists() and not force:
        return item_id, "cached", None
    assets = item["assets"]
    if "CSARP_layer" not in assets or not assets["CSARP_layer"].get("href"):
        return item_id, "skipped", "no CSARP_layer asset"
    try:
        meta = fetch_layer_metadata(assets["CSARP_layer"]["href"])
        if not all(k in meta for k in CANONICAL):
            return item_id, "failed", f"missing vars (got {list(meta.keys())})"
        df = along_track_extract(meta, downsample_m, item_id)
        df.to_parquet(cache_path, index=False)
        return item_id, "fetched", None
    except Exception as e:
        return item_id, "failed", f"{type(e).__name__}: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="Greenland")
    parser.add_argument("--min-flights", type=int, default=5)
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    print(f"Selecting frames in {args.region} corridors (>= {args.min_flights} flights)...")
    target_ids = set(items_in_corridors(args.min_flights))
    print(f"  {len(target_ids):,} frames in qualifying corridors")

    print("Loading STAC items...")
    catalog = load_region_items(args.region)
    catalog = catalog[catalog["id"].isin(target_ids)].reset_index(drop=True)
    print(f"  {len(catalog):,} items resolved")

    if args.limit:
        catalog = catalog.head(args.limit)

    failures: list[tuple[str, str]] = []
    n_cached = n_fetched = n_skipped = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(cached_or_fetch, row, args.force, DOWNSAMPLE_M)
                   for _, row in catalog.iterrows()]
        for i, fut in enumerate(as_completed(futures), 1):
            item_id, status, err = fut.result()
            if status == "cached":
                n_cached += 1
            elif status == "fetched":
                n_fetched += 1
            elif status == "skipped":
                n_skipped += 1
            else:  # failed
                failures.append((item_id, err))
                print(f"  FAIL {item_id}: {err}", flush=True)
            if i % 50 == 0 or i == len(futures):
                elapsed = time.time() - t0
                rate = n_fetched / elapsed if elapsed > 0 else 0
                eta = (len(futures) - i) / rate if rate > 0 else 0
                print(f"  [{i}/{len(futures)}] cached={n_cached} fetched={n_fetched} "
                      f"skipped={n_skipped} failed={len(failures)} "
                      f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    print(f"\nDone. cached={n_cached}, fetched={n_fetched}, "
          f"skipped={n_skipped}, failed={len(failures)}")
    if failures:
        print("First few failures:")
        for f in failures[:5]:
            print(f"  {f[0]}: {f[1]}")

    print("Concatenating cache files...")
    dfs = [pd.read_parquet(p) for p in CACHE_DIR.glob("*.parquet")]
    if dfs:
        big = pd.concat(dfs, ignore_index=True)
        out = OUT_DIR / "altitudes.parquet"
        big.to_parquet(out, index=False)
        print(f"Wrote {out}: {len(big):,} rows, {big['item_id'].nunique()} items")


if __name__ == "__main__":
    sys.exit(main())
