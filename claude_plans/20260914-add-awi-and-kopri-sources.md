# Plan: add AWI radar-sounder tracks and the UTIG/KOPRI helicopter surveys

Status: implemented 2026-09-14 on branch `add-direct-sources`; results in `claude_notes/20260914_direct_sources_implementation.md`.
Background: `claude_notes/20260914_qiceradar_assessment.md`,
`claude_notes/20260914_direct_provider_sources.md`.

## Goal

Bring two sources that the current BedMap + xOPR pipeline misses into all
figures (bar charts, combined availability, survey density, coastal gaps):

1. AWI ice sounders (EMR, UWB, UWBM) from the AWI map server, dated per
   profile, 1994/95 to 2025/26. Replaces the AWI BedMap files (subset, undated).
2. UTIG/KOPRI helicopter radar over the Amundsen Sea (QIceRadar layers
   ASE2 to ASE6), 2017/18 to 2025/26, absent from every other source.

Do it through one new module so every figure picks the data up the same way,
mirroring how `bedmap_common.py` centralised the BedMap rules.

## Design

### 1. `extra_sources.py`: one campaign table for all consumers

`load_extra_campaigns() -> GeoDataFrame` with the same columns
`load_bedmap_catalog` returns (`name, institution, geometry [lon/lat],
ts, te`) plus `source` ("awi_wfs" | "qiceradar"), `season` (July-June start
year), `line_km`, `access` ("released" | "committed" | "none"), `doi`,
`season_confidence` ("dated" | "high" | "low"). One row per profile/segment
(AWI) or per feature (KOPRI); consumers aggregate.

Also `extra_track_points(spacing_m=250) -> DataFrame[x, y, source_file]`
that densifies the geometries for the density/coastal grids, so
`plot_survey_density.load_bedmap`-style consumers can concatenate it.

### 2. AWI ingest

- Fetch `https://maps.awi.de/services/common/radar/ows`, WFS 2.0
  `GetFeature typeNames=radar:tracks_3031 outputFormat=application/json`
  (~245 MB). Cache raw GeoJSON under `radar_cache/awi/` (gitignored) and a
  slim parquet next to it; record the fetch date in the parquet metadata.
- Keep `radar_system in {EMR, UWB, UWBM}` (AWI's own ice sounders). Drop
  ACCU/SNOW/ASIRAS (accumulation/snow radars) and all non-AWI holdings
  (PASIN, MCoRDS, HiCARS, MARFA), which are already covered by BedMap/xOPR.
- Deduplicate the two-product listing (EMR60/EMR600, UWB standard/qlook):
  derive a profile key from `prof_id` with the radar/product token removed,
  keep the longest feature set per key, then verify per season that
  summed length / buffered-union length is within 1.05 (it is 2.0-2.2
  before dedup; see `awi_portal_overlap.py`). If the key approach does not
  reach 1.05 for a season, fall back to a buffered-union geometry for that
  season.
- `season` from `date_time_start` (month >= 7 → year, else year-1).
  `access`: "released" if `doi` is a dataset DOI (not the bibliography
  10.1594/PANGAEA.972094), else "committed" (AWI policy, matches the
  current chart's treatment of AWI).
- Expected result after dedup: ~540k km, per season e.g. 2001 44k, 2003
  44k, 2013 47k, 2022 14k, 2023 13k, 2024 23k, 2025 25k (table in the
  provider-sources note). Nothing for 2019-2021.
- Remove BedMap `AWI_*` entries when the AWI source is active (add to a
  `SUPERSEDED_BY_EXTRA` set in `bedmap_common`, applied in
  `load_bedmap_catalog`/`kept_source_files`) to avoid double counting.
  Note this changes historical AWI bars, roughly +60k km over 2010-2014
  (flights never submitted to BedMap).

### 3. UTIG/KOPRI helicopter ingest

- Download the QIceRadar gpkg from the pinned DOI 10.5281/zenodo.21964546
  (v0.3.0) into `radar_cache/qiceradar/`; read layers ASE2..ASE6 only.
  Reproject 3031 → 4326.
- Season table (manual, in the module). KOPRI Araon cruises are annual and
  visit Thwaites every other year, most recently 2025/26 (user, 2026-09-14),
  which fixes the alternating sequence:
  KRT1 2016, ASE2 2017 (Dodson/Getz), KRT2 2018, ASE3 2019 (ANA10B, files
  dated 31 Jan-11 Feb 2020, TDR doi:10.18738/T8/730HSL), ASE4 2021 (ANA12,
  files dated 31 Jan-4 Feb 2022; the TDR title/date fields say 2023 but the
  file names, KML label, Level 2 title and build notes say 2022; TDR
  doi:10.18738/T8/3RSCPO), ASE5 2023, ASE6 2025. All `access` = "none":
  TDR hosts Level 1B focused radargrams and Level 2 tables, not raw data,
  and no release commitment exists. `institution` = "KOPRI", country "Other".
- ~14.8k km total; ~9.7k km within 150 km of Thwaites.
- No overlap with BedMap/xOPR, so nothing to remove.

### 4. Consumers

- `plot_bedmap_availability.py` and `plot_combined_availability.py`:
  concatenate `load_bedmap_catalog(...)` with `load_extra_campaigns()`.
  Extra rows carry a single `season`; put their line-km in that year (no
  calendar spread, because they have real dates). Combined chart: AWI
  "released" and KOPRI "released" go to the "commitment" bucket unless a
  fourth category "released outside xOPR" is added (decision below).
  Extend the x-axis to 2025 (decision below).
- `plot_survey_density.py` / `coastal_survey_gaps.py`: when `--source`
  is a BedMap variant, append `extra_track_points()` to the point table
  before `bin_line_km`; add `--no-extra` to disable. The 1 km gap filter
  is irrelevant at 250 m spacing.
- `README.md`: document the new module, the cache, the pinned versions.

### 5. Verification

- Per-season AWI km after dedup matches the table in the provider note
  within a few percent; buffered-union ratio <= 1.05 per season.
- BedMap AWI files no longer appear in `kept_source_files`; total AWI km
  in the bar chart = extra-source AWI km.
- Cell-overlap check (reuse `qiceradar_spatial_overlap.py` machinery):
  extra-source km inside xOPR cells stays < 2 %.
- Re-run all four figures; compare summary totals before/after and record
  them in `claude_notes/`.

## Decisions (user, 2026-09-14)

1. Replace all AWI BedMap entries with the portal source.
2. Extend chart windows to 2025.
3. Three categories. All publicly available AWI data counts as "commitment
   to release" (it is not in xOPR); KOPRI helicopter data is "no access".
4. New sources by season year; existing BedMap handling unchanged.
5. Fix the renamed BM2/BM3 duplicates in the same change.
6. Added scope: tooling to substitute xOPR collections for unreliable
   BedMap files (the 2009 CReSIS swath case), loading real flight geometry
   rather than the simplified STAC geometry, with a length check against
   BedMap for every substituted file (expected to agree except 2009).
7. PRIC is out of scope (season labels untrusted).

## Decisions needed before implementing (resolved above)

1. Replace all AWI BedMap entries with the WFS source (recommended: dated,
   complete, superset) or only add seasons >= 2020 (keeps historical bars
   unchanged)?
2. Extend chart windows to 2025 (data exists for 2022/23-2025/26)?
3. Combined chart: keep three categories (AWI/KOPRI releases counted as
   "commitment") or add "released outside xOPR"?
4. Season-year assignment for the new sources (recommended) while BedMap
   keeps the calendar spread the user chose to leave alone; accept the
   mixed convention or revisit 3a of the repo review at the same time.
5. Fix the renamed BM2/BM3 duplicates (section 5 of the QIceRadar note,
   ~55k km) in the same `bedmap_common` change?

## Out of scope

PRIC/CHINARE lines (season labels untrusted, see the QIceRadar note);
other QIceRadar-only layers (unreleased UTIG SOAR surveys, BAS 1969-1988,
SPRI), which are pre-window or undated.
