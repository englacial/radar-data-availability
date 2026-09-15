# Radar Data Availability Figures

Figures showing airborne radar data availability and survey density for Antarctic and Greenland ice sheets, using data from [xOPR](https://github.com/englacial/xopr) and BedMap catalogs.

## Setup

```bash
uv sync
```

For local development with a local xopr checkout, override the source:

```bash
uv sync --override ../xopr
```

Or edit `pyproject.toml` to point `xopr` at your local path.

## Entry points

All BedMap consumers share `bedmap_common.py`, which applies the campaign
exclusion list, drops BM2 files superseded by renamed BM3 files, keeps one
version of campaigns present in both BedMap2 and BedMap3, parses temporal
metadata (a missing or year-9999 end date falls back to the start date), and
defines the gap rule (`MAX_POINT_SPACING_M`, 1.5 km) that decides which
consecutive points form a flight line. Per-campaign line-km for the bar
charts come from that rule applied to the point data, stored in
`bedmap_campaign_km.csv` (rebuild with `uv run python bedmap_common.py`
after changing the rules; needs the point cache), not from the simplified
catalog geometry. Change those rules there, not in individual plots.

Direct provider sources live in `extra_sources.py` and are added to every
BedMap-based figure (bar charts, survey density, coastal gaps; `--no-extra`
disables them in the map scripts):

- **AWI** ice-sounder tracks (EMR, UWB, UWBM) from the AWI map server WFS
  layer `radar:tracks_3031`, dated per profile, 1994/95 to date. These
  replace the AWI BedMap files (`EXTERNAL_SOURCE_PREFIXES`). The layer lists
  the 60/600 ns EMR pulses and UWB standard/quicklook products as separate
  features; `load_awi_campaigns` keeps one per flight.
- **KOPRI/UTIG helicopter radar** (Amundsen Sea, layers ASE2-ASE6) from the
  QIceRadar index (Lindzey 2026, Zenodo 21964546 v0.3.0). Seasons are set in
  `KOPRI_SEASONS`; the raw data are not released.
- **xOPR substitutes** (`opr_tracks.py`): collections listed in
  `bedmap_common.OPR_SUBSTITUTIONS` are loaded from the CReSIS per-segment
  CSV flight tracks (real geometry, not the simplified STAC lines) in place of
  BedMap files that cannot be chained into flight lines (the 2009 Thwaites
  swath product), or for xOPR-only seasons not in BedMap3 (UTIG COLDEX
  2022/23 and 2023/24, GHOST2 2024/25). The institution comes from the STAC
  ``opr:provider`` field. `uv run python opr_tracks.py --verify` compares CSV track,
  STAC and BedMap lengths for the mapped collections.

Downloads are cached under `radar_cache/` (gitignored). Dated sources are
placed in their July-June season year; BedMap campaigns keep the calendar-year
spread of their metadata range (`campaign_years`).


### `plot_combined_availability.py`

Stacked bar chart comparing data availability across BedMap and xOPR by year (2001–2025). Categorizes data as open access (xOPR), committed to release (AWI, including data released outside xOPR; UTIG 2008+), or not released.

```bash
uv run python plot_combined_availability.py
# -> outputs/combined_data_availability.png
```

### `plot_survey_density.py`

Gridded survey density maps showing equivalent survey spacing. Supports BedMap and xOPR sources, Antarctica and Greenland regions, and optional zoom to sub-regions.

```bash
# BedMap Antarctica (remote catalog)
uv run python plot_survey_density.py --source bedmap_local --region antarctica

# xOPR Greenland
uv run python plot_survey_density.py --source xopr --region greenland

# Zoom to Amundsen Sea Embayment
uv run python plot_survey_density.py --source bedmap_local --region antarctica --zoom ase
```

Key options: `--grid-km` (cell size, default 30; use 0 for raw survey lines), `--zoom` (named sub-region: `ase`, `wilkes`, `aurora`), `--vmin`/`--vmax` (colorbar limits, default 0.1/10), `--target-spacing` (target survey spacing in km; switches to diverging PiYG colorscale).

Using `--source bedmap_local` uses BedMap data with local cache enabled (if not downloaded yet, it will be downloaded). You can also use `--source bedmap` to ignore any local cache and not create one.

### `plot_bedmap_availability.py`

Stacked bar chart of BedMap line-km per year, colored by country of origin.

```bash
uv run python plot_bedmap_availability.py
# -> outputs/bedmap_data_availability.png
uv run python plot_bedmap_availability.py --no-haps
# -> outputs/bedmap_data_availability_nohaps.png (without the HAPS capability line)
```

### `plot_opr_availability.py`

Stacked bar chart of xOPR Antarctic line-km per year, colored by institution.

```bash
uv run python plot_opr_availability.py
# -> outputs/opr_data_availability.png
```

### `coastal_survey_gaps.py`

Coastal survey density analysis by IMBIE drainage basin, with gap analysis showing additional line-km needed to reach a target resolution.

```bash
uv run python coastal_survey_gaps.py --source bedmap_local --coast-dist-km 20 --target-km 2
```

## CI / GitHub Pages

A GitHub Actions workflow (`.github/workflows/generate-figures.yml`) regenerates the key figures on each push to `main` and deploys them to GitHub Pages. Enable Pages (source: GitHub Actions) in the repo settings to activate.
