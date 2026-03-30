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

### `plot_combined_availability.py`

Stacked bar chart comparing data availability across BedMap and xOPR by year (2001–2023). Categorizes data as open access (xOPR), committed to release, or not released.

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

Key options: `--grid-km` (cell size, default 30), `--vmin`/`--vmax` (colorbar limits), `--target-spacing` (diverging colorscale around target).

Using `--source bedmap_local` uses BedMap data with local cache enabled (if not downloaded yet, it will be downloaded). You can also use `--source bedmap` to ignore any local cache and not create one.

### `plot_bedmap_availability.py`

Stacked bar chart of BedMap line-km per year, colored by country of origin.

```bash
uv run python plot_bedmap_availability.py
# -> outputs/bedmap_data_availability.png
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
