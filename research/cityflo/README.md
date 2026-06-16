# Cityflo Mobility Analysis

GPS-based travel demand pipeline for Cityflo's Mumbai bus network. Processes legacy vehicle location pings into OD matrices, service reliability metrics, and model-ready travel demand features.

**Study window:** September 2021 – October 2022  
**Data:** ~165M GPS pings, 6,115 stops, 135,125 scheduled trips, 15-point weather grid

Full documentation:

- [`docs/cityflo/dataset.md`](../../docs/cityflo/dataset.md) — data schemas, EDA findings, cleaning decisions
- [`docs/cityflo/methodology.md`](../../docs/cityflo/methodology.md) — pipeline design, script breakdown, configuration reference

## Repository Layout

```md
research/cityflo/
├── data/
│   ├── raw/            original GPS CSVs, stops, trips, weather grid
│   ├── processed/      cleaned reference files and pipeline parquet outputs
│   └── interim/        intermediate bucket parquets
├── notebooks/          EDA and preprocessing notebooks
├── outputs/
│   ├── figures/        model plots and SHAP summaries
│   ├── models/         saved model files
│   └── tables/         predictions and evaluation metrics
├── scripts/            pipeline scripts (01–12) and config.py
├── slurm/              SLURM job scripts for HPC execution
└── req.txt             Python dependencies
```

## Setup

### Recommended: uv

[`uv`](https://docs.astral.sh/uv/) is a Python package and environment manager. It is generally faster than a standard `pip` workflow for creating environments and installing dependencies.

<details>
<summary>macOS / Linux</summary>

```bash
cd research/cityflo

uv venv
source .venv/bin/activate

uv pip install -r req.txt
```

</details>

<details>
<summary>Windows (PowerShell)</summary>

```powershell
cd research/cityflo

uv venv
.venv\Scripts\Activate.ps1

uv pip install -r req.txt
```

</details>

### Alternative: pip

<details>
<summary>macOS / Linux</summary>

```bash
cd research/cityflo

python -m venv .venv
source .venv/bin/activate

pip install -r req.txt
```

</details>

<details>
<summary>Windows (PowerShell)</summary>

```powershell
cd research/cityflo

python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r req.txt
```

</details>

## Running the Pipeline

Before running any script, the two EDA notebooks must be run in order — they produce `stops_clean.csv` and `trips_clean.csv` which are inputs to every downstream script.

```bash
jupyter notebook notebooks/01_reference_data_audit.ipynb
jupyter notebook notebooks/02_gps_data_audit.ipynb
```

The pipeline scripts then run in the following order. Scripts 02 and 03–05 are independent and can run in parallel once `pings_clean.parquet` is available.

```bash
# Route catalog (fast, no GPS dependency — run first)
python scripts/02_route_catalog.py

# GPS ingestion (single-machine)
python scripts/01_1_ingest_legacy.py --bucket_id 0 --bucket_count 1
python scripts/01_2_finalize_pings.py --bucket_id 0
python scripts/01_3_merge_buckets.py

# Or on HPC (8 parallel buckets):
# for i in $(seq 0 7); do sbatch slurm/01_ingest_bucket.sh $i 8; done
# sbatch slurm/01_merge.sh

# Core pipeline
python scripts/03_trip_segmentation.py
python scripts/04_stop_snapping.py
python scripts/05_route_inference.py

# Run 06 and 07 independently (no dependency between them)
python scripts/06_od_matrix.py
python scripts/07_reliability.py

python scripts/08_weather_consolidate.py
python scripts/09_feature_engineering.py
python scripts/10_ward_aggregation.py

# Models
python scripts/11_model_nb.py
python scripts/12_model_xgboost.py
```

To validate route inference accuracy before running on GPS data:

```bash
python scripts/05_route_inference.py --run_validation
```

All parameters are in `scripts/config.py`. Changing a parameter requires re-running from the first script that reads it — the methodology doc has the dependency chain.
