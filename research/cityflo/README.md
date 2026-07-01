# Cityflo Mobility Analysis

GPS-based travel demand pipeline for Cityflo's Mumbai bus network. Processes legacy vehicle location pings into OD matrices, service reliability metrics, and model-ready travel demand features — then trains and evaluates three forecasting models.

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
│   ├── figures/        model plots, EDA dashboards, spatial maps
│   ├── models/         saved model files and metadata JSON
│   └── tables/         predictions, evaluation metrics, network summary
├── scripts/            pipeline scripts (01–15) and config.py
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

The pipeline scripts then run in the following order. Scripts 06 and 07 are independent once `segments_inferred.parquet` is available and can run in parallel.

```bash
# Route catalog (fast, no GPS dependency — run first)
python scripts/02_route_catalog.py

# GPS ingestion (single-machine, all data in one bucket)
python scripts/01_1_ingest_legacy.py --bucket_id 0 --bucket_count 1
python scripts/01_2_finalize_pings.py --bucket_id 0
python scripts/01_3_merge_buckets.py

# Or on HPC (8 parallel buckets via SLURM):
# sbatch --array=0-7 slurm/01_1_ingest_legacy.slurm
# sbatch --dependency=afterok:... --array=0-7 slurm/01_2_finalize_pings.slurm
# sbatch --dependency=afterok:... slurm/01_3_merge_buckets.slurm
# (see slurm/run_pipeline.sh for the full dependency chain)

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

# Models (can run independently once features_master.parquet exists)
python scripts/11_model_nb.py
python scripts/12_model_xgboost.py
python scripts/13_model_stgnn.py

# Post-modelling analysis and policy outputs
python scripts/14_analysis_reporting.py
python scripts/15_policy_outputs.py
```

To validate route inference accuracy before running on GPS data:

```bash
python scripts/05_route_inference.py --run_validation
```

## HPC / SLURM Execution

The `slurm/` directory contains `.slurm` job scripts for every pipeline stage. [`slurm/run_pipeline.sh`](./slurm/run_pipeline.sh) is an orchestration script that submits all jobs in dependency order — it is the recommended entry point on an HPC cluster.

```bash
# Submit the full pipeline
bash slurm/run_pipeline.sh
```

Individual scripts can also be submitted standalone:

```bash
sbatch slurm/09_feature_engineering.slurm
sbatch slurm/13_model_stgnn.slurm
```

All parameters are in `scripts/config.py`. Changing a parameter requires re-running from the first script that reads it — the methodology doc has the dependency chain.
