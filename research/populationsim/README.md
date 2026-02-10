# PopulationSim Exploration

**Purpose:** Population synthesis for agent-based travel demand modeling
**Repository:** <https://github.com/ActivitySim/populationsim>
**Documentation:** <https://activitysim.github.io/populationsim/>
**PyPI:** <https://pypi.org/project/populationsim/>

## What is PopulationSim?

PopulationSim is an open-source population synthesis tool that:

- Takes a **sample** of households/persons (e.g., 2% from surveys)
- Expands to **100% synthetic population** using control totals (Census)
- Uses **entropy maximization** and **iterative proportional fitting (IPF)**
- Supports **multiple geographic levels** (Region → PUMA → Tract → TAZ)

### Why We Need It

In our travel demand model:

- Each **agent** (person) needs to be represented
- We have **sample data** (Household Travel Surveys)
- We have **control totals** (Census demographics)
- PopulationSim bridges the gap → synthetic population for MATSim

## Installation

### Prerequisites

- Python 3.8+ (64-bit)
- Conda (Anaconda or Miniforge)

### Steps

```bash
# Create environment
conda create -n popsim python=3.8
conda activate popsim

# Install dependencies
conda install pytables

# Install PopulationSim
pip install populationsim
```

### Verify Installation

```bash
python -c "import populationsim; print(populationsim.__version__)"
```

## Core Concepts

### 1. Seed Sample

The input population sample (from HTS or PUMS):

- `seed_households.csv` - Household records with attributes
- `seed_persons.csv` - Person records linked to households

### 2. Control Totals

Census marginal distributions that the synthetic population must match:

- Total households by zone
- Age distribution
- Income distribution
- Household size distribution

### 3. Geographic Hierarchy

PopulationSim works with nested geographies:

```md
REGION (1 zone - entire study area)
   └── PUMA (Public Use Microdata Areas - seed level)
          └── TRACT (Census tracts)
                 └── TAZ (Traffic Analysis Zones)
```

### 4. Entropy Maximization

The algorithm finds weights that:

- Match control totals as closely as possible
- Keep weights close to original sample weights
- Minimize information loss

## Input Files

### Required Files

| File | Description | Key Columns |
| --- | --- | --- |
| `seed_households.csv` | Household sample | `hh_id`, `WGTP` (weight), `PUMA`, `NP` (persons), `HINCP` (income), `VEH` (vehicles) |
| `seed_persons.csv` | Person sample | `per_id`, `hh_id`, `AGEP` (age), `SEX`, `ESR` (employment) |
| `geo_crosswalk.csv` | Geographic mapping | `TAZ`, `TRACT`, `PUMA`, `REGION` |
| `control_totals_TAZ.csv` | TAZ-level controls | `TAZ`, `HHBASE`, demographic columns |
| `control_totals_TRACT.csv` | Tract-level controls | `TRACT`, demographic columns |
| `control_totals_REGION.csv` | Region-level controls | `REGION`, demographic columns |

### Example: seed_households.csv

```csv
hh_id,PUMA,WGTP,NP,HINCP,VEH,APTS,BLD,TEN
1001,00100,45,2,75000,1,0,3,1
1002,00100,52,4,125000,2,0,2,1
1003,00100,38,1,35000,0,1,5,2
```

### Example: seed_persons.csv

```csv
per_id,hh_id,AGEP,SEX,ESR,PINCP,SCHG
1,1001,35,1,1,45000,0
2,1001,32,2,1,30000,0
3,1002,42,1,1,80000,0
4,1002,40,2,1,45000,0
5,1002,15,1,0,0,5
6,1002,12,2,0,0,4
```

### Example: geo_crosswalk.csv

```csv
TAZ,TRACT,PUMA,REGION
101,10101,00100,1
102,10101,00100,1
103,10102,00100,1
201,20101,00200,1
```

### Example: control_totals_TAZ.csv

```csv
TAZ,HHBASE,PERSONS,HH_SIZE_1,HH_SIZE_2,HH_SIZE_3P
101,500,1200,150,200,150
102,350,850,100,150,100
103,200,500,60,80,60
```

## Configuration

### Directory Structure

```md
project/
├── run_populationsim.py
├── configs/
│   ├── settings.yaml      # Main configuration
│   ├── controls.csv       # Control specifications
│   └── logging.yaml       # Logging settings
├── data/
│   ├── seed_households.csv
│   ├── seed_persons.csv
│   ├── geo_crosswalk.csv
│   ├── control_totals_TAZ.csv
│   ├── control_totals_TRACT.csv
│   └── control_totals_REGION.csv
└── output/                # Generated after run
    ├── synthetic_households.csv
    ├── synthetic_persons.csv
    └── pipeline.h5
```

### settings.yaml

```yaml
# Geographic hierarchy (must be in order: largest → smallest)
geographies: [REGION, PUMA, TRACT, TAZ]
seed_geography: PUMA

# Column names
household_weight_col: WGTP
household_id_col: hh_id
total_hh_control: num_hh

# Algorithm settings
INTEGERIZE_WITH_BACKSTOPPED_CONTROLS: true
SUB_BALANCE_WITH_FLOAT_SEED_WEIGHTS: false
GROUP_BY_INCIDENCE_SIGNATURE: true
USE_SIMUL_INTEGERIZER: true
max_expansion_factor: 30
min_expansion_factor: 0.5
MAX_BALANCE_ITERATIONS_SIMULTANEOUS: 1000

# Input tables
input_table_list:
  - tablename: households
    filename: seed_households.csv
    index_col: hh_id
  - tablename: persons
    filename: seed_persons.csv
  - tablename: geo_cross_walk
    filename: geo_crosswalk.csv
  - tablename: TAZ_control_data
    filename: control_totals_TAZ.csv
  - tablename: TRACT_control_data
    filename: control_totals_TRACT.csv
  - tablename: REGION_control_data
    filename: control_totals_REGION.csv

# Output configuration
output_tables:
  action: include
  tables:
    - expanded_household_ids
    - synthetic_households
    - synthetic_persons

output_synthetic_population:
  household_id: hh_id
  households:
    filename: synthetic_households.csv
    columns:
      - NP
      - HINCP
      - VEH
      - AGEHOH
  persons:
    filename: synthetic_persons.csv
    columns:
      - AGEP
      - SEX
      - ESR

# Run steps
run_list:
  steps:
    - input_pre_processor
    - setup_data_structures
    - initial_seed_balancing
    - meta_control_factoring
    - final_seed_balancing
    - integerize_final_seed_weights
    - sub_balancing.geography=TRACT
    - sub_balancing.geography=TAZ
    - expand_households
    - write_results
    - summarize
```

### controls.csv

```csv
target,geography,seed_table,importance,control_field,expression
num_hh,TAZ,households,100000000,HHBASE,households.hh_id > 0
hh_size_1,TRACT,households,1000,HHSIZE1,households.NP == 1
hh_size_2,TRACT,households,1000,HHSIZE2,households.NP == 2
hh_size_3p,TRACT,households,1000,HHSIZE3P,households.NP >= 3
age_0_17,REGION,persons,500,AGE0_17,(persons.AGEP >= 0) & (persons.AGEP <= 17)
age_18_64,REGION,persons,500,AGE18_64,(persons.AGEP >= 18) & (persons.AGEP <= 64)
age_65p,REGION,persons,500,AGE65P,persons.AGEP >= 65
workers,TRACT,persons,1000,WORKERS,persons.ESR == 1
```

## Running PopulationSim

### Basic Run

```bash
cd project/
conda activate popsim

# Run with default configs
python run_populationsim.py

# Or specify paths
populationsim -c configs -d data -o output
```

### run_populationsim.py

```python
import populationsim

populationsim.run_populationsim(
    config_dir='configs',
    data_dir='data',
    output_dir='output'
)
```

### Multi-Processing

For large regions, use multiple processors:

```yaml
# In configs_mp/settings.yaml
inherit_settings: true
multiprocess: true
num_processes: 4
slice_geography: TRACT
```

```bash
python run_populationsim.py -c configs_mp -c configs
```

## Output Files

### synthetic_households.csv

Complete synthetic household population:

```csv
hh_id,TAZ,TRACT,PUMA,NP,HINCP,VEH,AGEHOH
1001_1,101,10101,00100,2,75000,1,35
1001_2,101,10101,00100,2,75000,1,35
1002_1,102,10101,00100,4,125000,2,42
...
```

### synthetic_persons.csv

Complete synthetic person population:

```csv
per_id,hh_id,AGEP,SEX,ESR
1_1,1001_1,35,1,1
2_1,1001_1,32,2,1
1_2,1001_2,35,1,1
...
```

### expanded_household_ids.csv

Mapping from seed to synthetic households:

```csv
hh_id,TAZ,num_hh
1001,101,2
1002,102,1
1003,101,0
...
```

## Integration with Our Platform

### Data Flow

```md
Census Data (controls)  ─┐
                         ├──▶ PopulationSim ──▶ Synthetic Population ──▶ MATSim
HTS Data (seed sample)  ─┘
```

### Airflow DAG Integration

```python
# dags/population_synthesis_dag.py

from airflow import DAG
from airflow.operators.python import PythonOperator
import populationsim

def run_popsim(**context):
    populationsim.run_populationsim(
        config_dir='/data/popsim/configs',
        data_dir='/data/popsim/data',
        output_dir='/data/popsim/output'
    )

dag = DAG('population_synthesis', ...)

popsim_task = PythonOperator(
    task_id='run_populationsim',
    python_callable=run_popsim,
    dag=dag
)
```

## Validation

### Key Metrics to Check

1. **Control Matching**
   - Compare synthetic vs. target totals
   - Check `summary_TAZ.csv`, `summary_TRACT.csv`

2. **Weight Distribution**
   - Examine expansion factors
   - Should be close to 1.0 on average

3. **Demographic Distributions**
   - Age pyramid
   - Income distribution
   - Household size distribution

### Validation Script

```python
import pandas as pd

# Load outputs
synth_hh = pd.read_csv('output/synthetic_households.csv')
synth_per = pd.read_csv('output/synthetic_persons.csv')
controls = pd.read_csv('data/control_totals_TAZ.csv')

# Compare totals
for taz in controls['TAZ'].unique():
    target = controls[controls['TAZ'] == taz]['HHBASE'].iloc[0]
    actual = len(synth_hh[synth_hh['TAZ'] == taz])
    error = (actual - target) / target * 100
    print(f"TAZ {taz}: Target={target}, Actual={actual}, Error={error:.1f}%")
```

## Common Issues

### 1. Zero-Person Households

- Filter out before running
- Check seed data quality

### 2. Control Mismatch

- Ensure control totals sum correctly across geographies
- Check geographic crosswalk is complete

### 3. Convergence Issues

- Increase `MAX_BALANCE_ITERATIONS_SIMULTANEOUS`
- Adjust importance weights
- Check for conflicting controls

### 4. Memory Issues

- Use multi-processing for large regions
- Reduce number of control variables

## Next Steps for Our Project

1. [ ] Obtain Census data for target city (Mumbai/Delhi)
2. [ ] Obtain or synthesize HTS seed sample
3. [ ] Create geographic crosswalk (TAZ ↔ Census tracts)
4. [ ] Build control totals from Census
5. [ ] Configure and test PopulationSim
6. [ ] Validate synthetic population
7. [ ] Convert output to MATSim format

## Resources

- [PopulationSim Documentation](https://activitysim.github.io/populationsim/)
- [GitHub Repository](https://github.com/ActivitySim/populationsim)
- [Example Projects](https://github.com/ActivitySim/populationsim/tree/master/example_calm)
- [SEMCOG Implementation](https://github.com/SEMCOG/SEMCOG_popsim)
- [Medium Article](https://medium.com/zephyrfoundation/populationsim-the-synthetic-commons-670e17383048)
