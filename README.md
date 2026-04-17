# MLB Pitch Aging

Modeling how pitcher "stuff" (velocity, spin rate, movement) changes with age using Statcast data.

## Data Sources

> A prebuilt master dataset is provided in this repository. The instructions below are only needed if you want to replicate the data pipeline from scratch.

### Statcast
Pulled automatically via `pybaseball` when running `src/data.py`. No manual download needed.

If pybaseball is failing, download CSVs manually from Baseball Savant:
1. Go to https://baseballsavant.mlb.com/statcast_search
2. Set filters: player type = pitcher, date range = full season
3. Download CSV — note the 40,000 row limit per download, so multiple pulls per season are required

### Lahman Database
Used for player birth years to compute age. PyBaseball's Lahman functions are currently broken — download directly:
1. Go to https://sabr.org/lahman-database/
2. Download the latest version (CSV format)
3. Extract `People.csv` and place it in `data/`

`People.csv` is bridged to Statcast MLBAM IDs via `chadwick_register()` from pybaseball. The Chadwick register source data is also available at https://github.com/chadwickbureau/register.

## Reproducing the Data Pipeline

```bash
# 1. Pull Statcast data (2015-2025) — takes ~2-3 hours if parallel = False
python src/data.py

# 2. Build master dataset with age.
python src/prepare.py
```