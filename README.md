# MLB Pitch Aging Study

**Aging Curves for Pitcher "Stuff" Using Statcast Data (2015–2025)**

By: Ethan Wong

---

## Overview

This project builds aging curves for MLB pitcher **stuff** (velocity, spin rate, and movement) using Statcast pitch-level data from 2015 to 2025. We introduce Bayesian multilevel modeling to this domain, providing the first Statcast-era aging curves with full posterior uncertainty on peak age estimates.

**Key findings:**
- Velocity declines monotonically throughout observable MLB careers (−0.13 to −0.26 mph/yr depending on pitch type)
- Spin rate peaks mid-career (ages 25–32), *after* velocity has already begun declining showing evidence of an active compensation mechanism
- The **Stuff Compensation Gap (SCG)**: spin peak age minus velocity peak age which ranges from −0.5 years (changeup) to +8.3 years (sinker, bivariate estimate)
- Bivariate modeling of velocity and spin reveals a positive pitcher-level correlation (ρ ≈ 0.32–0.34), indicating harder throwers naturally spin more
- Naive cross-sectional aging curves exhibit survivorship bias at both career boundaries so used mixed-effects models to correct for this

---

## Repository Structure

```
mlb-pitch-aging/
├── data/
│   └── pitching_stats_{year}.parquet     # per-year Statcast aggregates
├── master_data/
│   ├── pitching_master.csv               # full dataset with age
│   ├── model_results.csv                 # posterior summaries (all models)
│   ├── peak_age_posteriors_base.csv      # posterior peak age HDIs (base)
│   ├── peak_age_posteriors_with_ext.csv  # posterior peak age HDIs (with_ext)
│   ├── decline_rate_posteriors_base.csv  # decline rate posteriors (base)
│   ├── decline_rate_posteriors_with_ext.csv
│   ├── scg_results.csv                   # SCG univariate
│   └── scg_bivariate_results.csv         # SCG bivariate
├── tables/
│   ├── table1_velo_decline_rates.csv
│   ├── table2_spin_peak_ages.csv
│   ├── table3_bivariate_correlation.csv
│   ├── table_s1_model_results.csv
│   ├── table_s2_peak_age_cis.csv
│   ├── table_s3_decline_rate_cis.csv
│   └── table_s4_scg_summary.csv
├── plots/                                # all figures
└── src/
    ├── data.py                           # Statcast pull via pybaseball
    ├── prepare.py                        # age join + master dataset
    ├── models.py                         # Bayesian multilevel models (primary)
    ├── inference.py                      # posterior extraction (peak ages, decline rates)
    ├── bivariate.py                      # PyMC bivariate velo/spin model
    ├── scg.py                            # Stuff Compensation Gap
    ├── tables.py                         # paper tables
    ├── eda_plots.py                      # exploratory analysis
    └── utils/
        ├── __init__.py
        ├── utils.py                      # shared constants + helpers
        ├── plots.py                      # all plotting functions
        └── sampling.py                   # Bambi/PyMC sampler config
```

---

## Pipeline

Run scripts individually, or use the convenience script (requires the conda environment to be active):

```bash
conda activate mlb-pitch-aging
./scripts/run_full_pipeline.sh
```

Or run steps individually:

```bash
# 1. Pull Statcast data (slow — hits pybaseball API per year)
python src/data.py

# 2. Join birth years, compute age, build master dataset
python src/prepare.py

# 3. Fit univariate Bayesian mixed-effects models — primary results
python src/models.py

# 4. Extract posterior peak age HDIs and decline rates
python src/inference.py

# 5. Bivariate PyMC model — FF and SI only (velo + spin jointly)
python src/bivariate.py

# 6. Stuff Compensation Gap
python src/scg.py

# 7. Paper tables
python src/tables.py
```

> **Note:** `models.py` is the bottleneck (~2.5 hrs for Bayesian screen+full passes across all models).
> `bivariate.py` takes ~20–30 min for MCMC. All other scripts are fast.
> If re-running after code fixes, you can skip `data.py`, `prepare.py`, and `models.py`
> if `pitching_master.csv` and `fitted_idatas.pkl` are already present.

---

## Models

### Primary — Bayesian Multilevel (Bambi/PyMC)

```
y_it = β₀ + β₁·age_c + β₂·age_c² + u_i + v_t + ε_it
u_i ~ N(0, σ²_u)   (pitcher random intercept)
v_t ~ N(0, σ²_v)   (year random intercept)
ε_it ~ StudentT(ν, 0, σ)
```

- One model per pitch type × outcome (6 × 6 = 36 base, 36 with_ext = 72 combinations)
- `age_c = age − 28.89` (centered at sample mean)
- Year random intercept absorbs secular trends including 2021 foreign substance crackdown
- `with_ext` experiment adds `mean_ext` (release extension) as a fixed covariate
- Two-pass screening: 500-draw screen pass to check significance, 2000-draw full fit only if significant
- Linear fallback if `age_c_sq` 95% HDI spans zero and LOO favors the linear model

Priors (weakly informative, scaled to each outcome's SD):
```
β₀         ~ N(ȳ, 2·SD(y))
β₁         ~ N(0, SD(y)/4)
β₂         ~ N(0, SD(y)/2)
σ_u, σ_v   ~ HalfNormal(SD(y))
σ          ~ HalfNormal(SD(y))
ν          ~ Gamma(2, 0.1)
```

### Bivariate — PyMC

Joint model for velocity and spin rate (FF and SI only):

```
[velo, spin] ~ MVN(μ, Σ)
Σ = diag(σ) @ Corr @ diag(σ)
```

Estimates pitcher-level correlation ρ between velocity and spin random effects
via LKJ Cholesky parameterization with non-centered random effects.

---

## Key Metrics

**Stuff Compensation Gap (SCG)**
```
SCG = spin_peak_age − velocity_peak_age
```
Measures the career window where spin continues developing despite velocity decline.
Positive SCG = active compensation mechanism. Bivariate estimates used for FF and SI;
univariate for remaining pitch types.

---

## Data Sources

| Source | Access | Purpose |
|--------|--------|---------|
| Statcast | `pybaseball.statcast()` | Pitch-level metrics 2015–2025 |
| Lahman `People.csv` | Manual download from sabr.org | Birth years for age computation |
| Chadwick register | `pybaseball.chadwick_register()` | MLBAM → bbref ID bridge |

Minimum 50 pitches per pitcher × season × pitch type to filter position players.
Minimum 3 distinct seasons per pitcher for model inclusion. Regular season only.
2020 COVID shortened season (60 games) retained with year random effect.

---

## Environment

```bash
conda create -n mlb-pitch-aging python=3.12
conda activate mlb-pitch-aging
pip install pybaseball statsmodels bambi pymc arviz pandas matplotlib seaborn
```

Tested on Python 3.12, macOS (Apple Silicon). All sampling runs on CPU via PyMC.

---

## Acknowledgements

Statcast data via [pybaseball](https://github.com/jldbc/pybaseball).
Lahman database via [SABR](https://sabr.org).
Methodology informed by Nguyen & Matthews (JSA 2024) and Albert (2023).
