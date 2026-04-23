# MLB Pitch Aging Study

**Aging Curves for Pitcher "Stuff" Using Statcast Data (2015–2025)**

By: Ethan Wong

---

## Overview

This project builds aging curves for MLB pitcher **stuff** (velocity, spin rate, and movement) using Statcast pitch-level data from 2015 to 2025. We use Bayesian multilevel modeling to provide among the first publicly documented Statcast-era aging curves with full posterior uncertainty on peak age estimates.

**Status:** Updated April 22, 2026 to match current code and generated outputs.

**Key findings:**
- Velocity decline at age 28 is strongest for SI/FF and weakest for FC (Table 1 point estimates: FF -0.156, SL -0.058, SI -0.189, CH -0.052, CU -0.047, FC -0.022 mph/year).
- Spin peaks later than velocity for most pitch types (Table 2 spin peak ages: FF 29.99, SL 31.41, SI 30.11, CH 25.84, CU 32.01).
- The **Stuff Compensation Gap (SCG)**, defined as spin peak age minus velocity peak age, ranges from -0.63 (CH) to +11.74 (SI) in univariate base models.
- Bivariate modeling of velocity and spin reveals a positive pitcher-level correlation (FF rho_mean=0.307 [0.243, 0.368], SI rho_mean=0.334 [0.260, 0.410]).
- Naive cross-sectional aging curves exhibit survivorship bias at both career boundaries; mixed-effects models were employed to partially correct for this
- *Note:* Spin axis analyses were explicitly removed from this study, as standard arithmetic models are systematically inappropriate for bounded circular directional data.

## Current Results Snapshot

Main paper CSV outputs are written under `tables/` by `src/tables.py`:

- `table1_velo_decline_rates.csv`: velocity decline rates at age 28 with 95% CIs
- `table2_spin_peak_ages.csv`: spin peak ages with 95% CIs
- `table3_bivariate_correlation.csv`: pitcher-level velocity/spin correlation from bivariate models

Selected values from current outputs:

| Metric | Current value |
|--------|---------------|
| Fastball (FF) velocity decline at age 28 | -0.156 mph/year (95% CI: -0.191, -0.124) |
| Sinker (SI) velocity decline at age 28 | -0.189 mph/year (95% CI: -0.233, -0.138) |
| Fastball (FF) spin peak age | 29.99 (95% CI: 28.54, 31.54) |
| Curveball (CU) spin peak age | 32.01 (95% CI: 27.85, 36.72) |
| Bivariate rho (FF) | 0.307 (95% HDI: 0.243, 0.368) |
| Bivariate rho (SI) | 0.334 (95% HDI: 0.260, 0.410) |

Univariate SCG summary (`master_data/scg_results.csv`): FF 8.71, SL 5.61, SI 11.74, CH -0.63, CU 5.37, FC 1.83.
Bivariate SCG summary (`master_data/scg_bivariate_results.csv`): FF 8.60 [6.40, 10.91], SI 12.05 [8.31, 15.89].

---

## Repository Structure

```
mlb-pitch-aging/
├── data/
│   └── pitching_stats_{year}.parquet     # per-year Statcast aggregates
├── master_data/
│   ├── pitching_master.csv               # full dataset with age
│   ├── model_results.csv                 # posterior summaries (all models)
│   ├── model_results.log                 # univariate model run log
│   ├── peak_age_posteriors_base.csv      # posterior peak age HDIs (base)
│   ├── peak_age_posteriors_with_ext.csv  # posterior peak age HDIs (with_ext)
│   ├── decline_rate_posteriors_base.csv  # decline rate posteriors (base)
│   ├── decline_rate_posteriors_with_ext.csv
│   ├── scg_results.csv                   # SCG univariate
│   ├── scg_bivariate_results.csv         # SCG bivariate
│   ├── bivariate_base_FF.nc              # bivariate posterior (FF)
│   ├── bivariate_base_SI.nc              # bivariate posterior (SI)
│   └── fitted_idatas/                    # per-model posterior netCDF outputs
├── logs/                                 # full pipeline execution logs
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

Run scripts individually, or use the convenience script:

```bash
./scripts/run_full_pipeline.sh
```

`run_full_pipeline.sh` will:
- auto-activate the conda env (`mlb-pitch-aging`, configurable via `PIPELINE_CONDA_ENV`)
- keep macOS awake during long runs via `caffeinate` when available
- write a timestamped log file to `logs/full_pipeline_*.log`

Or run steps individually:

```bash
# 1. Pull Statcast data (slow — hits pybaseball API per year)
python src/data.py

# 2. Join birth years, compute age, build master dataset
python src/prepare.py

# (Optional) Generate Exploratory Data Analysis (EDA) plots
python src/eda_plots.py

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
> if `master_data/pitching_master.csv` and the needed files in `master_data/fitted_idatas/` are already present.

---

## Models

### Primary — Bayesian Multilevel (Bambi/PyMC)

```
y_it = β₀ + β₁·age_c + β₂·age_c² + u_i + v_t + ε_it
u_i ~ N(0, σ²_u)   (pitcher random intercept)
v_t ~ N(0, σ²_v)   (year random intercept — partial pooling over seasons)
ε_it ~ StudentT(ν, 0, σ)
```

- One model per pitch type × outcome (5 base outcomes and 5 with_ext outcomes, spin axis excluded)
- `age_c = age − mean(age)` (centered at the global sample mean, ≈28.9)
- Year random intercept absorbs secular trends across pitching eras (e.g., the 2021 sticky-stuff crackdown)
- `with_ext` experiment adds `mean_ext_c` as a fixed covariate. Extension is re-centered **per pitch type on the final analysis sample** (after the minimum-seasons filter), so `mean(mean_ext_c) = 0` in the exact data fed to each model. The base and with_ext models are fitted on independent samples: the base model does not require extension to be non-missing, preventing selection bias.
- Two-pass screening: 500-draw screen pass (+500 tune) to check significance; 2000-draw full fit (+4000 tune) only if any age coefficient's 95% HDI excludes zero
- Linear fallback: if the `age_c_sq` 95% HDI spans zero, a linear model is also fitted and PSIS-LOO is compared. The linear model is selected only if its ELPD-LOO exceeds the quadratic by more than 4 units; otherwise the quadratic is retained.

Priors (weakly informative, scaled to each outcome's SD):
```
β₀         ~ N(ȳ, 2·SD(y))
β₁         ~ N(0, SD(y)/4)
β₂         ~ N(0, SD(y)/2)
β_ext      ~ N(0, SD(y))          [with_ext experiment only]
σ_u, σ_v   ~ HalfNormal(SD(y))
σ          ~ HalfNormal(SD(y))
ν          ~ Gamma(2, 0.1)
```

### Peak Age Estimation (Cauchy-Ratio Correction)

Peak ages are derived directly from the MCMC posterior samples via the quadratic vertex formula:

```
peak_age = age_mean + (−β₁ / 2β₂)
```

Because `−β₁ / (2β₂)` is a ratio of two approximately normal quantities, its distribution is heavy-tailed (Cauchy-like) when β₂ is near zero or has mixed sign. To produce valid estimates:

1. A peak age is only computed if the **majority (> 50%) of posterior draws** are physically valid (`β₂ < 0` indicating a maximum).
2. Only the draws where `β₂ < 0` are evaluated.
3. Computed peak ages are restricted to the plausible range [15, 50].
4. At least 100 filtered draws are required; otherwise no peak age is reported.
5. The **posterior median** is reported (not the mean), since the mean is undefined or unstable for Cauchy-like distributions.

### Bivariate — PyMC

Joint model for velocity and spin rate (FF and SI only). Velo and spin are modeled as **independent Student-t observations**, but their pitcher-level random intercepts are jointly modeled via a correlated bivariate structure:

```
velo_obs_it ~ StudentT(ν_velo, μ_velo_it, σ_velo)
spin_obs_it ~ StudentT(ν_spin, μ_spin_it, σ_spin)

μ_velo_it = β₀_velo + u_i[0] + v_t[0] + β₁_velo·age_c + β₂_velo·age_c²
μ_spin_it = β₀_spin + u_i[1] + v_t[1] + β₁_spin·age_c + β₂_spin·age_c²

[u_i[0], u_i[1]] ~ MVN(0, Σ_pitcher)   (LKJ Cholesky, non-centered)
[v_t[0], v_t[1]] ~ MVN(0, Σ_year)      (LKJ Cholesky, non-centered)
```

The off-diagonal element of `Σ_pitcher` is the pitcher-level velocity/spin correlation ρ, estimated via the LKJ prior (`η=2`). Non-centered parameterization (`z ~ N(0,1)`, `u = z @ chol.T`) is used to avoid funnel geometry. Year random effects are also correlated across outcomes.

Both outcomes use `ν ~ Gamma(2, 0.1)` for robustness to outlier pitcher-seasons, matching the univariate model's StudentT likelihood.

---

## Key Metrics

**Stuff Compensation Gap (SCG)**
```
SCG = spin_peak_age − velocity_peak_age
```
Measures the career window where spin continues developing despite velocity decline.
Positive SCG = active compensation mechanism. Bivariate peak ages are derived from joint posterior samples (draws where both β₂_velo < 0 and β₂_spin < 0), preserving their correlation. SCG point estimates use the posterior median; 95% HDI computed via ArviZ.

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
When bridging Lahman and Chadwick IDs, defensive deduplication (`drop_duplicates`) is enforced to prevent rare many-to-one MLBAM keys from silently inflating pitcher-season observation counts.

---

## Environment

```bash
conda env create -f environment.yml
conda activate mlb-pitch-aging
```

Optional update after dependency changes:

```bash
conda env update -n mlb-pitch-aging -f environment.yml --prune
```

Tested on Python 3.12, macOS (Apple Silicon).

---

## Acknowledgements

Statcast data via [pybaseball](https://github.com/jldbc/pybaseball).
Lahman database via [SABR](https://sabr.org).
Methodology informed by Nguyen & Matthews (JSA 2024) and Albert (2023).
