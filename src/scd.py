"""Selection-corrected decline using inverse-probability weighting — fully Bayesian."""

from pathlib import Path

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

from models import CHAINS, SAMPLER_KWARGS
from utils import get_age_mean, get_valid_pitch_types, load_data, setup_logger
from utils.plots import plot_scd_bars, plot_scd_curves

BASE_DIR = Path(__file__).resolve().parent.parent
MASTER_DATA_DIR = BASE_DIR / "master_data"
PLOTS_DIR = BASE_DIR / "plots"
OUTCOMES = ["mean_velo", "mean_spin_rate"]
SURVIVAL_FEATURES = ["mean_velo", "mean_spin_rate", "mean_pfx_z", "age"]

DRAWS_SCD = 500
TUNE_SCD = 1000
TARGET_ACCEPT_SCD = 0.95
NUTS_KWARGS_SCD = {"max_treedepth": 12}


def build_survival_dataset(df: pd.DataFrame) -> pd.DataFrame:
    pitcher_season = (
        df.groupby(["pitcher", "player_name", "age", "year"])
        .agg(
            mean_velo=("mean_velo", "mean"),
            mean_spin_rate=("mean_spin_rate", "mean"),
            mean_pfx_z=("mean_pfx_z", "mean"),
        )
        .reset_index()
        .sort_values(["pitcher", "age"])
    )

    pitcher_season["next_age"] = pitcher_season.groupby("pitcher")["age"].shift(-1)
    pitcher_season["survived"] = (
        pitcher_season["next_age"] == pitcher_season["age"] + 1
    ).astype(float)
    pitcher_season["is_last"] = pitcher_season["next_age"].isna()

    return pitcher_season


def fit_survival_model_bambi(
    pitcher_season: pd.DataFrame,
    features: list[str],
) -> tuple[az.InferenceData, pd.DataFrame, dict[str, float]]:
    """Bernoulli bambi survival model. Returns (idata, train_df, scale_params)."""
    train = (
        pitcher_season[~pitcher_season["is_last"]]
        .dropna(subset=features + ["survived"])
        .copy()
    )

    # Standardize features for better sampling; store params for prediction
    scale = {}
    for f in features:
        mu, sd = float(train[f].mean()), float(train[f].std())
        scale[f] = (mu, sd)
        train[f"{f}_sc"] = (train[f] - mu) / (sd + 1e-9)

    sc_features = [f"{f}_sc" for f in features]
    formula = "survived ~ " + " + ".join(sc_features)

    priors = {
        "Intercept": bmb.Prior("Normal", mu=0, sigma=2),
        **{f: bmb.Prior("Normal", mu=0, sigma=1) for f in sc_features},
    }
    model = bmb.Model(formula, train, family="bernoulli", link="logit", priors=priors)
    fit_kwargs = dict(
        draws=DRAWS_SCD,
        tune=TUNE_SCD,
        chains=CHAINS,
        target_accept=TARGET_ACCEPT_SCD,
        progressbar=False,
        nuts_sampler_kwargs=NUTS_KWARGS_SCD,
        **SAMPLER_KWARGS,
    )
    idata = model.fit(**fit_kwargs)
    return idata, train, scale


def compute_ipw_from_posterior(
    idata: az.InferenceData,
    pitcher_season: pd.DataFrame,
    features: list[str],
    scale: dict[str, float],
    trim_quantile: float = 0.90,
) -> pd.DataFrame:
    """Compute IPW from posterior mean logit scores."""
    valid = pitcher_season.dropna(subset=features).copy()

    # Apply same standardization used at fit time
    for f in features:
        mu, sd = scale[f]
        valid[f"{f}_sc"] = (valid[f] - mu) / (sd + 1e-9)

    sc_features = [f"{f}_sc" for f in features]

    # Posterior mean intercept and coefficients
    intercept = float(idata.posterior["Intercept"].values.mean())
    logit = np.full(len(valid), intercept)
    for f in sc_features:
        coef = float(idata.posterior[f].values.mean())
        logit += coef * valid[f].values

    p_survive = 1.0 / (1.0 + np.exp(-logit))
    p_survive = np.clip(p_survive, 0.1, 0.9)

    valid["p_survive"] = p_survive
    valid["ipw"] = 1.0 / p_survive

    cap = valid["ipw"].quantile(trim_quantile)
    valid["ipw"] = valid["ipw"].clip(upper=cap)

    return valid[["pitcher", "age", "ipw"]]


def fit_weighted_aging_model(
    pt_df: pd.DataFrame,
    outcome: str,
) -> az.InferenceData:
    """Bambi mixed model with IPW applied via integer row expansion."""
    model_df = pt_df.dropna(subset=[outcome, "age_c", "age_c_sq", "ipw"]).copy()
    model_df["ipw_int"] = (model_df["ipw"] * 100).round().astype(int).clip(lower=1)
    expanded = model_df.loc[model_df.index.repeat(model_df["ipw_int"])].reset_index(
        drop=True
    )

    y_sd = float(expanded[outcome].std())
    formula = f"{outcome} ~ age_c + age_c_sq + C(year) + (1|pitcher)"
    priors = {
        "Intercept": bmb.Prior(
            "Normal", mu=float(expanded[outcome].mean()), sigma=2 * y_sd
        ),
        "age_c": bmb.Prior("Normal", mu=0, sigma=y_sd),
        "age_c_sq": bmb.Prior("Normal", mu=0, sigma=y_sd / 2),
        "1|pitcher": bmb.Prior(
            "Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=y_sd)
        ),
        "sigma": bmb.Prior("HalfNormal", sigma=y_sd),
    }
    model = bmb.Model(formula, expanded, family="gaussian", priors=priors)
    fit_kwargs = dict(
        draws=DRAWS_SCD,
        tune=TUNE_SCD,
        chains=CHAINS,
        target_accept=TARGET_ACCEPT_SCD,
        progressbar=False,
        nuts_sampler_kwargs=NUTS_KWARGS_SCD,
        **SAMPLER_KWARGS,
    )
    return model.fit(**fit_kwargs)


def compute_scd_posterior(
    idata_naive: az.InferenceData,
    idata_corrected: az.InferenceData,
    age_mean: float,
    eval_age: float = 35,
) -> dict:
    """Posterior SCD = naive decline rate − corrected decline rate at eval_age."""
    age_c_eval = eval_age - age_mean

    def get_rate_samples(idata: az.InferenceData) -> np.ndarray:
        b1 = idata.posterior["age_c"].values.flatten()
        try:
            b2 = idata.posterior["age_c_sq"].values.flatten()
            return b1 + 2 * b2 * age_c_eval
        except KeyError:
            return b1

    naive_rate = get_rate_samples(idata_naive)
    corrected_rate = get_rate_samples(idata_corrected)
    scd_samples = naive_rate - corrected_rate

    hdi_scd = az.hdi(scd_samples, hdi_prob=0.95)

    naive_mean = float(naive_rate.mean())
    corrected_mean = float(corrected_rate.mean())
    scd_mean = float(scd_samples.mean())

    scd_pct = (
        round(scd_mean / abs(corrected_mean) * 100, 2)
        if abs(corrected_mean) > 1e-9
        else float("nan")
    )

    return {
        "eval_age": eval_age,
        "naive_rate": round(naive_mean, 4),
        "corrected_rate": round(corrected_mean, 4),
        "scd": round(scd_mean, 4),
        "scd_hdi_lo": round(float(hdi_scd[0]), 4),
        "scd_hdi_hi": round(float(hdi_scd[1]), 4),
        "scd_pct": scd_pct,
    }


if __name__ == "__main__":
    log = setup_logger("scd", MASTER_DATA_DIR / "scd.log")

    df = load_data()
    age_mean = get_age_mean(df)

    pitcher_season = build_survival_dataset(df)
    survival_rate = pitcher_season[~pitcher_season["is_last"]]["survived"].mean()
    log.info(f"Survival rate: {survival_rate * 100:.1f}% of pitcher-seasons")
    log.info(
        f"Pitcher-season rows: {len(pitcher_season)} | unique pitchers: {pitcher_season['pitcher'].nunique()}"
    )

    log.info("Fitting Bernoulli survival model (bambi)...")
    idata_survival, train_df, scale = fit_survival_model_bambi(
        pitcher_season, SURVIVAL_FEATURES
    )

    ipw_df = compute_ipw_from_posterior(
        idata_survival, pitcher_season, SURVIVAL_FEATURES, scale
    )
    p_survive_vals = 1.0 / ipw_df["ipw"]
    log.info(
        f"Survival prob range: {p_survive_vals.min():.3f} - {p_survive_vals.max():.3f} | "
        f"std: {p_survive_vals.std():.4f}"
    )

    all_scd = []

    for pt in get_valid_pitch_types():
        if pt == "FC":
            continue
        log.info(f"\nComputing SCD for {pt}...")
        pt_df = df[df["pitch_type"] == pt].copy()
        pt_df = pt_df.merge(ipw_df, on=["pitcher", "age"], how="left")
        pt_df["ipw"] = pt_df["ipw"].fillna(1.0)

        log.info(
            f"  IPW stats: mean={pt_df['ipw'].mean():.3f} "
            f"max={pt_df['ipw'].max():.3f} "
            f"std={pt_df['ipw'].std():.4f}"
        )

        for outcome in OUTCOMES:
            log.info(f"  Fitting naive model: {pt} {outcome}")
            naive_df = pt_df.copy()
            naive_df["ipw"] = 1.0
            idata_naive = fit_weighted_aging_model(naive_df, outcome)

            log.info(f"  Fitting IPW-corrected model: {pt} {outcome}")
            idata_corrected = fit_weighted_aging_model(pt_df, outcome)

            for eval_age in [32, 35, 38]:
                scd = compute_scd_posterior(
                    idata_naive, idata_corrected, age_mean, eval_age=eval_age
                )
                all_scd.append({"pitch_type": pt, "outcome": outcome, **scd})
                log.info(
                    f"  SCD at {eval_age}: naive={scd['naive_rate']:+.4f} "
                    f"corrected={scd['corrected_rate']:+.4f} "
                    f"SCD={scd['scd']:+.4f} ({scd['scd_pct']:+.1f}%)"
                )

    scd_df = pd.DataFrame(all_scd)
    out_path = MASTER_DATA_DIR / "scd_results.csv"
    scd_df.to_csv(out_path, index=False)
    log.info(f"\nSaved SCD results: {out_path}")
    log.info("\nSCD Summary:\n" + scd_df.to_string(index=False))

    try:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plot_scd_curves(scd_df, "mean_velo", PLOTS_DIR)
        plot_scd_bars(scd_df, "mean_spin_rate", PLOTS_DIR)
        log.info(f"Saved plots to {PLOTS_DIR}")
    except Exception as e:
        log.error(f"Plot generation failed: {e}")
