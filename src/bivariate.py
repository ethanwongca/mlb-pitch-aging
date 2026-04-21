"""Bivariate mixed model for joint velocity/spin aging."""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from pathlib import Path

from utils import (
    filter_pitchers_by_min_distinct_seasons,
    get_age_mean,
    get_pymc_sampler_kwargs,
    load_data,
    setup_logger,
)
from utils.plots import plot_bivariate_correlation, plot_bivariate_peak_comparison

BASE_DIR = Path(__file__).resolve().parent.parent
MASTER_DATA_DIR = BASE_DIR / "master_data"
PLOTS_DIR = BASE_DIR / "plots"
PITCH_TYPES = ["FF", "SI"]
MIN_SEASONS = 3
CHAINS = 4
DRAWS_BIVARIATE = 3000
TUNE_BIVARIATE = 3000
TARGET_ACCEPT_BIVARIATE = 0.95
NUTS_KWARGS_BIVARIATE = {"max_treedepth": 12}


def prepare_bivariate_data(
    df: pd.DataFrame,
    pitch_type: str,
    outcome_velo: str = "mean_velo",
    outcome_spin: str = "mean_spin_rate",
) -> dict:
    pt_df = (
        df[df["pitch_type"] == pitch_type]
        .dropna(subset=[outcome_velo, outcome_spin, "age_c", "age_c_sq"])
        .copy()
    )

    pt_df = filter_pitchers_by_min_distinct_seasons(pt_df, MIN_SEASONS)

    pitchers = pt_df["pitcher"].unique()
    pitcher_idx = {pid: i for i, pid in enumerate(pitchers)}
    pt_df["pitcher_idx"] = pt_df["pitcher"].map(pitcher_idx)

    return {
        "df": pt_df,
        "pitchers": pitchers,
        "pitcher_idx": pt_df["pitcher_idx"].values,
        "age_c": pt_df["age_c"].values,
        "age_c_sq": pt_df["age_c_sq"].values,
        "velo": pt_df[outcome_velo].values,
        "spin": pt_df[outcome_spin].values,
        "n_pitchers": len(pitchers),
        "n_obs": len(pt_df),
    }


def build_bivariate_model(data: dict) -> pm.Model:
    with pm.Model() as model:
        b0_velo = pm.Normal("b0_velo", mu=data["velo"].mean(), sigma=5)
        b1_velo = pm.Normal("b1_velo", mu=0, sigma=1)
        b2_velo = pm.Normal("b2_velo", mu=0, sigma=0.5)

        b0_spin = pm.Normal("b0_spin", mu=data["spin"].mean(), sigma=100)
        b1_spin = pm.Normal("b1_spin", mu=0, sigma=50)
        b2_spin = pm.Normal("b2_spin", mu=0, sigma=20)

        sd_dist = pm.HalfNormal.dist(sigma=np.array([2.0, 100.0]))
        chol, _, _ = pm.LKJCholeskyCov(
            "chol", n=2, eta=2, sd_dist=sd_dist, compute_corr=True
        )

        # Non-centered parameterization: avoids funnel geometry in large hierarchies
        z = pm.Normal("z", mu=0, sigma=1, shape=(data["n_pitchers"], 2))
        u = pm.Deterministic("u", pm.math.dot(z, chol.T))

        age_c = data["age_c"]
        age_c_sq = data["age_c_sq"]
        pidx = data["pitcher_idx"]

        mu_velo = b0_velo + u[pidx, 0] + b1_velo * age_c + b2_velo * age_c_sq
        mu_spin = b0_spin + u[pidx, 1] + b1_spin * age_c + b2_spin * age_c_sq

        sigma_velo = pm.HalfNormal("sigma_velo", sigma=2)
        sigma_spin = pm.HalfNormal("sigma_spin", sigma=50)

        pm.Normal("velo_obs", mu=mu_velo, sigma=sigma_velo, observed=data["velo"])
        pm.Normal("spin_obs", mu=mu_spin, sigma=sigma_spin, observed=data["spin"])

        age_mean_val = float(data["df"]["age"].mean())
        pm.Deterministic("peak_age_velo", age_mean_val + (-b1_velo / (2 * b2_velo)))
        pm.Deterministic("peak_age_spin", age_mean_val + (-b1_spin / (2 * b2_spin)))

    return model


if __name__ == "__main__":
    log = setup_logger("bivariate", MASTER_DATA_DIR / "bivariate.log")

    df = load_data()
    age_mean = get_age_mean(df)

    if "age_c" not in df.columns:
        df["age_c"] = df["age"] - age_mean
        df["age_c_sq"] = df["age_c"] ** 2

    results = {}
    sampler_kwargs = get_pymc_sampler_kwargs(CHAINS)
    log.info(f"Sampler kwargs: {sampler_kwargs}")

    for pt in PITCH_TYPES:
        log.info(f"\n{'=' * 50}")
        log.info(f"Bivariate model: {pt}")

        data = prepare_bivariate_data(df, pt)
        log.info(f"  {data['n_obs']} obs, {data['n_pitchers']} pitchers")

        model = build_bivariate_model(data)

        with model:
            idata = pm.sample(
                draws=DRAWS_BIVARIATE,
                tune=TUNE_BIVARIATE,
                target_accept=TARGET_ACCEPT_BIVARIATE,
                chains=CHAINS,
                progressbar=True,
                nuts_sampler_kwargs=NUTS_KWARGS_BIVARIATE,
                **sampler_kwargs,
            )

        out_path = MASTER_DATA_DIR / f"bivariate_{pt}.nc"
        idata.to_netcdf(str(out_path))
        log.info(f"  Saved: {out_path}")
        results[pt] = idata

        summary = az.summary(
            idata,
            var_names=[
                "b1_velo",
                "b2_velo",
                "b1_spin",
                "b2_spin",
                "peak_age_velo",
                "peak_age_spin",
            ],
        )
        log.info("\n" + summary.to_string())

        rho = idata.posterior["chol_corr"].values[:, :, 0, 1].flatten()
        log.info(
            f"\n  Velo/spin random effect correlation:\n"
            f"  mean={rho.mean():.3f}  "
            f"95% HDI=[{np.percentile(rho, 2.5):.3f}, {np.percentile(rho, 97.5):.3f}]"
        )

    converged = {pt: idata for pt, idata in results.items()}
    if converged:
        try:
            results_df = pd.read_csv(MASTER_DATA_DIR / "model_results.csv")
            PLOTS_DIR.mkdir(parents=True, exist_ok=True)
            plot_bivariate_correlation(converged, PLOTS_DIR)
            plot_bivariate_peak_comparison(converged, results_df, PLOTS_DIR)
            log.info(f"Saved plots to {PLOTS_DIR}")
        except Exception as e:
            log.error(f"Plot generation failed: {e}")
