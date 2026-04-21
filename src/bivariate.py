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
        .dropna(subset=[outcome_velo, outcome_spin, "age_c", "age_c_sq", "mean_ext_c"])
        .copy()
    )

    pt_df = filter_pitchers_by_min_distinct_seasons(pt_df, MIN_SEASONS)

    pitchers = pt_df["pitcher"].unique()
    pitcher_idx = {pid: i for i, pid in enumerate(pitchers)}
    pt_df["pitcher_idx"] = pt_df["pitcher"].map(pitcher_idx)

    years = pt_df["year"].unique()
    year_idx = {y: i for i, y in enumerate(years)}
    pt_df["year_idx"] = pt_df["year"].map(year_idx)

    return {
        "df": pt_df,
        "pitchers": pitchers,
        "pitcher_idx": pt_df["pitcher_idx"].values,
        "years": years,
        "year_idx": pt_df["year_idx"].values,
        "age_c": pt_df["age_c"].values,
        "age_c_sq": pt_df["age_c_sq"].values,
        "mean_ext_c": pt_df["mean_ext_c"].values,
        "velo": pt_df[outcome_velo].values,
        "spin": pt_df[outcome_spin].values,
        "n_pitchers": len(pitchers),
        "n_years": len(years),
        "n_obs": len(pt_df),
    }


def build_bivariate_model(data: dict, age_mean: float, experiment: str = "base") -> pm.Model:
    velo_sd = float(data["velo"].std())
    spin_sd = float(data["spin"].std())

    with pm.Model() as model:
        b0_velo = pm.Normal("b0_velo", mu=data["velo"].mean(), sigma=2 * velo_sd)
        b1_velo = pm.Normal("b1_velo", mu=0, sigma=velo_sd / 4)
        b2_velo = pm.Normal("b2_velo", mu=0, sigma=velo_sd / 2)

        b0_spin = pm.Normal("b0_spin", mu=data["spin"].mean(), sigma=2 * spin_sd)
        b1_spin = pm.Normal("b1_spin", mu=0, sigma=spin_sd / 4)
        b2_spin = pm.Normal("b2_spin", mu=0, sigma=spin_sd / 2)

        # Pitcher random effect (correlated across velo and spin)
        sd_dist = pm.HalfNormal.dist(sigma=np.array([velo_sd, spin_sd]))
        chol, _, _ = pm.LKJCholeskyCov(
            "chol", n=2, eta=2, sd_dist=sd_dist, compute_corr=True
        )
        # Non-centered parameterization: avoids funnel geometry in large hierarchies
        z = pm.Normal("z", mu=0, sigma=1, shape=(data["n_pitchers"], 2))
        u = pm.Deterministic("u", pm.math.dot(z, chol.T))

        # Year random effect (partial pooling over seasons)
        sd_year = pm.HalfNormal.dist(sigma=np.array([velo_sd, spin_sd]))
        chol_year, _, _ = pm.LKJCholeskyCov(
            "chol_year", n=2, eta=2, sd_dist=sd_year, compute_corr=True
        )
        z_year = pm.Normal("z_year", mu=0, sigma=1, shape=(data["n_years"], 2))
        u_year = pm.Deterministic("u_year", pm.math.dot(z_year, chol_year.T))

        age_c = data["age_c"]
        age_c_sq = data["age_c_sq"]
        ext_c = data["mean_ext_c"]
        pidx = data["pitcher_idx"]
        yidx = data["year_idx"]

        mu_velo = b0_velo + u[pidx, 0] + u_year[yidx, 0] + b1_velo * age_c + b2_velo * age_c_sq
        mu_spin = b0_spin + u[pidx, 1] + u_year[yidx, 1] + b1_spin * age_c + b2_spin * age_c_sq

        if experiment == "with_ext":
            b_ext_velo = pm.Normal("b_ext_velo", mu=0, sigma=velo_sd / 2)
            b_ext_spin = pm.Normal("b_ext_spin", mu=0, sigma=spin_sd / 2)
            mu_velo = mu_velo + b_ext_velo * ext_c
            mu_spin = mu_spin + b_ext_spin * ext_c

        sigma_velo = pm.HalfNormal("sigma_velo", sigma=velo_sd)
        sigma_spin = pm.HalfNormal("sigma_spin", sigma=spin_sd)

        pm.Normal("velo_obs", mu=mu_velo, sigma=sigma_velo, observed=data["velo"])
        pm.Normal("spin_obs", mu=mu_spin, sigma=sigma_spin, observed=data["spin"])

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

    EXPERIMENTS = ["base", "with_ext"]

    for experiment in EXPERIMENTS:
        for pt in PITCH_TYPES:
            log.info(f"\n{'=' * 50}")
            log.info(f"Bivariate model: {pt} [{experiment}]")

            data = prepare_bivariate_data(df, pt)
            log.info(f"  {data['n_obs']} obs, {data['n_pitchers']} pitchers")

            model = build_bivariate_model(data, age_mean, experiment=experiment)

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

            out_path = MASTER_DATA_DIR / f"bivariate_{experiment}_{pt}.nc"
            idata.to_netcdf(str(out_path))
            log.info(f"  Saved: {out_path}")
            
            key = f"{experiment}_{pt}"
            results[key] = idata

            var_names = [
                "b1_velo",
                "b2_velo",
                "b1_spin",
                "b2_spin",
            ]
            if experiment == "with_ext":
                var_names += ["b_ext_velo", "b_ext_spin"]
                
            summary = az.summary(idata, var_names=var_names)
            log.info("\n" + summary.to_string())

            rho = idata.posterior["chol_corr"].values[:, :, 0, 1].flatten()
            log.info(
                f"\n  Velo/spin random effect correlation:\n"
                f"  mean={rho.mean():.3f}  "
                f"95% HDI=[{np.percentile(rho, 2.5):.3f}, {np.percentile(rho, 97.5):.3f}]"
            )

    # Use 'base' runs for correlation summary plots
    converged = {pt: idata for key, idata in results.items() if key.startswith("base_") for pt in [key.split("_")[1]]}
    if converged:
        try:
            results_df = pd.read_csv(MASTER_DATA_DIR / "model_results.csv")
            PLOTS_DIR.mkdir(parents=True, exist_ok=True)
            plot_bivariate_correlation(converged, PLOTS_DIR)
            plot_bivariate_peak_comparison(converged, results_df, PLOTS_DIR)
            log.info(f"Saved plots to {PLOTS_DIR}")
        except Exception as e:
            log.error(f"Plot generation failed: {e}")
