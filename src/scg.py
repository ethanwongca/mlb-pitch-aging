"""Stuff Compensation Gap (SCG) = spin peak age − velocity peak age."""

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from utils import setup_logger
from utils.plots import plot_scg_bars, plot_scg_comparison, plot_scg_dumbbell

BASE_DIR = Path(__file__).resolve().parent.parent
MASTER_DATA_DIR = BASE_DIR / "master_data"
PLOTS_DIR = BASE_DIR / "plots"


def compute_scg(results_df: pd.DataFrame, experiment: str = "base") -> pd.DataFrame:
    exp_df = results_df[results_df["experiment"] == experiment]

    velo = (
        exp_df[exp_df["outcome"] == "mean_velo"][
            ["pitch_type", "peak_age_median", "peak_age_hdi_lo", "peak_age_hdi_hi"]
        ]
        .rename(columns={
            "peak_age_median": "velo_peak_age",
            "peak_age_hdi_lo": "velo_hdi_lo",
            "peak_age_hdi_hi": "velo_hdi_hi",
        })
    )

    spin = (
        exp_df[exp_df["outcome"] == "mean_spin_rate"][
            ["pitch_type", "peak_age_median", "peak_age_hdi_lo", "peak_age_hdi_hi"]
        ]
        .rename(columns={
            "peak_age_median": "spin_peak_age",
            "peak_age_hdi_lo": "spin_hdi_lo",
            "peak_age_hdi_hi": "spin_hdi_hi",
        })
    )

    scg_df = velo.merge(spin, on="pitch_type", how="inner")
    scg_df = scg_df.dropna(subset=["velo_peak_age", "spin_peak_age"])
    scg_df["scg"] = scg_df["spin_peak_age"] - scg_df["velo_peak_age"]
    scg_df["experiment"] = experiment

    return scg_df.sort_values("scg", ascending=False).reset_index(drop=True)


def compute_bivariate_scg(pitch_types: list[str], master_data_dir: Path) -> pd.DataFrame:
    rows = []
    for pt in pitch_types:
        nc_path = master_data_dir / f"bivariate_{pt}.nc"
        if not nc_path.exists():
            continue

        idata = az.from_netcdf(str(nc_path))
        post = idata.posterior

        peak_velo = post["peak_age_velo"].values.flatten()
        peak_spin = post["peak_age_spin"].values.flatten()

        # Filter to physically plausible peak ages — mirrors the bounds used in
        # inference.py's peak_age_from_posterior to keep comparisons consistent.
        velo_median = float(np.median(peak_velo))
        spin_median = float(np.median(peak_spin))
        scg_median = float(np.median(peak_spin - peak_velo))
        
        valid = (
            (peak_velo > 15) & (peak_velo < 50)
            & (peak_spin > 15) & (peak_spin < 50)
        )
        peak_velo = peak_velo[valid]
        peak_spin = peak_spin[valid]

        if len(peak_velo) < 100:
            continue

        scg_samples = peak_spin - peak_velo
        hdi = az.hdi(scg_samples, hdi_prob=0.95)

        rows.append({
            "pitch_type": pt,
            "velo_peak_age": velo_median,
            "spin_peak_age": spin_median,
            "scg": scg_median,
            "scg_hdi_lo": float(hdi[0]),
            "scg_hdi_hi": float(hdi[1]),
        })

    return pd.DataFrame(rows).sort_values("scg", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    log = setup_logger("scg", MASTER_DATA_DIR / "scg.log")

    results_df = pd.read_csv(MASTER_DATA_DIR / "model_results.csv")
    scg_df = compute_scg(results_df, experiment="base")

    out_path = MASTER_DATA_DIR / "scg_results.csv"
    scg_df.to_csv(out_path, index=False)
    log.info(f"Saved SCG results: {out_path}")
    log.info(
        "\nUnivariate SCG Summary:\n"
        + scg_df[["pitch_type", "velo_peak_age", "spin_peak_age", "scg"]].to_string(index=False)
    )

    from bivariate import PITCH_TYPES as BIVARIATE_PITCH_TYPES
    biv_scg_df = compute_bivariate_scg(BIVARIATE_PITCH_TYPES, MASTER_DATA_DIR)
    if not biv_scg_df.empty:
        biv_out_path = MASTER_DATA_DIR / "scg_bivariate_results.csv"
        biv_scg_df.to_csv(biv_out_path, index=False)
        log.info(f"Saved bivariate SCG results: {biv_out_path}")
        log.info(
            "\nBivariate SCG Summary:\n"
            + biv_scg_df[["pitch_type", "velo_peak_age", "spin_peak_age", "scg", "scg_hdi_lo", "scg_hdi_hi"]].to_string(index=False)
        )

    try:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plot_scg_dumbbell(scg_df, PLOTS_DIR)
        plot_scg_bars(scg_df, PLOTS_DIR)
        if not biv_scg_df.empty:
            plot_scg_comparison(scg_df, biv_scg_df, PLOTS_DIR)
        log.info(f"Saved plots to {PLOTS_DIR}")
    except Exception as e:
        log.error(f"Plot generation failed: {e}")
