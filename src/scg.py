"""Stuff Compensation Gap (SCG) = spin peak age − velocity peak age."""

from pathlib import Path

import pandas as pd

from utils import setup_logger
from utils.plots import plot_scg_bars, plot_scg_dumbbell

BASE_DIR = Path(__file__).resolve().parent.parent
MASTER_DATA_DIR = BASE_DIR / "master_data"
PLOTS_DIR = BASE_DIR / "plots"


def compute_scg(results_df: pd.DataFrame, experiment: str = "base") -> pd.DataFrame:
    exp_df = results_df[results_df["experiment"] == experiment]

    velo = (
        exp_df[exp_df["outcome"] == "mean_velo"][
            ["pitch_type", "peak_age_mean", "peak_age_hdi_lo", "peak_age_hdi_hi"]
        ]
        .rename(columns={
            "peak_age_mean": "velo_peak_age",
            "peak_age_hdi_lo": "velo_hdi_lo",
            "peak_age_hdi_hi": "velo_hdi_hi",
        })
    )

    spin = (
        exp_df[exp_df["outcome"] == "mean_spin_rate"][
            ["pitch_type", "peak_age_mean", "peak_age_hdi_lo", "peak_age_hdi_hi"]
        ]
        .rename(columns={
            "peak_age_mean": "spin_peak_age",
            "peak_age_hdi_lo": "spin_hdi_lo",
            "peak_age_hdi_hi": "spin_hdi_hi",
        })
    )

    scg_df = velo.merge(spin, on="pitch_type", how="inner")
    scg_df = scg_df.dropna(subset=["velo_peak_age", "spin_peak_age"])
    scg_df["scg"] = scg_df["spin_peak_age"] - scg_df["velo_peak_age"]
    scg_df["experiment"] = experiment

    return scg_df.sort_values("scg", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    log = setup_logger("scg", MASTER_DATA_DIR / "scg.log")

    results_df = pd.read_csv(MASTER_DATA_DIR / "model_results.csv")
    scg_df = compute_scg(results_df, experiment="base")

    out_path = MASTER_DATA_DIR / "scg_results.csv"
    scg_df.to_csv(out_path, index=False)
    log.info(f"Saved SCG results: {out_path}")
    log.info(
        "\nSCG Summary:\n"
        + scg_df[["pitch_type", "velo_peak_age", "spin_peak_age", "scg"]].to_string(index=False)
    )

    try:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plot_scg_dumbbell(scg_df, PLOTS_DIR)
        plot_scg_bars(scg_df, PLOTS_DIR)
        log.info(f"Saved plots to {PLOTS_DIR}")
    except Exception as e:
        log.error(f"Plot generation failed: {e}")
