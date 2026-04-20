"""Stuff Durability Index (SDI) — Bayesian population-level spin/velo slope ratio."""

import pickle
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from utils import (
    ensure_mean_pfx_x_norm,
    get_age_mean,
    get_valid_pitch_types,
    load_data,
    setup_logger,
)
from utils.plots import plot_sdi_distributions

BASE_DIR = Path(__file__).resolve().parent.parent
MASTER_DATA_DIR = BASE_DIR / "master_data"
PLOTS_DIR = BASE_DIR / "plots"


def posterior_slope_at_age(
    idata: az.InferenceData,
    age_mean: float,
    eval_age: float,
) -> np.ndarray:
    """Posterior samples of dy/d(age) at eval_age from a fitted bambi model."""
    b1 = idata.posterior["age_c"].values.flatten()
    try:
        b2 = idata.posterior["age_c_sq"].values.flatten()
        return b1 + 2 * b2 * (eval_age - age_mean)
    except KeyError:
        return b1


def compute_sdi_from_posteriors(
    fitted_idatas: dict,
    raw_df: pd.DataFrame,
    age_mean: float,
    pitch_type: str,
    experiment: str = "base",
    eval_age: float = 28.0,
) -> dict | None:
    """Population-level SDI = (spin slope / spin SD) / (velo slope / velo SD).

    Returns None if either model is missing from fitted_idatas.
    """
    key_velo = f"{experiment}_{pitch_type}_mean_velo"
    key_spin = f"{experiment}_{pitch_type}_mean_spin_rate"

    if key_velo not in fitted_idatas or key_spin not in fitted_idatas:
        return None

    idata_velo = fitted_idatas[key_velo]
    idata_spin = fitted_idatas[key_spin]

    slope_velo = posterior_slope_at_age(idata_velo, age_mean, eval_age)
    slope_spin = posterior_slope_at_age(idata_spin, age_mean, eval_age)

    pt_df = raw_df[raw_df["pitch_type"] == pitch_type]
    sd_velo = float(pt_df["mean_velo"].std())
    sd_spin = float(pt_df["mean_spin_rate"].std())

    slope_velo_norm = slope_velo / sd_velo
    slope_spin_norm = slope_spin / sd_spin

    # SDI per posterior sample (both declines are negative → ratio is positive)
    with np.errstate(invalid="ignore", divide="ignore"):
        sdi_samples = np.where(
            np.abs(slope_velo_norm) > 1e-6,
            slope_spin_norm / slope_velo_norm,
            np.nan,
        )
    sdi_samples = sdi_samples[np.isfinite(sdi_samples)]

    if len(sdi_samples) < 100:
        return None

    hdi = az.hdi(sdi_samples, hdi_prob=0.95)
    p_sdi_gt1 = float(np.mean(sdi_samples > 1.0))

    return {
        "pitch_type": pitch_type,
        "experiment": experiment,
        "eval_age": eval_age,
        "sdi_mean": round(float(np.mean(sdi_samples)), 4),
        "sdi_sd": round(float(np.std(sdi_samples)), 4),
        "sdi_hdi_lo": round(float(hdi[0]), 4),
        "sdi_hdi_hi": round(float(hdi[1]), 4),
        "p_sdi_gt1": round(p_sdi_gt1, 4),
        "slope_velo_mean": round(float(np.mean(slope_velo)), 4),
        "slope_spin_mean": round(float(np.mean(slope_spin)), 4),
        "n_posterior": len(sdi_samples),
    }


def classify_sdi(sdi: float) -> str:
    if pd.isna(sdi):
        return "unknown"
    if sdi > 1.2:
        return "spin_first"
    if sdi < 0.8:
        return "velo_first"
    return "balanced"


if __name__ == "__main__":
    log = setup_logger("sdi", MASTER_DATA_DIR / "sdi.log")

    df = load_data()
    age_mean = get_age_mean(df)

    df = ensure_mean_pfx_x_norm(df)

    models_pkl = MASTER_DATA_DIR / "fitted_idatas.pkl"
    if not models_pkl.exists():
        raise FileNotFoundError(f"Missing {models_pkl}. Run src/models.py first.")

    with open(models_pkl, "rb") as fh:
        fitted_idatas = pickle.load(fh)

    log.info(f"Loaded {len(fitted_idatas)} fitted models | age_mean={age_mean:.2f}")

    all_sdi = []

    for pt in get_valid_pitch_types():
        log.info(f"Computing SDI for {pt}...")
        result = compute_sdi_from_posteriors(
            fitted_idatas, df, age_mean, pt, experiment="base", eval_age=28.0
        )
        if result is None:
            log.warning(f"  {pt}: missing velo or spin model — skipped")
            continue

        result["sdi_class"] = classify_sdi(result["sdi_mean"])
        all_sdi.append(result)

        log.info(
            f"  SDI={result['sdi_mean']:.3f} "
            f"[{result['sdi_hdi_lo']:.3f}, {result['sdi_hdi_hi']:.3f}] "
            f"p(SDI>1)={result['p_sdi_gt1']:.3f} | {result['sdi_class']}"
        )

    sdi_df = pd.DataFrame(all_sdi)
    out_path = MASTER_DATA_DIR / "sdi_results.csv"
    sdi_df.to_csv(out_path, index=False)
    log.info(f"Saved SDI results: {out_path}")

    try:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plot_sdi_distributions(sdi_df, PLOTS_DIR)
        log.info(f"Saved plots to {PLOTS_DIR}")
    except Exception as e:
        log.error(f"Plot generation failed: {e}")

    log.info("\nSDI Summary:\n" + sdi_df.to_string(index=False))
