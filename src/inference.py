"""
Bayesian inference — peak age HDIs and decline rate posteriors.

Full posteriors from bambi/PyMC replace the delta-method approximation.
Peak ages and decline rates are computed directly from posterior samples.
"""

import pickle
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from utils import PITCH_ORDER, get_age_mean, load_data, setup_logger
from utils.plots import plot_decline_rate_ci, plot_peak_age_ci

BASE_DIR = Path(__file__).resolve().parent.parent
MASTER_DIR = BASE_DIR / "master_data"
PLOTS_DIR = BASE_DIR / "plots"


def peak_age_from_posterior(
    idata: az.InferenceData,
    age_mean: float,
    hdi_prob: float = 0.95,
) -> dict:
    """Compute peak age directly from posterior samples."""
    posterior = idata.posterior
    try:
        b1 = posterior["age_c"].values.flatten()
        b2 = posterior["age_c_sq"].values.flatten()
    except KeyError:
        return {
            "peak_age": None,
            "hdi_lo": None,
            "hdi_hi": None,
            "pct_b2_negative": None,
        }

    mask = b2 < 0
    pct_neg = float(mask.mean())

    if pct_neg < 0.5:
        return {
            "peak_age": None,
            "hdi_lo": None,
            "hdi_hi": None,
            "pct_b2_negative": pct_neg,
        }

    peak_samples_raw = age_mean + (-b1[mask] / (2 * b2[mask]))
    
    peak_samples = peak_samples_raw[(peak_samples_raw > 15) & (peak_samples_raw < 50)]

    if len(peak_samples) < 100:
        return {
            "peak_age": None,
            "hdi_lo": None,
            "hdi_hi": None,
            "pct_b2_negative": pct_neg,
        }

    peak_median = float(np.median(peak_samples))

    hdi = az.hdi(peak_samples, hdi_prob=hdi_prob)

    return {
        "peak_age": round(peak_median, 3),
        "hdi_lo": round(float(hdi[0]), 3),
        "hdi_hi": round(float(hdi[1]), 3),
        "pct_b2_negative": round(pct_neg, 4),
    }


def decline_rate_from_posterior(
    idata: az.InferenceData,
    age_mean: float,
    eval_age: float = 28,
    hdi_prob: float = 0.95,
) -> dict:
    """Posterior of decline rate at eval_age: dy/d(age) = b1 + 2*b2*(eval_age - age_mean)."""
    posterior = idata.posterior
    age_c_eval = eval_age - age_mean

    b1 = posterior["age_c"].values.flatten()
    try:
        b2 = posterior["age_c_sq"].values.flatten()
        rate = b1 + 2 * b2 * age_c_eval
    except KeyError:
        rate = b1

    hdi = az.hdi(rate, hdi_prob=hdi_prob)

    return {
        "eval_age": eval_age,
        "rate_mean": round(float(np.mean(rate)), 4),
        "rate_sd": round(float(np.std(rate)), 5),
        "hdi_lo": round(float(hdi[0]), 4),
        "hdi_hi": round(float(hdi[1]), 4),
    }


def compute_all_posteriors(
    fitted_idatas: dict,
    results_df: pd.DataFrame,
    age_mean: float,
    experiment: str = "base",
    eval_ages: list[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute peak age and decline rate posteriors for all significant models."""
    if eval_ages is None:
        eval_ages = [24, 28, 32, 36]

    sig_df = results_df[
        (results_df["experiment"] == experiment) & (results_df["significant"] == True)
    ]

    peak_rows = []
    decline_rows = []

    for _, row in sig_df.iterrows():
        pt = row["pitch_type"]
        outcome = row["outcome"]
        key = f"{experiment}_{pt}_{outcome}"

        if key not in fitted_idatas:
            continue

        idata = fitted_idatas[key]

        peak = peak_age_from_posterior(idata, age_mean)
        peak_rows.append(
            {"experiment": experiment, "pitch_type": pt, "outcome": outcome, **peak}
        )

        for age in eval_ages:
            rate = decline_rate_from_posterior(idata, age_mean, eval_age=age)
            decline_rows.append(
                {"experiment": experiment, "pitch_type": pt, "outcome": outcome, **rate}
            )

    peak_df = pd.DataFrame(peak_rows).sort_values(["outcome", "pitch_type"])
    decline_df = pd.DataFrame(decline_rows).sort_values(
        ["outcome", "pitch_type", "eval_age"]
    )

    return peak_df, decline_df


def shrinkage_summary(
    fitted_idatas: dict,
    results_df: pd.DataFrame,
    age_mean: float,
    experiment: str = "base",
    outcome: str = "mean_velo",
) -> pd.DataFrame:
    """Summarize random-effect shrinkage (u_i SD vs prior σ_u) by pitch type."""
    rows = []
    mask = (
        (results_df["experiment"] == experiment)
        & (results_df["outcome"] == outcome)
        & (results_df["significant"] == True)
    )
    for _, model_row in results_df[mask].iterrows():
        pt = model_row["pitch_type"]
        key = f"{experiment}_{pt}_{outcome}"
        if key not in fitted_idatas:
            continue
        idata = fitted_idatas[key]
        try:
            u_samples = idata.posterior[
                "1|pitcher"
            ].values  # (chains, draws, n_pitchers)
            u_mean = u_samples.mean(axis=(0, 1))  # (n_pitchers,)
            u_sd = float(u_samples.std(axis=(0, 1)).mean())
            sigma_u = float(idata.posterior["1|pitcher_sigma"].values.mean())
            rows.append(
                {
                    "pitch_type": pt,
                    "outcome": outcome,
                    "experiment": experiment,
                    "n_pitchers": len(u_mean),
                    "u_mean_abs": round(float(np.abs(u_mean).mean()), 4),
                    "u_sd": round(u_sd, 4),
                    "sigma_u": round(sigma_u, 4),
                    "shrinkage_ratio": round(u_sd / (sigma_u + 1e-9), 4),
                }
            )
        except Exception:
            pass
    return pd.DataFrame(rows)


def _log_peak_table(log, peak_df: pd.DataFrame, outcome: str, experiment: str) -> None:
    out_df = peak_df[peak_df["outcome"] == outcome].copy()
    if out_df.empty:
        log.warning(f"No peak-age rows for {outcome} [{experiment}]")
        return
    out_df["peak_age_ci"] = out_df.apply(
        lambda r: (
            f"{r['peak_age']:.1f} [{r['hdi_lo']:.1f}, {r['hdi_hi']:.1f}]"
            if pd.notna(r["peak_age"])
            else "NO_PEAK"
        ),
        axis=1,
    )
    report = (
        out_df[["pitch_type", "peak_age_ci"]]
        .set_index("pitch_type")
        .reindex(PITCH_ORDER)
        .reset_index()
    )
    log.info(
        f"\nPeak ages ({experiment}, {outcome}):\n" + report.to_string(index=False)
    )


def _log_decline_table(
    log, decline_df: pd.DataFrame, outcome: str, experiment: str
) -> None:
    out_df = decline_df[
        (decline_df["outcome"] == outcome) & (decline_df["eval_age"] == 28)
    ].copy()
    if out_df.empty:
        log.warning(f"No decline rows for age 28, {outcome} [{experiment}]")
        return
    report = (
        out_df[["pitch_type", "rate_mean", "hdi_lo", "hdi_hi"]]
        .set_index("pitch_type")
        .reindex(PITCH_ORDER)
        .reset_index()
    )
    log.info(
        f"\nDecline rates at age 28 ({experiment}, {outcome}):\n"
        + report.to_string(index=False)
    )


if __name__ == "__main__":
    log = setup_logger("inference", MASTER_DIR / "inference.log")

    raw_df = load_data()
    age_mean = get_age_mean(raw_df)
    results_df = pd.read_csv(MASTER_DIR / "model_results.csv")

    models_pkl = MASTER_DIR / "fitted_idatas.pkl"
    if not models_pkl.exists():
        raise FileNotFoundError(f"Missing {models_pkl}. Run src/models.py first.")

    with open(models_pkl, "rb") as f:
        fitted_idatas = pickle.load(f)

    log.info(f"Loaded {len(fitted_idatas)} fitted models | age_mean={age_mean:.2f}")

    for experiment in ["base", "with_ext"]:
        log.info(f"\n{'='*50}  {experiment}")

        peak_df, decline_df = compute_all_posteriors(
            fitted_idatas,
            results_df,
            age_mean,
            experiment=experiment,
        )

        peak_csv = MASTER_DIR / f"peak_age_posteriors_{experiment}.csv"
        decline_csv = MASTER_DIR / f"decline_rate_posteriors_{experiment}.csv"
        peak_df.to_csv(peak_csv, index=False)
        decline_df.to_csv(decline_csv, index=False)
        log.info(f"Saved: {peak_csv.name}, {decline_csv.name}")

        for outcome in ["mean_velo", "mean_spin_rate"]:
            _log_peak_table(log, peak_df, outcome, experiment)
            _log_decline_table(log, decline_df, outcome, experiment)

            out_peak = peak_df[peak_df["outcome"] == outcome].dropna(
                subset=["peak_age", "hdi_lo", "hdi_hi"]
            )
            if not out_peak.empty:
                plot_peak_age_ci(
                    out_peak.rename(columns={"hdi_lo": "ci_lo", "hdi_hi": "ci_hi"}),
                    out_dir=PLOTS_DIR,
                    outcome=outcome,
                    experiment=experiment,
                )

            out_dec = decline_df[decline_df["outcome"] == outcome].dropna(
                subset=["rate_mean", "hdi_lo", "hdi_hi"]
            )
            if not out_dec.empty:
                plot_decline_rate_ci(
                    out_dec.rename(
                        columns={
                            "rate_mean": "rate",
                            "hdi_lo": "ci_lo",
                            "hdi_hi": "ci_hi",
                        }
                    ),
                    out_dir=PLOTS_DIR,
                    outcome=outcome,
                    eval_age=28,
                    experiment=experiment,
                )

        # Shrinkage summary
        for outcome in ["mean_velo", "mean_spin_rate"]:
            shrink_df = shrinkage_summary(
                fitted_idatas,
                results_df,
                age_mean,
                experiment=experiment,
                outcome=outcome,
            )
            if not shrink_df.empty:
                log.info(
                    f"\nShrinkage ({experiment}, {outcome}):\n"
                    + shrink_df.to_string(index=False)
                )

    log.info("\nDone")
