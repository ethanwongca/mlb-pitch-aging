"""
Univariate Bayesian mixed effects models for MLB pitch aging study.

Replaces statsmodels REML with full Bayesian inference via bambi/PyMC.

Model:
    y_it = β₀ + β₁·age_c + β₂·age_c² + u_i + v_t + ε_it
    u_i ~ N(0, σ²_u)   (pitcher random intercept)
    v_t ~ N(0, σ²_v)   (year random intercept — partial pooling over seasons)
    ε_it ~ StudentT(ν, 0, σ)

Priors (weakly informative, scaled to outcome):
    β₀         ~ N(ȳ, 2·SD(y))
    β₁, β₂     ~ N(0, SD(y))
    σ_u        ~ HalfNormal(SD(y))
    σ_v        ~ HalfNormal(SD(y))
    σ          ~ HalfNormal(SD(y))
    ν          ~ Gamma(2, 0.1)

Two-pass screening:
    Screen pass (500 draws + 500 tune): fit fast, check significance.
    Full pass (2000 draws + 4000 tune): only if screen shows any HDI excludes zero.

Outputs:
    master_data/model_results.csv        ← posterior summaries
    master_data/fitted_idatas/           ← ArviZ InferenceData per model (.nc)
    master_data/fitted_idatas.pkl        ← dict of all InferenceData objects
    master_data/model_results.log
"""

import pickle
import time
from pathlib import Path

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

from utils import (
    ensure_mean_pfx_x_norm,
    filter_pitchers_by_min_distinct_seasons,
    get_bambi_sampler_kwargs,
    get_age_mean,
    get_data_pitch_type_dict,
    get_default_outcomes,
    get_valid_pitch_types,
    load_data,
    setup_logger,
)
from utils.plots import (
    plot_aging_curves_grid,
    plot_delta_method_comparison,
    plot_ext_loo_heatmap,
    plot_pareto_k_heatmap,
    plot_spaghetti,
    plot_spin_velo_divergence,
    plot_survivorship_bias,
)

BASE_DIR = Path(__file__).resolve().parent.parent
MASTER_DIR = BASE_DIR / "master_data"
IDATAS_DIR = MASTER_DIR / "fitted_idatas"
PLOTS_DIR = BASE_DIR / "plots"
OUTCOMES = get_default_outcomes()
EXPERIMENTS = ["base", "with_ext"]

DRAWS_SCREEN = 500
TUNE_SCREEN = 500
DRAWS_FULL = 2000
TUNE_FULL = 4000
CHAINS = 4
TARGET_ACCEPT_SCREEN = 0.9
TARGET_ACCEPT_FULL = 0.9
NUTS_KWARGS = {"max_treedepth": 14}
MIN_SEASONS_PER_PITCHER = 3

SAMPLER_KWARGS = get_bambi_sampler_kwargs(CHAINS)


# ── Model specification ───────────────────────────────────────────────────────


def build_bambi_formula(outcome: str, experiment: str, quadratic: bool = True) -> str:
    """Build bambi lme4-style formula."""
    fixed = f"{outcome} ~ age_c"
    if quadratic:
        fixed = f"{outcome} ~ age_c + age_c_sq"
    if experiment == "with_ext":
        fixed += " + mean_ext_c"
    return fixed + " + (1|pitcher) + (1|year)"


def build_bambi_priors(
    outcome: str, data: pd.DataFrame, quadratic: bool = True
) -> dict:
    y_sd = float(data[outcome].std())
    priors = {
        "Intercept": bmb.Prior(
            "Normal", mu=float(data[outcome].mean()), sigma=2 * y_sd
        ),
        "age_c": bmb.Prior("Normal", mu=0, sigma=y_sd / 4),
        "1|pitcher": bmb.Prior(
            "Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=y_sd)
        ),
        "sigma": bmb.Prior("HalfNormal", sigma=y_sd),
        "1|year": bmb.Prior(
            "Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=y_sd)
        ),
        "nu": bmb.Prior("Gamma", alpha=2, beta=0.1),
    }
    if quadratic:
        priors["age_c_sq"] = bmb.Prior("Normal", mu=0, sigma=y_sd / 2)
    return priors


def fit_bambi_model(
    data: pd.DataFrame,
    outcome: str,
    experiment: str,
    quadratic: bool = True,
    full_inference: bool = True,
) -> az.InferenceData:
    draws = DRAWS_FULL if full_inference else DRAWS_SCREEN
    tune = TUNE_FULL if full_inference else TUNE_SCREEN
    target_accept = TARGET_ACCEPT_FULL if full_inference else TARGET_ACCEPT_SCREEN
    formula = build_bambi_formula(outcome, experiment, quadratic=quadratic)
    priors = build_bambi_priors(outcome, data, quadratic=quadratic)
    model = bmb.Model(formula, data, family="t", priors=priors)
    fit_kwargs = dict(
        draws=draws,
        tune=tune,
        chains=CHAINS,
        target_accept=target_accept,
        progressbar=False,
        idata_kwargs={"log_likelihood": True},
        nuts_sampler_kwargs=NUTS_KWARGS,
        **SAMPLER_KWARGS,
    )
    return model.fit(**fit_kwargs)


def _screen_is_significant(idata: az.InferenceData) -> bool:
    """Return True if any age coefficient's 95% HDI excludes zero."""
    for name in ["age_c", "age_c_sq"]:
        try:
            samples = idata.posterior[name].values.flatten()
            hdi = az.hdi(samples, hdi_prob=0.95)
            if not (hdi[0] < 0 < hdi[1]):
                return True
        except KeyError:
            pass
    return False


def _log_rhat_diagnostics(idata: az.InferenceData, log, label: str = "") -> None:
    """Log any parameters with R-hat > 1.01 to help diagnose convergence failures."""
    try:
        check_vars = [v for v in ["age_c", "age_c_sq", "1|pitcher_sigma", "1|year_sigma"] if v in idata.posterior]
        summary = az.summary(idata, var_names=check_vars)
        bad = summary[summary["r_hat"] > 1.01][["mean", "sd", "r_hat", "ess_bulk"]]
        if bad.empty:
            log.info(f"  Rhat diagnostic{' ' + label if label else ''}: all params OK (≤1.01)")
        else:
            log.warning(f"  Rhat diagnostic{' ' + label if label else ''}: {len(bad)} params with Rhat>1.01\n{bad.to_string()}")
    except Exception as e:
        log.warning(f"  Rhat diagnostic failed: {e}")


def fit_with_fallback(
    model_df: pd.DataFrame,
    outcome: str,
    experiment: str,
    log,
) -> tuple[az.InferenceData, bool] | None:
    """Two-pass: screen (500 draws) → full (2000 draws) only if significant.
    Returns None if screen finds no significant age effect.
    Returns (idata, is_linear) otherwise.
    """
    idata_screen = fit_bambi_model(
        model_df, outcome, experiment, quadratic=True, full_inference=False
    )

    if not _screen_is_significant(idata_screen):
        log.info("  Screen: not significant — skipping full fit")
        return None

    idata_quad = fit_bambi_model(
        model_df, outcome, experiment, quadratic=True, full_inference=True
    )
    _log_rhat_diagnostics(idata_quad, log, label="(quadratic)")

    b2_samples = idata_quad.posterior["age_c_sq"].values.flatten()
    hdi_b2 = az.hdi(b2_samples, hdi_prob=0.95)
    b2_spans_zero = hdi_b2[0] < 0 < hdi_b2[1]

    if b2_spans_zero:
        log.info(
            f"  age_c_sq HDI spans zero [{hdi_b2[0]:.4f}, {hdi_b2[1]:.4f}] — trying linear"
        )
        idata_lin = fit_bambi_model(
            model_df, outcome, experiment, quadratic=False, full_inference=True
        )
        _log_rhat_diagnostics(idata_lin, log, label="(linear fallback)")

        loo_quad = az.loo(idata_quad).elpd_loo
        loo_lin = az.loo(idata_lin).elpd_loo

        if loo_lin > loo_quad:
            log.info(
                f"  Linear preferred by LOO (lin={loo_lin:.1f} vs quad={loo_quad:.1f})"
            )
            return idata_lin, True

    return idata_quad, False


# ── Result extraction ─────────────────────────────────────────────────────────


def extract_posterior_summaries(
    idata: az.InferenceData,
    pitch_type: str,
    outcome: str,
    experiment: str,
    age_mean: float,
    is_linear: bool,
) -> dict:
    """Extract posterior means, SDs, and 95% HDIs for key parameters."""
    posterior = idata.posterior

    def get_param(name: str) -> np.ndarray | None:
        try:
            return posterior[name].values.flatten()
        except KeyError:
            return None

    def summarize(samples: np.ndarray | None) -> tuple:
        if samples is None or len(samples) == 0:
            return None, None, None, None
        hdi = az.hdi(samples, hdi_prob=0.95)
        return (
            float(np.mean(samples)),
            float(np.std(samples)),
            float(hdi[0]),
            float(hdi[1]),
        )

    b1_samples = get_param("age_c")
    b2_samples = get_param("age_c_sq") if not is_linear else None
    intercept_samples = get_param("Intercept")

    b1_mean, b1_sd, b1_lo, b1_hi = summarize(b1_samples)
    b2_mean, b2_sd, b2_lo, b2_hi = summarize(b2_samples)
    intercept_mean = (
        float(np.mean(intercept_samples)) if intercept_samples is not None else 0.0
    )

    # Peak age posterior (only when majority of b2 samples are negative)
    peak_age_median = peak_age_lo = peak_age_hi = None
    pct_b2_negative = None
    if b2_samples is not None and b1_samples is not None:
        neg_mask = b2_samples < 0
        pct_b2_negative = round(float(neg_mask.mean()), 4)
        if pct_b2_negative >= 0.5:
            peak_samples_raw = age_mean + (
                -b1_samples[neg_mask] / (2 * b2_samples[neg_mask])
            )
            peak_samples = peak_samples_raw[(peak_samples_raw > 15) & (peak_samples_raw < 50)]
            if len(peak_samples) >= 100:
                hdi_pa = az.hdi(peak_samples, hdi_prob=0.95)
                peak_age_median = round(float(np.median(peak_samples)), 3)
                peak_age_lo = round(float(hdi_pa[0]), 3)
                peak_age_hi = round(float(hdi_pa[1]), 3)

    # HDI-based significance: excludes zero
    def hdi_excludes_zero(lo, hi) -> bool:
        if lo is None or hi is None:
            return False
        return not (lo < 0 < hi)

    b1_sig = hdi_excludes_zero(b1_lo, b1_hi)
    b2_sig = hdi_excludes_zero(b2_lo, b2_hi)

    try:
        sigma_u_mean = float(np.mean(get_param("1|pitcher_sigma")))
    except (TypeError, AttributeError):
        sigma_u_mean = float("nan")

    loo = float("nan")
    n_high_pareto_k = None
    pct_high_pareto_k = None
    try:
        loo_result = az.loo(idata)
        loo = float(loo_result.elpd_loo)
        k_vals = loo_result.pareto_k.values
        n_high_pareto_k = int((k_vals > 0.7).sum())
        pct_high_pareto_k = round(float(n_high_pareto_k / len(k_vals)), 4)
    except Exception:
        pass

    try:
        n_obs = int(idata.observed_data[outcome].size)
    except Exception:
        n_obs = None

    try:
        n_groups = int(len(idata.posterior.coords["1|pitcher__factor_dim"].values))
    except Exception:
        n_groups = None

    return {
        "experiment": experiment,
        "pitch_type": pitch_type,
        "outcome": outcome,
        "intercept": intercept_mean,
        "b1_mean": b1_mean,
        "b1_sd": b1_sd,
        "b1_hdi_lo": b1_lo,
        "b1_hdi_hi": b1_hi,
        "b2_mean": b2_mean,
        "b2_sd": b2_sd,
        "b2_hdi_lo": b2_lo,
        "b2_hdi_hi": b2_hi,
        "b1_significant": b1_sig,
        "b2_significant": b2_sig,
        "significant": b1_sig or b2_sig,
        "pct_b2_negative": pct_b2_negative,
        "peak_age_median": peak_age_median,
        "peak_age_hdi_lo": peak_age_lo,
        "peak_age_hdi_hi": peak_age_hi,
        "decline_at_mean": b1_mean,
        "sigma_u_mean": sigma_u_mean,
        "loo": loo,
        "n_high_pareto_k": n_high_pareto_k,
        "pct_high_pareto_k": pct_high_pareto_k,
        "is_linear_model": is_linear,
        "n_obs": n_obs,
        "n_groups": n_groups,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    start = time.time()
    log = setup_logger("models", MASTER_DIR / "model_results.log")

    IDATAS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    age_mean = get_age_mean(df)
    log.info(f"Age mean: {age_mean:.2f}")
    log.info(f"Sampler kwargs: {SAMPLER_KWARGS}")

    df = ensure_mean_pfx_x_norm(df)

    pitch_type_dict = get_data_pitch_type_dict(df, get_valid_pitch_types())
    log.info(f"Loaded {len(df)} rows | {len(pitch_type_dict)} pitch types")
    log.info(f"Minimum distinct seasons per pitcher: {MIN_SEASONS_PER_PITCHER}")
    log.info(f"Outcomes: {OUTCOMES}")
    log.info(f"Experiments: {EXPERIMENTS}\n")

    all_results = []
    fitted_idatas = {}
    total = len(EXPERIMENTS) * len(pitch_type_dict) * len(OUTCOMES)
    count = 0

    for experiment in EXPERIMENTS:
        log.info(f"{'='*50}  {experiment}")

        for pitch_type, pt_df in pitch_type_dict.items():
            for outcome in OUTCOMES:
                count += 1
                required = [outcome, "age_c", "age_c_sq", "year", "pitcher", "mean_ext_c"]
                model_df = pt_df.dropna(subset=required).copy()
                model_df = filter_pitchers_by_min_distinct_seasons(
                    model_df, MIN_SEASONS_PER_PITCHER
                )

                if len(model_df) < 50:
                    log.warning(
                        f"[{count}/{total}] SKIP {experiment} "
                        f"{pitch_type} {outcome} n={len(model_df)} "
                        f"(after min-seasons filter)"
                    )
                    continue

                log.info(
                    f"[{count}/{total}] {experiment} | "
                    f"{pitch_type} | {outcome} | n={len(model_df)}"
                )

                try:
                    result = fit_with_fallback(model_df, outcome, experiment, log)

                    if result is None:
                        log.info(
                            f"[{count}/{total}] SKIP (screen: not sig) "
                            f"{experiment} {pitch_type} {outcome}"
                        )
                        continue

                    idata, is_linear = result

                    row = extract_posterior_summaries(
                        idata, pitch_type, outcome, experiment, age_mean, is_linear
                    )
                    all_results.append(row)

                    key = f"{experiment}_{pitch_type}_{outcome}"
                    fitted_idatas[key] = idata
                    idata.to_netcdf(str(IDATAS_DIR / f"{key}.nc"))

                    peak_str = (
                        f"peak={row['peak_age_median']:.1f} "
                        f"[{row['peak_age_hdi_lo']:.1f}, {row['peak_age_hdi_hi']:.1f}]"
                        if row["peak_age_median"]
                        else "monotonic"
                    )
                    n_k = row["n_high_pareto_k"] or 0
                    pct_k = (row["pct_high_pareto_k"] or 0) * 100
                    pareto_str = f"Pareto-k>0.7: {n_k} ({pct_k:.1f}%)"
                    log.info(
                        f"  → {peak_str} | "
                        f"{'sig' if row['significant'] else 'not sig'} | "
                        f"LOO={row['loo']:.1f} | "
                        f"{pareto_str} | "
                        f"linear={'Y' if is_linear else 'N'}"
                    )
                    if n_k > 0:
                        log.warning(f"  !! {pareto_str} — {experiment} {pitch_type} {outcome}")

                except Exception as e:
                    log.error(
                        f"[{count}/{total}] FAILED "
                        f"{experiment} {pitch_type} {outcome}: {e}"
                    )

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(MASTER_DIR / "model_results.csv", index=False)
    log.info(f"\nSaved {len(results_df)} models → model_results.csv")

    with open(MASTER_DIR / "fitted_idatas.pkl", "wb") as f:
        pickle.dump(fitted_idatas, f)
    log.info("Saved fitted_idatas.pkl")

    del fitted_idatas  # free RAM before plotting

    # Summaries
    if not results_df.empty:
        for experiment in EXPERIMENTS:
            exp_df = results_df[results_df["experiment"] == experiment]
            sig_df = exp_df[exp_df["significant"]]
            not_sig = exp_df[~exp_df["significant"]]
            log.info(f"\n{'='*50}")
            log.info(f"Summary: {experiment} — {len(sig_df)}/{len(exp_df)} significant")
            if not sig_df.empty:
                log.info(
                    "\n"
                    + sig_df[
                        [
                            "pitch_type",
                            "outcome",
                            "peak_age_median",
                            "b1_mean",
                            "b2_mean",
                            "significant",
                            "is_linear_model",
                            "loo",
                        ]
                    ].to_string(index=False)
                )
            if not not_sig.empty:
                log.info(
                    f"Not significant: {list(zip(not_sig['pitch_type'], not_sig['outcome']))}"
                )

        # PSIS-LOO comparison
        loo_compare = results_df.pivot_table(
            index=["pitch_type", "outcome"], columns="experiment", values="loo"
        )
        if {"base", "with_ext"}.issubset(loo_compare.columns):
            loo_compare["delta"] = loo_compare["with_ext"] - loo_compare["base"]
            meaningful = loo_compare[loo_compare["delta"].abs() > 2][["delta"]]
            log.info("\nLOO delta >2 (positive = ext improves):\n" + meaningful.to_string())

        # Pareto k > 0.7 summary across all models
        k_df = results_df[results_df["n_high_pareto_k"].notna()].copy()
        k_df["n_high_pareto_k"] = k_df["n_high_pareto_k"].astype(int)
        flagged = k_df[k_df["n_high_pareto_k"] > 0].sort_values("n_high_pareto_k", ascending=False)
        total_high_k = int(k_df["n_high_pareto_k"].sum())
        log.info(f"\n{'='*50}")
        log.info(f"Pareto k > 0.7 summary: {total_high_k} total bad obs across {len(flagged)} models")
        if not flagged.empty:
            log.info(
                "\n"
                + flagged[
                    ["experiment", "pitch_type", "outcome", "n_high_pareto_k", "pct_high_pareto_k", "n_obs"]
                ].to_string(index=False)
            )

        # Plots — reload fitted_idatas from pkl (del'd above to free RAM during sampling)
        try:
            for experiment in EXPERIMENTS:
                plot_aging_curves_grid(
                    results_df, df, age_mean, PLOTS_DIR, experiment=experiment
                )
            plot_spin_velo_divergence(
                results_df,
                df,
                age_mean,
                PLOTS_DIR,
                pitch_types=["FF", "SL", "SI", "CH", "CU"],
                experiment="with_ext",
            )
            for pt, outcome in [
                ("FF", "mean_velo"),
                ("FF", "mean_spin_rate"),
                ("SI", "mean_velo"),
                ("SL", "mean_spin_rate"),
            ]:
                plot_survivorship_bias(
                    results_df,
                    df,
                    age_mean,
                    PLOTS_DIR,
                    pitch_type=pt,
                    outcome=outcome,
                    experiment="base",
                )
            plot_delta_method_comparison(
                results_df, df, age_mean, PLOTS_DIR, experiment="base"
            )
            plot_ext_loo_heatmap(results_df, PLOTS_DIR)
            for experiment in EXPERIMENTS:
                plot_pareto_k_heatmap(results_df, PLOTS_DIR, experiment=experiment)
            plot_spaghetti(
                results_df,
                df,
                age_mean,
                PLOTS_DIR,
                experiment="with_ext",
                min_seasons=5,
                n_sample=40,
                seed=42,
            )
            log.info(f"Saved plots → {PLOTS_DIR}")
        except Exception as e:
            log.error(f"Plot generation failed: {e}")
    else:
        log.warning("No models completed successfully; skipping summaries and plots.")

    log.info(f"Total runtime: {(time.time() - start) / 60:.1f} minutes")
