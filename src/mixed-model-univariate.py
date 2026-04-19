"""
Univariate mixed effects models for MLB pitch aging study.

Fits one model per pitch type × outcome combination.

Experiment A (base):     outcome ~ age_c + age_c_sq + C(year) + (1|pitcher)
Experiment B (with_ext): outcome ~ age_c + age_c_sq + C(year) + mean_ext + (1|pitcher)

REML=True  for parameter estimation (unbiased variance components)
REML=False for AIC/BIC model comparison (AIC not valid under REML)

Outputs:
    master_data/model_results.log
"""

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLMResults
from pathlib import Path
import time

from utils import (
    build_univariate_equation,
    build_univariate_equation_with_ext,
    get_age_mean,
    get_data_pitch_type_dict,
    get_default_outcomes,
    get_n_groups,
    get_valid_pitch_types,
    load_data,
    setup_logger,
)

BASE_DIR = Path(__file__).resolve().parent.parent
MASTER_DATA_DIR = BASE_DIR / "master_data"
OUTCOMES = get_default_outcomes()
EXPERIMENTS = {
    "base": build_univariate_equation,
    "with_ext": build_univariate_equation_with_ext,
}

def extract_results(
    result_reml: MixedLMResults,
    result_mle: MixedLMResults,
    pitch_type: str,
    outcome: str,
    experiment: str,
    age_mean: float,
) -> dict:
    b1 = result_reml.fe_params.get("age_c", float("nan"))
    b2 = result_reml.fe_params.get("age_c_sq", float("nan"))
    peak_age = age_mean + (-b1 / (2 * b2)) if pd.notna(b2) and b2 < 0 else None
    pval_b1 = result_reml.pvalues.get("age_c", float("nan"))
    pval_b2 = result_reml.pvalues.get("age_c_sq", float("nan"))
    has_ext = "mean_ext" in result_reml.fe_params.index
    ext_coef = result_reml.fe_params["mean_ext"] if has_ext else None
    ext_pval = result_reml.pvalues["mean_ext"] if has_ext else None
    is_sig_b1 = pd.notna(pval_b1) and pval_b1 < 0.05
    is_sig_b2 = pd.notna(pval_b2) and pval_b2 < 0.05

    return {
        "experiment": experiment,
        "pitch_type": pitch_type,
        "outcome": outcome,
        "intercept": result_reml.fe_params["Intercept"],
        "b1": b1,
        "b2": b2,
        "pval_b1": pval_b1,
        "pval_b2": pval_b2,
        "significant": is_sig_b1 or is_sig_b2,
        "ext_coef": ext_coef,
        "ext_pval": ext_pval,
        "peak_age": peak_age,
        "decline_rate_at_mean": b1,
        "sigma_u": float(result_reml.cov_re.values[0][0]) ** 0.5,
        "sigma": result_reml.scale ** 0.5,
        "aic": result_mle.aic,
        "bic": result_mle.bic,
        "n_obs": int(result_reml.nobs),
        "n_groups": get_n_groups(result_reml),
        "is_linear_model": False,
    }


def build_linear_equation(outcome: str, experiment: str) -> str:
    """Build linear mixed-model equation matching the current experiment."""
    if experiment == "with_ext":
        return f"{outcome} ~ age_c + C(year) + mean_ext"
    return f"{outcome} ~ age_c + C(year)"


def fit_with_linear_fallback(
    model_df: pd.DataFrame,
    outcome: str,
    experiment: str,
    log,
) -> tuple[MixedLMResults, MixedLMResults, bool]:
    """Fit quadratic and linear models, selecting linear when justified by p-value and AIC."""
    eq_quad = EXPERIMENTS[experiment](outcome)
    eq_linear = build_linear_equation(outcome, experiment)

    result_quad_reml = fit_model(model_df, eq_quad, reml=True)
    result_quad_mle = fit_model(model_df, eq_quad, reml=False)
    result_lin_reml = fit_model(model_df, eq_linear, reml=True)
    result_lin_mle = fit_model(model_df, eq_linear, reml=False)

    b2_pval = result_quad_reml.pvalues.get("age_c_sq", float("nan"))
    is_linear = pd.notna(b2_pval) and b2_pval > 0.05 and result_lin_mle.aic < result_quad_mle.aic

    if is_linear:
        log.info(
            f"  -> linear preferred (b2 p={b2_pval:.3f}, "
            f"AIC quad={result_quad_mle.aic:.1f} vs linear={result_lin_mle.aic:.1f})"
        )
        return result_lin_reml, result_lin_mle, True

    return result_quad_reml, result_quad_mle, False

def fit_model(
    data: pd.DataFrame,
    equation: str,
    reml: bool = True,
    method: str = "lbfgs",
    verbose: bool = False,
) -> MixedLMResults:
    model = smf.mixedlm(equation, data, groups=data["pitcher"])
    result = model.fit(method=[method], reml=reml)
    if verbose:
        print(result.summary())
    return result


if __name__ == "__main__":
    start = time.time()
    log = setup_logger("models", MASTER_DATA_DIR / "model_results.log")

    df = load_data()
    age_mean = get_age_mean(df)
    log.info(f"Age mean: {age_mean:.2f}")

    if {"mean_pfx_x", "p_throws"}.issubset(df.columns):
        df["mean_pfx_x_norm"] = df["mean_pfx_x"].where(df["p_throws"] != "L", -df["mean_pfx_x"])
        log.info("Normalized mean_pfx_x into mean_pfx_x_norm using pitcher handedness")
    else:
        missing = [c for c in ["mean_pfx_x", "p_throws"] if c not in df.columns]
        log.warning(f"Could not create mean_pfx_x_norm; missing columns: {missing}")

    pitch_type_dict = get_data_pitch_type_dict(df, get_valid_pitch_types())
    log.info(f"Loaded {len(df)} rows across {len(pitch_type_dict)} pitch types")
    log.info(f"Outcomes: {OUTCOMES}")
    log.info(f"Experiments: {list(EXPERIMENTS.keys())}")

    all_results = []
    fitted_models = {}
    total = len(EXPERIMENTS) * len(pitch_type_dict) * len(OUTCOMES)
    count = 0

    for experiment, eq_fn in EXPERIMENTS.items():
        log.info(f"{'='*50}")
        log.info(f"Experiment: {experiment}")
        log.info(f"{'='*50}")

        for pitch_type, pt_df in pitch_type_dict.items():
            for outcome in OUTCOMES:
                count += 1
                required = [outcome, "age_c", "age_c_sq"]
                if experiment == "with_ext":
                    required.append("mean_ext")

                model_df = pt_df.dropna(subset=required).copy()

                if len(model_df) < 50:
                    log.warning(
                        f"[{count}/{total}] SKIP {experiment} | {pitch_type} | "
                        f"{outcome} — insufficient data ({len(model_df)} rows)"
                    )
                    continue

                equation = eq_fn(outcome)
                log.info(
                    f"[{count}/{total}] Fitting {experiment} | "
                    f"{pitch_type} | {outcome} | n={len(model_df)}"
                )

                try:
                    result_reml, result_mle, is_linear_model = fit_with_linear_fallback(
                        model_df=model_df,
                        outcome=outcome,
                        experiment=experiment,
                        log=log,
                    )

                    row = extract_results(
                        result_reml,
                        result_mle,
                        pitch_type,
                        outcome,
                        experiment,
                        age_mean,
                    )
                    row["is_linear_model"] = is_linear_model
                    all_results.append(row)
                    fitted_models[f"{experiment}_{pitch_type}_{outcome}"] = result_reml

                    peak_str = f"peak={row['peak_age']:.1f}" if row["peak_age"] else "monotonic"
                    log.info(
                        f"[{count}/{total}] {experiment} | {pitch_type} | {outcome} | "
                        f"{peak_str} | {'sig' if row['significant'] else 'not sig'} | "
                        f"AIC={row['aic']:.1f} | linear={'Y' if row['is_linear_model'] else 'N'}"
                    )

                except Exception as e:
                    log.error(f"  → FAILED: {e}")

    results_df = pd.DataFrame(all_results)
    log.info(f"Computed {len(results_df)} models")

    for experiment in EXPERIMENTS:
        exp_df = results_df[results_df["experiment"] == experiment]
        sig_df = exp_df[exp_df["significant"]]

        log.info(f"\n{'='*50}")
        log.info(f"Summary: {experiment} — {len(sig_df)}/{len(exp_df)} significant")

        if not sig_df.empty:
            log.info(
                "\n"
                + sig_df[
                    [
                        "pitch_type",
                        "outcome",
                        "peak_age",
                        "decline_rate_at_mean",
                        "is_linear_model",
                        "pval_b1",
                        "pval_b2",
                        "aic",
                    ]
                ].to_string(index=False)
            )

        not_sig = exp_df[~exp_df["significant"]]
        if not not_sig.empty:
            names = list(zip(not_sig["pitch_type"], not_sig["outcome"]))
            log.info(f"Not significant: {names}")

    # AIC comparison — show meaningful deltas only.
    log.info("\nAIC delta (with_ext minus base) — negative = ext improves:")
    aic_compare = results_df.pivot_table(
        index=["pitch_type", "outcome"],
        columns="experiment",
        values="aic",
    )
    aic_compare["delta"] = aic_compare["with_ext"] - aic_compare["base"]
    aic_compare["improves"] = aic_compare["delta"] < 0
    meaningful = aic_compare[aic_compare["delta"].abs() > 2][["delta", "improves"]]
    log.info("\n" + meaningful.to_string())

    # Full side-by-side comparison for all model stats.
    base_df = results_df[results_df["experiment"] == "base"].drop(columns=["experiment"])
    with_ext_df = results_df[results_df["experiment"] == "with_ext"].drop(columns=["experiment"])

    compare_df = base_df.merge(
        with_ext_df,
        on=["pitch_type", "outcome"],
        how="outer",
        suffixes=("_base", "_with_ext"),
    )

    compare_df["aic_delta"] = compare_df["aic_with_ext"] - compare_df["aic_base"]
    compare_df["bic_delta"] = compare_df["bic_with_ext"] - compare_df["bic_base"]
    compare_df["b1_delta"] = compare_df["b1_with_ext"] - compare_df["b1_base"]
    compare_df["b2_delta"] = compare_df["b2_with_ext"] - compare_df["b2_base"]
    compare_df["peak_age_delta"] = compare_df["peak_age_with_ext"] - compare_df["peak_age_base"]
    compare_df["decline_rate_delta"] = (
        compare_df["decline_rate_at_mean_with_ext"] - compare_df["decline_rate_at_mean_base"]
    )
    compare_df["sigma_u_delta"] = compare_df["sigma_u_with_ext"] - compare_df["sigma_u_base"]
    compare_df["sigma_delta"] = compare_df["sigma_with_ext"] - compare_df["sigma_base"]

    compare_df["sig_base"] = compare_df["significant_base"].fillna(False).astype(bool)
    compare_df["sig_with_ext"] = compare_df["significant_with_ext"].fillna(False).astype(bool)
    compare_df["sig_changed"] = compare_df["sig_base"] != compare_df["sig_with_ext"]
    compare_df["ext_improves"] = compare_df["aic_with_ext"] < compare_df["aic_base"]

    elapsed = time.time() - start
    log.info(f"Total runtime: {elapsed / 60:.1f} minutes")