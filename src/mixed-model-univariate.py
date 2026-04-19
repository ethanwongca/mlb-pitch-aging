"""
Univariate mixed effects models for MLB pitch aging study.

Fits one model per pitch type × outcome combination.

Experiment A (base):     outcome ~ age_c + age_c_sq + C(year) + (1|pitcher)
Experiment B (with_ext): outcome ~ age_c + age_c_sq + C(year) + mean_ext + (1|pitcher)

REML=True  for parameter estimation (unbiased variance components)
REML=False for AIC/BIC model comparison (AIC not valid under REML)

Outputs:
    master_data/model_results.csv
    master_data/model_results.log
"""

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLMResults
from pathlib import Path

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
    b1 = result_reml.fe_params["age_c"]
    b2 = result_reml.fe_params["age_c_sq"]
    peak_age = age_mean + (-b1 / (2 * b2)) if b2 < 0 else None
    pval_b1 = result_reml.pvalues["age_c"]
    pval_b2 = result_reml.pvalues["age_c_sq"]
    has_ext = "mean_ext" in result_reml.fe_params.index
    ext_coef = result_reml.fe_params["mean_ext"] if has_ext else None
    ext_pval = result_reml.pvalues["mean_ext"] if has_ext else None

    return {
        "experiment": experiment,
        "pitch_type": pitch_type,
        "outcome": outcome,
        "intercept": result_reml.fe_params["Intercept"],
        "b1": b1,
        "b2": b2,
        "pval_b1": pval_b1,
        "pval_b2": pval_b2,
        "significant": pval_b1 < 0.05 or pval_b2 < 0.05,
        "ext_coef": ext_coef,
        "ext_pval": ext_pval,
        "peak_age": peak_age,
        "sigma_u": float(result_reml.cov_re.values[0][0]) ** 0.5,
        "sigma": result_reml.scale ** 0.5,
        "aic": result_mle.aic,
        "bic": result_mle.bic,
        "n_obs": int(result_reml.nobs),
        "n_groups": get_n_groups(result_reml),
    }

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
    log = setup_logger("models", MASTER_DATA_DIR / "model_results.log")

    df = load_data()
    age_mean = get_age_mean(df)
    log.info(f"Age mean: {age_mean:.2f}")

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
                log.info(f"  equation: {equation}")

                try:
                    result_reml = fit_model(model_df, equation, reml=True)
                    result_mle = fit_model(model_df, equation, reml=False)

                    row = extract_results(
                        result_reml,
                        result_mle,
                        pitch_type,
                        outcome,
                        experiment,
                        age_mean,
                    )
                    all_results.append(row)
                    fitted_models[f"{experiment}_{pitch_type}_{outcome}"] = result_reml

                    peak_str = f"peak={row['peak_age']:.1f}" if row["peak_age"] else "no peak (monotonic decline)"
                    log.info(
                        f"  → {peak_str} | "
                        f"{'sig' if row['significant'] else 'not sig'} | "
                        f"AIC={row['aic']:.1f} | "
                        f"sigma_u={row['sigma_u']:.3f}"
                    )

                except Exception as e:
                    log.error(f"  → FAILED: {e}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(MASTER_DATA_DIR / "model_results.csv", index=False)
    log.info(f"Saved {len(results_df)} models to model_results.csv")

    for experiment in EXPERIMENTS:
        exp_df = results_df[results_df["experiment"] == experiment].copy()
        sig_df = exp_df[exp_df["significant"]]
        not_sig_df = exp_df[~exp_df["significant"]]

        log.info(f"\nSummary ({experiment})")
        log.info(f"  total models: {len(exp_df)}")
        log.info(f"  significant: {len(sig_df)}")
        log.info(f"  not significant: {len(not_sig_df)}")

        if not sig_df.empty:
            log.info(f"\nSignificant models ({experiment}):")
            log.info(
                "\n"
                + sig_df[
                    [
                        "pitch_type",
                        "outcome",
                        "peak_age",
                        "pval_b1",
                        "pval_b2",
                        "aic",
                        "bic",
                        "n_obs",
                        "n_groups",
                    ]
                ].to_string(index=False)
            )

        if not not_sig_df.empty:
            log.info(f"\nNot significant models ({experiment}):")
            log.info(
                "\n"
                + not_sig_df[
                    [
                        "pitch_type",
                        "outcome",
                        "peak_age",
                        "pval_b1",
                        "pval_b2",
                        "aic",
                        "bic",
                        "n_obs",
                        "n_groups",
                    ]
                ].to_string(index=False)
            )

    # AIC comparison — does extension improve fit?
    log.info("\nAIC comparison (base vs with_ext):")
    aic_compare = results_df.pivot_table(
        index=["pitch_type", "outcome"],
        columns="experiment",
        values="aic",
    )
    aic_compare["ext_improves"] = aic_compare["with_ext"] < aic_compare["base"]
    aic_compare["aic_delta"] = aic_compare["with_ext"] - aic_compare["base"]
    log.info("\n" + aic_compare.to_string())

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
    compare_df["sigma_u_delta"] = compare_df["sigma_u_with_ext"] - compare_df["sigma_u_base"]
    compare_df["sigma_delta"] = compare_df["sigma_with_ext"] - compare_df["sigma_base"]

    compare_df["sig_base"] = compare_df["significant_base"].fillna(False).astype(bool)
    compare_df["sig_with_ext"] = compare_df["significant_with_ext"].fillna(False).astype(bool)
    compare_df["sig_changed"] = compare_df["sig_base"] != compare_df["sig_with_ext"]
    compare_df["ext_improves"] = compare_df["aic_with_ext"] < compare_df["aic_base"]

    compare_path = MASTER_DATA_DIR / "model_results_comparison.csv"
    compare_df.to_csv(compare_path, index=False)
    log.info(f"Saved full comparison table to {compare_path.name}")

    compare_cols = [
        "pitch_type",
        "outcome",
        "sig_base",
        "sig_with_ext",
        "sig_changed",
        "pval_b1_base",
        "pval_b2_base",
        "pval_b1_with_ext",
        "pval_b2_with_ext",
        "ext_coef_with_ext",
        "ext_pval_with_ext",
        "aic_base",
        "aic_with_ext",
        "aic_delta",
        "bic_base",
        "bic_with_ext",
        "bic_delta",
        "peak_age_base",
        "peak_age_with_ext",
        "peak_age_delta",
        "ext_improves",
    ]
    log.info("\nFull comparison (base vs with_ext):")
    log.info("\n" + compare_df[compare_cols].to_string(index=False))