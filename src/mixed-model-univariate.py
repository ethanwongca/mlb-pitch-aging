"""
Univariate mixed effects models for MLB pitch aging study.

Fits one model per pitch type × outcome combination.
Formula: outcome ~ age_c + age_c_sq + C(year) + (1|pitcher)

Outputs:
    master_data/model_results.csv   — summary table of all fitted models
"""

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLMResults
from pathlib import Path

from utils import (
    build_univariate_equation,
    get_age_mean,
    get_data_pitch_type_dict,
    get_default_outcomes,
    get_valid_pitch_types,
    is_realistic_peak_age,
    load_data,
)

BASE_DIR = Path(__file__).resolve().parent.parent
MASTER_DATA_DIR = BASE_DIR / "master_data"

# All outcomes; significance filtering determines what is retained.
OUTCOMES = get_default_outcomes()


def get_n_groups(result: MixedLMResults) -> int:
    """Return group count across statsmodels versions."""
    if hasattr(result, "ngroups"):
        return int(result.ngroups)

    model = result.model
    if hasattr(model, "group_labels"):
        return int(len(model.group_labels))

    if hasattr(model, "groups"):
        return int(pd.Series(model.groups).nunique())

    return 0


def extract_results(
    result: MixedLMResults,
    pitch_type: str,
    outcome: str,
    age_mean: float,
) -> dict:
    b1 = result.fe_params["age_c"]
    b2 = result.fe_params["age_c_sq"]
    peak_age = age_mean + (-b1 / (2 * b2)) if b2 < 0 else None

    # p-values for age terms — tells you if aging signal is significant
    pval_b1 = result.pvalues["age_c"]
    pval_b2 = result.pvalues["age_c_sq"]

    return {
        "pitch_type": pitch_type,
        "outcome": outcome,
        "intercept": result.fe_params["Intercept"],
        "b1": b1,
        "b2": b2,
        "pval_b1": pval_b1,
        "pval_b2": pval_b2,
        "significant": pval_b1 < 0.05 or pval_b2 < 0.05,
        "peak_age": peak_age,
        "peak_age_realistic": is_realistic_peak_age(peak_age),
        "sigma_u": float(result.cov_re.values[0][0]) ** 0.5,
        "sigma": result.scale ** 0.5,
        "aic": result.aic,
        "bic": result.bic,
        "n_obs": int(result.nobs),
        "n_groups": get_n_groups(result),
    }

def fit_univariate_mixed_model(
    data: pd.DataFrame,
    groups: str,
    equation: str,
    method: str = "lbfgs",
    reml: bool = True,
    verbose: bool = False,
) -> MixedLMResults:
    model = smf.mixedlm(equation, data, groups=data[groups])
    result = model.fit(method=[method], reml=reml)
    if verbose:
        print(result.summary())
    return result

if __name__ == "__main__":
    df = load_data()
    age_mean = get_age_mean(df)
    print(f"Age mean: {age_mean:.2f}")

    pitch_type_dict = get_data_pitch_type_dict(df, get_valid_pitch_types())
    print(f"Loaded {len(df)} rows across {len(pitch_type_dict)} pitch types")

    all_results = []
    fitted_models = {}

    for pitch_type, pt_df in pitch_type_dict.items():
        for outcome in OUTCOMES:
            model_df = pt_df.dropna(subset=[outcome, "age_c", "age_c_sq"]).copy()

            if len(model_df) < 50:
                print(
                    f"Skipping {pitch_type} — {outcome}: "
                    f"insufficient data ({len(model_df)} rows)"
                )
                continue

            print(f"Fitting {pitch_type} — {outcome} ({len(model_df)} rows)...")
            equation = build_univariate_equation(outcome)

            try:
                result = fit_univariate_mixed_model(
                    data=model_df,
                    groups="pitcher",
                    equation=equation,
                )
                row = extract_results(result, pitch_type, outcome, age_mean)
                all_results.append(row)
                fitted_models[f"{pitch_type}_{outcome}"] = result

                peak_str = f"peak_age={row['peak_age']:.1f}" if row["peak_age"] else "no peak"
                sig_str = "significant" if row["significant"] else "not significant"
                print(f"  {peak_str}  {sig_str}")

            except Exception as e:
                print(f"  FAILED: {e}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(MASTER_DATA_DIR / "model_results.csv", index=False)

    print(f"\nSaved {len(results_df)} models")
    print(
        results_df[
            [
                "pitch_type",
                "outcome",
                "peak_age",
                "pval_b1",
                "pval_b2",
                "significant",
                "aic",
            ]
        ].to_string()
    )