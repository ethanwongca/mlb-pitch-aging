"""Bayesian convergence diagnostics for ArviZ InferenceData objects."""

import arviz as az


def check_convergence(
    idata: az.InferenceData,
    key: str,
    log,
    var_names: list[str] | None = None,
) -> bool:
    """
    Check MCMC convergence diagnostics.
    Returns True if all checks pass, False if any are suspect.

    Criteria:
        R-hat  < 1.01   (chain mixing)
        ESS    > 400    (effective samples)
        Divergences == 0
    """
    if var_names is None:
        var_names = [v for v in ["age_c", "age_c_sq"] if v in idata.posterior]

    try:
        summary = az.summary(idata, var_names=var_names)
        max_rhat = float(summary["r_hat"].max())
        min_ess = float(summary["ess_bulk"].min())
    except Exception as e:
        log.warning(f"  {key}: convergence summary failed — {e}")
        return False

    n_diverg = int(idata.sample_stats["diverging"].values.sum())

    ok = max_rhat < 1.01 and min_ess > 400 and n_diverg == 0

    if not ok:
        log.warning(
            f"  Convergence issue [{key}]: "
            f"max_rhat={max_rhat:.3f}  "
            f"min_ess={min_ess:.0f}  "
            f"divergences={n_diverg}"
        )
    return ok
