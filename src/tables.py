"""Generate all supplementary and main paper tables as CSVs."""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MASTER_DATA_DIR = BASE_DIR / "master_data"
TABLES_DIR = BASE_DIR / "tables"

PITCH_ORDER = ["FF", "SL", "SI", "CH", "CU", "FC"]


def _load_peak_cis(experiment: str) -> pd.DataFrame:
    """Load peak age CIs — supports both Bayesian (posteriors) and delta-method outputs."""
    for name in [
        f"peak_age_posteriors_{experiment}.csv",
        f"peak_age_cis_{experiment}.csv",
    ]:
        path = MASTER_DATA_DIR / name
        if path.exists():
            df = pd.read_csv(path)
            return df.rename(columns={"hdi_lo": "ci_lo", "hdi_hi": "ci_hi"})
    raise FileNotFoundError(f"No peak age CI file found for experiment='{experiment}'")


def _load_decline_cis(experiment: str) -> pd.DataFrame:
    """Load decline rate CIs — supports both Bayesian and delta-method outputs."""
    for name in [
        f"decline_rate_posteriors_{experiment}.csv",
        f"decline_rate_cis_{experiment}.csv",
    ]:
        path = MASTER_DATA_DIR / name
        if path.exists():
            df = pd.read_csv(path)
            return df.rename(
                columns={"rate_mean": "rate", "hdi_lo": "ci_lo", "hdi_hi": "ci_hi"}
            )
    raise FileNotFoundError(
        f"No decline rate CI file found for experiment='{experiment}'"
    )


def build_table_s1(
    results_df: pd.DataFrame,
    peak_base: pd.DataFrame,
    decline_base: pd.DataFrame,
) -> pd.DataFrame:
    """Complete model results for ALL outcomes including non-significant ones."""
    base = results_df[results_df["experiment"] == "base"].copy()

    dec28 = decline_base[decline_base["eval_age"] == 28][
        ["pitch_type", "outcome", "rate", "ci_lo", "ci_hi"]
    ].rename(
        columns={
            "rate": "decline_at_28",
            "ci_lo": "decline_ci_lo",
            "ci_hi": "decline_ci_hi",
        }
    )

    peak_cols = peak_base[["pitch_type", "outcome", "ci_lo", "ci_hi"]].rename(
        columns={"ci_lo": "peak_ci_lo", "ci_hi": "peak_ci_hi"}
    )

    # Normalize peak_age column (Bayesian uses peak_age_mean, delta method uses peak_age)
    if "peak_age_mean" in base.columns and "peak_age" not in base.columns:
        base = base.rename(columns={"peak_age_mean": "peak_age"})

    # Normalize significance columns
    sig_cols = ["pval_b1", "pval_b2", "b1_significant", "b2_significant"]
    keep_sig = [c for c in sig_cols if c in base.columns]

    base_cols = (
        ["pitch_type", "outcome", "significant", "peak_age"]
        + keep_sig
        + [c for c in ["is_linear_model", "n_obs", "n_groups"] if c in base.columns]
    )
    t = (
        base[base_cols]
        .merge(peak_cols, on=["pitch_type", "outcome"], how="left")
        .merge(dec28, on=["pitch_type", "outcome"], how="left")
    )

    t["pitch_type"] = pd.Categorical(
        t["pitch_type"], categories=PITCH_ORDER, ordered=True
    )
    return t.sort_values(["pitch_type", "outcome"]).reset_index(drop=True)


def build_table_s2(peak_base: pd.DataFrame, peak_ext: pd.DataFrame) -> pd.DataFrame:
    """Full peak age CI table — base and with_ext side by side."""
    peak_col = "peak_age_mean" if "peak_age_mean" in peak_base.columns else "peak_age"

    b = peak_base[["pitch_type", "outcome", peak_col, "ci_lo", "ci_hi"]].rename(
        columns={
            peak_col: "base_peak_age",
            "ci_lo": "base_ci_lo",
            "ci_hi": "base_ci_hi",
        }
    )
    e = peak_ext[["pitch_type", "outcome", peak_col, "ci_lo", "ci_hi"]].rename(
        columns={peak_col: "ext_peak_age", "ci_lo": "ext_ci_lo", "ci_hi": "ext_ci_hi"}
    )
    t = b.merge(e, on=["pitch_type", "outcome"], how="outer")
    t["pitch_type"] = pd.Categorical(
        t["pitch_type"], categories=PITCH_ORDER, ordered=True
    )
    return t.sort_values(["pitch_type", "outcome"]).reset_index(drop=True)


def build_table_s3(decline_base: pd.DataFrame) -> pd.DataFrame:
    """Full decline rate CI table — all pitch types, all outcomes, ages 24/28/32/36."""
    t = decline_base[decline_base["eval_age"].isin([24, 28, 32, 36])].copy()
    t["pitch_type"] = pd.Categorical(
        t["pitch_type"], categories=PITCH_ORDER, ordered=True
    )
    return t.sort_values(["pitch_type", "outcome", "eval_age"]).reset_index(drop=True)



def build_table_s5(scg_df: pd.DataFrame) -> pd.DataFrame:
    """SCG summary by pitch type."""
    t = scg_df[["pitch_type", "velo_peak_age", "velo_hdi_lo", "velo_hdi_hi",
                 "spin_peak_age", "spin_hdi_lo", "spin_hdi_hi", "scg"]].copy()
    t = t.round(2)
    t["pitch_type"] = pd.Categorical(
        t["pitch_type"], categories=PITCH_ORDER, ordered=True
    )
    return t.sort_values("pitch_type").reset_index(drop=True)


def build_table1(decline_base: pd.DataFrame) -> pd.DataFrame:
    """Table 1: Velocity decline rates at age 28 with 95% CIs — all 6 pitch types."""
    t = (
        decline_base[
            (decline_base["outcome"] == "mean_velo") & (decline_base["eval_age"] == 28)
        ][["pitch_type", "rate", "ci_lo", "ci_hi"]]
        .rename(
            columns={"rate": "decline_per_yr", "ci_lo": "ci_lo_95", "ci_hi": "ci_hi_95"}
        )
        .copy()
    )
    t["pitch_type"] = pd.Categorical(
        t["pitch_type"], categories=PITCH_ORDER, ordered=True
    )
    return t.sort_values("pitch_type").reset_index(drop=True)


def build_table2(peak_base: pd.DataFrame) -> pd.DataFrame:
    """Table 2: Spin rate peak ages with 95% CIs — FF SL SI CH CU (FC excluded)."""
    peak_col = "peak_age_mean" if "peak_age_mean" in peak_base.columns else "peak_age"
    t = (
        peak_base[
            (peak_base["outcome"] == "mean_spin_rate")
            & (peak_base["pitch_type"].isin(["FF", "SL", "SI", "CH", "CU"]))
        ][["pitch_type", peak_col, "ci_lo", "ci_hi"]]
        .rename(columns={peak_col: "peak_age"})
        .copy()
    )
    order = ["FF", "SL", "SI", "CH", "CU"]
    t["pitch_type"] = pd.Categorical(t["pitch_type"], categories=order, ordered=True)
    return t.sort_values("pitch_type").reset_index(drop=True)


def build_table3() -> pd.DataFrame:
    """Table 3: Bivariate correlation results — FF and SI posterior mean and 95% HDI."""
    try:
        import arviz as az
    except ImportError:
        print("  arviz not available — skipping Table 3")
        return pd.DataFrame()

    rows = []
    for pt in ["FF", "SI"]:
        nc_path = MASTER_DATA_DIR / f"bivariate_{pt}.nc"
        if not nc_path.exists():
            print(f"  Missing {nc_path.name} — skipping {pt}")
            continue
        idata = az.from_netcdf(str(nc_path))
        rho = idata.posterior["chol_corr"].values[:, :, 0, 1].flatten()
        hdi = az.hdi(rho, hdi_prob=0.95)
        rows.append(
            {
                "pitch_type": pt,
                "rho_mean": round(float(rho.mean()), 3),
                "hdi_lo_95": round(float(hdi[0]), 3),
                "hdi_hi_95": round(float(hdi[1]), 3),
                "n_posterior_samples": len(rho),
            }
        )
    return pd.DataFrame(rows)


def save(df: pd.DataFrame, name: str) -> None:
    path = TABLES_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"  Saved: {path.name}  ({len(df)} rows)")


if __name__ == "__main__":
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading source data...")
    results_df = pd.read_csv(MASTER_DATA_DIR / "model_results.csv")
    peak_base = _load_peak_cis("base")
    peak_ext = _load_peak_cis("with_ext")
    decline_base = _load_decline_cis("base")
    scg_df = pd.read_csv(MASTER_DATA_DIR / "scg_results.csv")

    print("\nGenerating supplementary tables...")
    save(build_table_s1(results_df, peak_base, decline_base), "table_s1_model_results")
    save(build_table_s2(peak_base, peak_ext), "table_s2_peak_age_cis")
    save(build_table_s3(decline_base), "table_s3_decline_rate_cis")
    save(build_table_s5(scg_df), "table_s4_scg_summary")

    print("\nGenerating main paper tables...")
    save(build_table1(decline_base), "table1_velo_decline_rates")
    save(build_table2(peak_base), "table2_spin_peak_ages")

    print("\nGenerating Table 3 (bivariate correlation)...")
    t3 = build_table3()
    if not t3.empty:
        save(t3, "table3_bivariate_correlation")

    print("\nDone — all tables saved to", TABLES_DIR)
