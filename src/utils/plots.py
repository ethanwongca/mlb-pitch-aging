"""Reusable plotting functions for the MLB pitch aging study."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

PITCH_COLORS = {
    "FF": "#E63946",
    "SL": "#457B9D",
    "SI": "#F4A261",
    "CH": "#2A9D8F",
    "CU": "#9B5DE5",
    "FC": "#F72585",
}
OUTCOME_LABELS = {
    "mean_velo":       "Velocity (mph)",
    "mean_spin_rate":  "Spin Rate (rpm)",
    "mean_pfx_x_norm": "Horizontal Break, norm (ft)",
    "mean_pfx_z":      "Vertical Break (ft)",
    "mean_spin_axis":  "Spin Axis (°)",
}
PITCH_LABELS = {
    "FF": "4-Seam FB", "SL": "Slider", "SI": "Sinker",
    "CH": "Changeup",  "CU": "Curveball", "FC": "Cutter",
}

plt.rcParams.update({
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         10,
})

MIN_N_NAIVE = 15   # minimum pitchers per age for naive means


def _predict_curve(row: pd.Series, age_grid_c: np.ndarray) -> np.ndarray:
    """Reconstruct population-level predicted values from stored coefficients."""
    y = row["intercept"] + row["b1"] * age_grid_c
    if not row["is_linear_model"] and pd.notna(row.get("b2")):
        y = y + row["b2"] * age_grid_c ** 2
    return y


def _age_grid(age_mean: float, lo: int = 21, hi: int = 38, n: int = 200):
    ages   = np.linspace(lo, hi, n)
    ages_c = ages - age_mean
    return ages, ages_c


def _naive_means(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """
    Per-age naive means with standard error and pitcher count.
    Filters to ages with MIN_N_NAIVE+ pitchers.
    """
    return (
        df.groupby("age")[outcome]
        .agg(
            mean="mean",
            sem=lambda x: x.std() / max(len(x) ** 0.5, 1),
            n="count",
        )
        .reset_index()
        .query(f"n >= {MIN_N_NAIVE}")
    )


def _plot_naive(ax, naive: pd.DataFrame, color: str = "gray") -> None:
    """Uniform-size dots with standard error bars — paper-quality naive means."""
    ax.errorbar(
        naive["age"], naive["mean"],
        yerr=naive["sem"],
        fmt="o",
        markersize=4,
        color="lightgray",
        ecolor="gray",
        elinewidth=0.8,
        capsize=2,
        zorder=2,
        label="Naive means ± SE",
    )


def _plot_mixed_curve(
    ax, ages: np.ndarray, y_shifted: np.ndarray,
    color: str, row: pd.Series, age_mean: float,
) -> None:
    """Mixed-effects curve with optional peak annotation."""
    ax.plot(ages, y_shifted, color=color, lw=2.5, zorder=3, label="Mixed-effects")

    peak_age = row.get("peak_age")
    if pd.notna(peak_age) and 21 <= peak_age <= 38:
        pa_c  = peak_age - age_mean
        y_pk  = (
            _predict_curve(row, np.array([pa_c]))[0]
            - _predict_curve(row, ages - age_mean).mean()
            + y_shifted.mean()
        )
        ax.axvline(peak_age, color=color, lw=1, ls="--", alpha=0.6)
        ax.annotate(
            f"Peak {peak_age:.1f}",
            xy=(peak_age, y_pk),
            xytext=(4, 6), textcoords="offset points",
            fontsize=8, color=color,
        )


def _shift_curve(row: pd.Series, ages_c: np.ndarray, obs_mean: float) -> np.ndarray:
    """Shift curve to observed mean for visual alignment."""
    y = _predict_curve(row, ages_c)
    return y - y.mean() + obs_mean


def plot_aging_curves_grid(
    results_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    age_mean: float,
    out_dir: Path,
    experiment: str = "base",
    outcomes: list[str] | None = None,
) -> None:
    """
    One figure per outcome. Rows = pitch types, two panels per row:
      left  — mixed-effects curve vs naive means (uniform dots + SE bars)
      right — sample size by age (shows survivorship)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_df = results_df[results_df["experiment"] == experiment]
    sig_df = exp_df[exp_df["significant"]]

    if outcomes is None:
        outcomes = sig_df["outcome"].unique().tolist()

    ages, ages_c = _age_grid(age_mean)

    for outcome in outcomes:
        pitch_rows  = sig_df[sig_df["outcome"] == outcome]
        pitch_types = pitch_rows["pitch_type"].tolist()
        if not pitch_types:
            continue

        n_pt = len(pitch_types)
        fig, axes = plt.subplots(
            n_pt, 2,
            figsize=(10, 2.8 * n_pt),
            gridspec_kw={"width_ratios": [3, 1]},
        )
        if n_pt == 1:
            axes = [axes]

        ylabel = OUTCOME_LABELS.get(outcome, outcome)
        fig.suptitle(
            f"Aging Curves — {ylabel}  [{experiment}]",
            fontsize=13, fontweight="bold", y=1.01,
        )

        for ax_row, pt in zip(axes, pitch_types):
            ax_curve, ax_n = ax_row
            row   = pitch_rows[pitch_rows["pitch_type"] == pt].iloc[0]
            color = PITCH_COLORS.get(pt, "steelblue")

            pt_raw = raw_df[raw_df["pitch_type"] == pt].dropna(subset=[outcome, "age"])
            naive  = _naive_means(pt_raw, outcome)

            _plot_naive(ax_curve, naive)

            y_shifted = _shift_curve(row, ages_c, pt_raw[outcome].mean())
            _plot_mixed_curve(ax_curve, ages, y_shifted, color, row, age_mean)

            ax_curve.set_ylabel(ylabel, fontsize=9)
            ax_curve.set_xlabel("Age")
            ax_curve.set_title(
                PITCH_LABELS.get(pt, pt), fontsize=10,
                loc="left", fontweight="bold", color=color,
            )
            ax_curve.legend(fontsize=8, frameon=False)
            ax_curve.xaxis.set_major_locator(mticker.MultipleLocator(2))

            n_by_age = pt_raw.groupby("age").size().reset_index(name="n")
            ax_n.bar(n_by_age["age"], n_by_age["n"],
                     color=color, alpha=0.6, width=0.8)
            ax_n.set_xlabel("Age")
            ax_n.set_ylabel("N pitchers", fontsize=8)
            ax_n.yaxis.set_label_position("right")
            ax_n.yaxis.tick_right()
            ax_n.xaxis.set_major_locator(mticker.MultipleLocator(4))

        fig.tight_layout()
        fname = out_dir / f"aging_curve_{outcome}_{experiment}.png"
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname.name}")


def plot_spin_velo_divergence(
    results_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    age_mean: float,
    out_dir: Path,
    pitch_types: list[str] | None = None,
    experiment: str = "base",
) -> None:
    """
    Overlay spin rate and velocity aging curves per pitch type.
    Dual y-axis. Highlights that spin peaks while velocity is already declining.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if pitch_types is None:
        pitch_types = ["FF", "SI", "SL"]

    ages, ages_c = _age_grid(age_mean)
    exp_df = results_df[results_df["experiment"] == experiment]

    n   = len(pitch_types)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=False)
    if n == 1:
        axes = [axes]

    fig.suptitle(
        "Velocity vs Spin Rate Aging — Mid-Career Spin Peak",
        fontsize=12, fontweight="bold",
    )

    for ax, pt in zip(axes, pitch_types):
        color  = PITCH_COLORS.get(pt, "steelblue")
        pt_raw = raw_df[raw_df["pitch_type"] == pt]
        ax2    = ax.twinx()

        for outcome, target_ax, ls, ylabel, label in [
            ("mean_velo",      ax,  "-",  "Velocity (mph)",   "Velocity"),
            ("mean_spin_rate", ax2, "--", "Spin Rate (rpm)",  "Spin Rate"),
        ]:
            row_match = exp_df[
                (exp_df["pitch_type"] == pt) & (exp_df["outcome"] == outcome)
            ]
            if row_match.empty:
                continue
            row = row_match.iloc[0]

            obs_mean  = pt_raw[outcome].dropna().mean()
            y_shifted = _shift_curve(row, ages_c, obs_mean)
            line_color = color if outcome == "mean_spin_rate" else "black"

            target_ax.plot(ages, y_shifted, color=line_color,
                           lw=2, ls=ls, label=label)
            target_ax.set_ylabel(ylabel,
                                  color=line_color, fontsize=8)
            target_ax.tick_params(axis="y", labelcolor=line_color)

            if outcome == "mean_spin_rate":
                peak_age = row.get("peak_age")
                if pd.notna(peak_age) and 21 <= peak_age <= 38:
                    ax2.axvline(peak_age, color=color, lw=1, ls=":", alpha=0.8)
                    ax2.annotate(
                        f"Spin peak\n{peak_age:.1f}",
                        xy=(peak_age, y_shifted.max()),
                        xytext=(4, -20), textcoords="offset points",
                        fontsize=7.5, color=color,
                    )

        ax.set_xlabel("Age")
        ax.set_title(PITCH_LABELS.get(pt, pt), fontsize=10, fontweight="bold")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(2))

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  fontsize=8, frameon=False, loc="lower left")

    fig.tight_layout()
    fname = out_dir / f"spin_velo_divergence_{experiment}.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname.name}")


def plot_survivorship_bias(
    results_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    age_mean: float,
    out_dir: Path,
    pitch_type: str = "FF",
    outcome: str = "mean_velo",
    experiment: str = "base",
) -> None:
    """
    Three-panel figure for one pitch type × outcome:
      1. Naive delta means by age
      2. Sample size by age
      3. Mixed-effects curve vs naive means overlaid
    Makes the survivorship bias argument explicitly for the paper.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_df    = results_df[results_df["experiment"] == experiment]
    row_match = exp_df[
        (exp_df["pitch_type"] == pitch_type) & (exp_df["outcome"] == outcome)
    ]
    if row_match.empty:
        print(f"  No results for {pitch_type} {outcome} [{experiment}] — skipping")
        return

    row    = row_match.iloc[0]
    color  = PITCH_COLORS.get(pitch_type, "steelblue")
    pt_raw = raw_df[raw_df["pitch_type"] == pitch_type].dropna(subset=[outcome, "age"])
    naive  = _naive_means(pt_raw, outcome)
    ages, ages_c = _age_grid(age_mean)
    y_shifted    = _shift_curve(row, ages_c, pt_raw[outcome].mean())

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(
        f"Survivorship Bias — {PITCH_LABELS.get(pitch_type, pitch_type)} "
        f"{OUTCOME_LABELS.get(outcome, outcome)}",
        fontsize=12, fontweight="bold",
    )

    _plot_naive(ax1, naive, color=color)
    ax1.set_title("Naive age means", fontsize=10)
    ax1.set_xlabel("Age")
    ax1.set_ylabel(OUTCOME_LABELS.get(outcome, outcome))
    if not naive.query("age >= 34").empty:
        ax1.annotate(
            "Flattening due to\nsurvivorship bias →",
            xy=(35, naive.query("age >= 34")["mean"].mean()),
            xytext=(30, naive["mean"].max()),
            fontsize=8, color="gray",
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
        )

    n_by_age = pt_raw.groupby("age").size().reset_index(name="n")
    ax2.bar(n_by_age["age"], n_by_age["n"], color=color, alpha=0.65, width=0.8)
    ax2.set_title("Sample size by age", fontsize=10)
    ax2.set_xlabel("Age")
    ax2.set_ylabel("N pitcher-seasons")

    _plot_naive(ax3, naive)
    _plot_mixed_curve(ax3, ages, y_shifted, color, row, age_mean)
    ax3.set_title("Mixed-effects vs naive", fontsize=10)
    ax3.set_xlabel("Age")
    ax3.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    fname = out_dir / f"survivorship_bias_{pitch_type}_{outcome}_{experiment}.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname.name}")


def plot_ext_aic_heatmap(
    results_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Heatmap of AIC delta (with_ext − base) per pitch type × outcome.
    Negative = extension improves fit.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base  = results_df[results_df["experiment"] == "base"].set_index(
        ["pitch_type", "outcome"])["aic"]
    ext   = results_df[results_df["experiment"] == "with_ext"].set_index(
        ["pitch_type", "outcome"])["aic"]
    delta = (ext - base).unstack("outcome")

    pt_order = ["FF", "SL", "SI", "CH", "CU", "FC"]
    oc_order = [o for o in [
        "mean_velo", "mean_spin_rate", "mean_pfx_z",
        "mean_pfx_x_norm", "mean_pfx_x", "mean_spin_axis",
    ] if o in delta.columns]

    delta      = delta.reindex(index=pt_order, columns=oc_order)
    col_labels = [OUTCOME_LABELS.get(c, c) for c in delta.columns]
    row_labels = [PITCH_LABELS.get(r, r) for r in delta.index]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.heatmap(
        delta, ax=ax,
        annot=True, fmt=".1f",
        cmap="RdYlGn_r", center=0,
        linewidths=0.5, linecolor="white",
        xticklabels=col_labels, yticklabels=row_labels,
        cbar_kws={"label": "ΔAIC (with_ext − base)", "shrink": 0.8},
    )
    ax.set_title(
        "Extension Covariate — AIC Improvement over Base Model",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    fname = out_dir / "ext_aic_heatmap.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname.name}")


def plot_delta_method_comparison(
    results_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    age_mean: float,
    out_dir: Path,
    experiment: str = "base",
    outcomes: list[str] | None = None,
) -> None:
    """
    Delta method baseline vs mixed-effects curve.

    One figure per outcome, one panel per pitch type.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_df = results_df[results_df["experiment"] == experiment]
    sig_df = exp_df[exp_df["significant"]]

    if outcomes is None:
        outcomes = sig_df["outcome"].unique().tolist()

    ages, ages_c = _age_grid(age_mean)

    for outcome in outcomes:
        pitch_rows = sig_df[sig_df["outcome"] == outcome]
        pitch_types = pitch_rows["pitch_type"].tolist()
        if not pitch_types:
            continue

        n_pt = len(pitch_types)
        fig, axes = plt.subplots(
            n_pt, 1,
            figsize=(10, 3.2 * n_pt),
        )
        if n_pt == 1:
            axes = [axes]

        ylabel = OUTCOME_LABELS.get(outcome, outcome)
        fig.suptitle(
            f"Delta Method vs Mixed-Effects — {ylabel}  [{experiment}]",
            fontsize=13, fontweight="bold", y=1.01,
        )

        for ax, pt in zip(axes, pitch_types):
            row = pitch_rows[pitch_rows["pitch_type"] == pt].iloc[0]
            color = PITCH_COLORS.get(pt, "steelblue")

            pt_raw = raw_df[raw_df["pitch_type"] == pt].dropna(subset=[outcome, "age"])
            delta = (
                pt_raw.groupby("age")[outcome]
                .agg(
                    mean="mean",
                    sem=lambda x: x.std() / max(len(x) ** 0.5, 1),
                    n="count",
                )
                .reset_index()
                .query(f"n >= {MIN_N_NAIVE}")
            )

            ax.errorbar(
                delta["age"], delta["mean"],
                yerr=delta["sem"],
                fmt="o--",
                markersize=5,
                color="gray",
                ecolor="lightgray",
                elinewidth=0.8,
                capsize=2,
                lw=1.2,
                zorder=2,
                label="Delta method (naive)",
            )

            y_shifted = _shift_curve(row, ages_c, pt_raw[outcome].mean())
            ax.plot(
                ages, y_shifted,
                color=color, lw=2.5, zorder=3,
                label="Mixed-effects",
            )

            peak_age = row.get("peak_age")
            if pd.notna(peak_age) and 21 <= peak_age <= 38:
                pa_c = peak_age - age_mean
                y_pk = float(
                    _predict_curve(row, np.array([pa_c]))[0]
                    - _predict_curve(row, ages_c).mean()
                    + pt_raw[outcome].mean()
                )
                ax.axvline(peak_age, color=color, lw=1, ls="--", alpha=0.6)
                ax.annotate(
                    f"Peak {peak_age:.1f}",
                    xy=(peak_age, y_pk),
                    xytext=(4, 6), textcoords="offset points",
                    fontsize=8, color=color,
                )

            ax.axvspan(33, 38, alpha=0.05, color="red", label="High survivorship bias region")

            decline = row.get("decline_rate_at_mean")
            if pd.notna(decline):
                sign = "+" if decline > 0 else ""
                ax.annotate(
                    f"Mixed-effects decline: {sign}{decline:.3f}/yr at age {age_mean:.0f}",
                    xy=(0.02, 0.05), xycoords="axes fraction",
                    fontsize=7.5, color=color,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
                )

            ax.set_title(
                PITCH_LABELS.get(pt, pt),
                fontsize=10, loc="left",
                fontweight="bold", color=color,
            )
            ax.set_xlabel("Age")
            ax.set_ylabel(ylabel, fontsize=9)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
            ax.legend(fontsize=8, frameon=False)

        fig.tight_layout()
        fname = out_dir / f"delta_vs_mixed_{outcome}_{experiment}.png"
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname.name}")


if __name__ == "__main__":
    from utils import get_age_mean, load_data

    BASE_DIR     = Path(__file__).resolve().parent.parent
    MASTER_DIR   = BASE_DIR / "master_data"
    PLOTS_DIR    = BASE_DIR / "plots"

    print("Loading data...")
    raw_df     = load_data()
    results_df = pd.read_csv(MASTER_DIR / "model_results.csv")
    age_mean   = get_age_mean(raw_df)

    if "mean_pfx_x_norm" not in raw_df.columns:
        raw_df["mean_pfx_x_norm"] = raw_df["mean_pfx_x"].where(
            raw_df["p_throws"] != "L", -raw_df["mean_pfx_x"]
        )

    print("\nPlotting aging curves grid (base)...")
    plot_aging_curves_grid(
        results_df, raw_df, age_mean,
        out_dir=PLOTS_DIR, experiment="base",
    )

    print("\nPlotting aging curves grid (with_ext)...")
    plot_aging_curves_grid(
        results_df, raw_df, age_mean,
        out_dir=PLOTS_DIR, experiment="with_ext",
    )

    print("\nPlotting spin vs velocity divergence...")
    plot_spin_velo_divergence(
        results_df, raw_df, age_mean,
        out_dir=PLOTS_DIR,
        pitch_types=["FF", "SL", "SI", "CH", "CU"],
        experiment="with_ext",
    )

    print("\nPlotting survivorship bias panels...")
    for pt, outcome in [("FF", "mean_velo"), ("FF", "mean_spin_rate"),
                         ("SI", "mean_velo"), ("SL", "mean_spin_rate")]:
        plot_survivorship_bias(
            results_df, raw_df, age_mean,
            out_dir=PLOTS_DIR,
            pitch_type=pt, outcome=outcome,
            experiment="base",
        )

    print("\nPlotting extension AIC heatmap...")
    plot_ext_aic_heatmap(results_df, out_dir=PLOTS_DIR)

    print("\nPlotting delta method comparison...")
    plot_delta_method_comparison(
        results_df, raw_df, age_mean,
        out_dir=PLOTS_DIR,
        experiment="base",
    )

    print("\nDone — all plots saved to", PLOTS_DIR)


def plot_spaghetti(
    results_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    age_mean: float,
    out_dir: Path,
    experiment: str = "base",
    outcomes: list[str] | None = None,
    min_seasons: int = 5,
    n_sample: int = 40,
    seed: int = 42,
) -> None:
    """
    Spaghetti plot per outcome × pitch type.

    Shows:
      - Faint individual pitcher trajectories (raw observed data)
      - Per-pitcher model fit (random intercept + population curve shape)
      - Bold population-level mixed-effects curve
      - Highlighted 'case study' pitchers (top/bottom/median trajectory)

    Parameters
    ----------
    min_seasons : minimum seasons for a pitcher to be included
    n_sample    : number of pitchers to sample (keeps plot readable)
    seed        : random seed for reproducibility
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_df = results_df[results_df["experiment"] == experiment]
    sig_df = exp_df[exp_df["significant"]]

    if outcomes is None:
        outcomes = sig_df["outcome"].unique().tolist()

    ages, ages_c = _age_grid(age_mean)
    rng = np.random.default_rng(seed)

    for outcome in outcomes:
        pitch_rows  = sig_df[sig_df["outcome"] == outcome]
        pitch_types = pitch_rows["pitch_type"].tolist()
        if not pitch_types:
            continue

        n_pt = len(pitch_types)
        fig, axes = plt.subplots(
            n_pt, 1,
            figsize=(10, 3.5 * n_pt),
            sharex=False,
        )
        if n_pt == 1:
            axes = [axes]

        ylabel = OUTCOME_LABELS.get(outcome, outcome)
        fig.suptitle(
            f"Spaghetti Plot — {ylabel}  [{experiment}]  "
            f"(≥{min_seasons} seasons, n={n_sample} sampled)",
            fontsize=12, fontweight="bold", y=1.01,
        )

        for ax, pt in zip(axes, pitch_types):
            row   = pitch_rows[pitch_rows["pitch_type"] == pt].iloc[0]
            color = PITCH_COLORS.get(pt, "steelblue")

            pt_raw = (
                raw_df[raw_df["pitch_type"] == pt]
                .dropna(subset=[outcome, "age", "age_c"])
            )
            season_counts = pt_raw.groupby("pitcher")["age"].count()
            eligible      = season_counts[season_counts >= min_seasons].index.tolist()

            if not eligible:
                ax.set_title(f"{PITCH_LABELS.get(pt, pt)} — insufficient data", color=color)
                continue

            sampled = rng.choice(
                eligible,
                size=min(n_sample, len(eligible)),
                replace=False,
            ).tolist()

            pitcher_means = (
                pt_raw[pt_raw["pitcher"].isin(sampled)]
                .groupby("pitcher")[outcome].mean()
                .sort_values()
            )
            n_cs = min(3, len(pitcher_means))
            idx  = [0, len(pitcher_means) // 2, len(pitcher_means) - 1][:n_cs]
            case_study_ids = set(pitcher_means.iloc[idx].index.tolist())

            obs_mean  = pt_raw[outcome].mean()
            y_pop     = _shift_curve(row, ages_c, obs_mean)

            for pid in sampled:
                p_data = pt_raw[pt_raw["pitcher"] == pid].sort_values("age")
                if len(p_data) < 2:
                    continue

                is_case = pid in case_study_ids
                age_vals = p_data["age"].to_numpy(dtype=float)
                y_vals = p_data[outcome].to_numpy(dtype=float)
                unique_ages = np.unique(age_vals)

                ax.scatter(
                    age_vals, y_vals,
                    color=color,
                    alpha=0.5 if is_case else 0.08,
                    s=12 if is_case else 8,
                    zorder=4 if is_case else 2,
                )

                deg = 2 if len(unique_ages) >= 3 else 1
                coeffs = np.polyfit(age_vals, y_vals, deg=deg)
                age_dense = np.linspace(age_vals.min(), age_vals.max(), 80)
                y_dense = np.polyval(coeffs, age_dense)
                ax.plot(
                    age_dense, y_dense,
                    color=color,
                    alpha=0.65 if is_case else 0.12,
                    lw=1.8 if is_case else 0.9,
                    zorder=4 if is_case else 2,
                )

                pitcher_mean = p_data[outcome].mean()
                y_pitcher    = _shift_curve(row, ages_c, pitcher_mean)
                age_lo = p_data["age"].min()
                age_hi = p_data["age"].max()
                mask = (ages >= age_lo) & (ages <= age_hi)
                ax.plot(
                    ages[mask], y_pitcher[mask],
                    color=color,
                    alpha=0.5 if is_case else 0.05,
                    lw=1.5 if is_case else 0.6,
                    ls="--",
                    zorder=3 if is_case else 1,
                )

                if is_case:
                    name = p_data["player_name"].iloc[0] if "player_name" in p_data else str(pid)
                    last_age = p_data["age"].iloc[-1]
                    last_val = p_data[outcome].iloc[-1]
                    ax.annotate(
                        name.split(",")[0] if "," in name else name,
                        xy=(last_age, last_val),
                        xytext=(4, 0), textcoords="offset points",
                        fontsize=7, color=color, alpha=0.85,
                        va="center",
                    )

            ax.plot(
                ages, y_pop,
                color="black", lw=3, zorder=5,
                label="Population curve (mixed-effects)",
            )

            peak_age = row.get("peak_age")
            if pd.notna(peak_age) and 21 <= peak_age <= 38:
                pa_c  = peak_age - age_mean
                y_pk  = float(_predict_curve(row, np.array([pa_c]))[0]
                              - _predict_curve(row, ages_c).mean() + obs_mean)
                ax.axvline(peak_age, color="black", lw=1, ls=":", alpha=0.5)
                ax.annotate(
                    f"Peak {peak_age:.1f}",
                    xy=(peak_age, y_pk),
                    xytext=(5, 8), textcoords="offset points",
                    fontsize=8.5, color="black",
                )

            ax.set_title(
                f"{PITCH_LABELS.get(pt, pt)}  "
                f"(n={len(sampled)} pitchers, {len(eligible)} eligible)",
                fontsize=10, loc="left", fontweight="bold", color=color,
            )
            ax.set_xlabel("Age")
            ax.set_ylabel(ylabel, fontsize=9)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
            ax.legend(fontsize=8, frameon=False, loc="upper right")

            from matplotlib.lines import Line2D
            handles, labels = ax.get_legend_handles_labels()
            handles += [
                Line2D([0], [0], color=color, alpha=0.4, lw=0.8,
                      label=f"Individual aging curves (sampled, n={len(sampled)})"),
                Line2D([0], [0], color=color, alpha=0.4, lw=1.2, ls="--",
                       label="Per-pitcher model fit"),
            ]
            ax.legend(handles=handles, fontsize=7.5, frameon=False, loc="upper right")

        fig.tight_layout()
        fname = out_dir / f"spaghetti_{outcome}_{experiment}.png"
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname.name}")