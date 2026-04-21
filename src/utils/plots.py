"""Reusable plotting functions for the MLB pitch aging study."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from .utils import OUTCOME_LABELS, PITCH_COLORS, PITCH_LABELS

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)

MIN_N_NAIVE = 15  # minimum pitchers per age for naive means


def _predict_curve(row: pd.Series, age_grid_c: np.ndarray) -> np.ndarray:
    """Reconstruct population-level predicted values from stored coefficients."""
    b1 = row.get("b1_mean", row.get("b1", 0.0))
    b2 = row.get("b2_mean", row.get("b2"))
    intercept = row.get("intercept", 0.0)
    y = intercept + b1 * age_grid_c
    if not row.get("is_linear_model", False) and pd.notna(b2):
        y = y + b2 * age_grid_c**2
    return y


def _age_grid(age_mean: float, lo: int = 21, hi: int = 38, n: int = 200):
    ages = np.linspace(lo, hi, n)
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
        naive["age"],
        naive["mean"],
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
    ax,
    ages: np.ndarray,
    y_shifted: np.ndarray,
    color: str,
    row: pd.Series,
    age_mean: float,
) -> None:
    """Mixed-effects curve with optional peak annotation."""
    ax.plot(ages, y_shifted, color=color, lw=2.5, zorder=3, label="Mixed-effects")

    peak_age = row.get("peak_age_median", row.get("peak_age"))
    if pd.notna(peak_age) and 21 <= peak_age <= 38:
        pa_c = peak_age - age_mean
        y_pk = (
            _predict_curve(row, np.array([pa_c]))[0]
            - _predict_curve(row, ages - age_mean).mean()
            + y_shifted.mean()
        )
        ax.axvline(peak_age, color=color, lw=1, ls="--", alpha=0.6)
        ax.annotate(
            f"Peak {peak_age:.1f}",
            xy=(peak_age, y_pk),
            xytext=(4, 6),
            textcoords="offset points",
            fontsize=8,
            color=color,
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
        pitch_rows = sig_df[sig_df["outcome"] == outcome]
        pitch_types = pitch_rows["pitch_type"].tolist()
        if not pitch_types:
            continue

        n_pt = len(pitch_types)
        fig, axes = plt.subplots(
            n_pt,
            2,
            figsize=(10, 2.8 * n_pt),
            gridspec_kw={"width_ratios": [3, 1]},
        )
        if n_pt == 1:
            axes = [axes]

        ylabel = OUTCOME_LABELS.get(outcome, outcome)
        fig.suptitle(
            f"Aging Curves — {ylabel}  [{experiment}]",
            fontsize=13,
            fontweight="bold",
            y=1.01,
        )

        for ax_row, pt in zip(axes, pitch_types):
            ax_curve, ax_n = ax_row
            row = pitch_rows[pitch_rows["pitch_type"] == pt].iloc[0]
            color = PITCH_COLORS.get(pt, "steelblue")

            pt_raw = raw_df[raw_df["pitch_type"] == pt].dropna(subset=[outcome, "age"])
            naive = _naive_means(pt_raw, outcome)

            _plot_naive(ax_curve, naive)

            y_shifted = _shift_curve(row, ages_c, pt_raw[outcome].mean())
            _plot_mixed_curve(ax_curve, ages, y_shifted, color, row, age_mean)

            ax_curve.set_ylabel(ylabel, fontsize=9)
            ax_curve.set_xlabel("Age")
            ax_curve.set_title(
                PITCH_LABELS.get(pt, pt),
                fontsize=10,
                loc="left",
                fontweight="bold",
                color=color,
            )
            ax_curve.legend(fontsize=8, frameon=False)
            ax_curve.xaxis.set_major_locator(mticker.MultipleLocator(2))

            n_by_age = pt_raw.groupby("age").size().reset_index(name="n")
            ax_n.bar(n_by_age["age"], n_by_age["n"], color=color, alpha=0.6, width=0.8)
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

    n = len(pitch_types)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=False)
    if n == 1:
        axes = [axes]

    fig.suptitle(
        "Velocity vs Spin Rate Aging — Mid-Career Spin Peak",
        fontsize=12,
        fontweight="bold",
    )

    for ax, pt in zip(axes, pitch_types):
        color = PITCH_COLORS.get(pt, "steelblue")
        pt_raw = raw_df[raw_df["pitch_type"] == pt]
        ax2 = ax.twinx()

        for outcome, target_ax, ls, ylabel, label in [
            ("mean_velo", ax, "-", "Velocity (mph)", "Velocity"),
            ("mean_spin_rate", ax2, "--", "Spin Rate (rpm)", "Spin Rate"),
        ]:
            row_match = exp_df[
                (exp_df["pitch_type"] == pt) & (exp_df["outcome"] == outcome)
            ]
            if row_match.empty:
                continue
            row = row_match.iloc[0]

            obs_mean = pt_raw[outcome].dropna().mean()
            y_shifted = _shift_curve(row, ages_c, obs_mean)
            line_color = color if outcome == "mean_spin_rate" else "black"

            target_ax.plot(ages, y_shifted, color=line_color, lw=2, ls=ls, label=label)
            target_ax.set_ylabel(ylabel, color=line_color, fontsize=8)
            target_ax.tick_params(axis="y", labelcolor=line_color)

            if outcome == "mean_spin_rate":
                peak_age = row.get("peak_age_median", row.get("peak_age"))
                if pd.notna(peak_age) and 21 <= peak_age <= 38:
                    ax2.axvline(peak_age, color=color, lw=1, ls=":", alpha=0.8)
                    ax2.annotate(
                        f"Spin peak\n{peak_age:.1f}",
                        xy=(peak_age, y_shifted.max()),
                        xytext=(4, -20),
                        textcoords="offset points",
                        fontsize=7.5,
                        color=color,
                    )

        ax.set_xlabel("Age")
        ax.set_title(PITCH_LABELS.get(pt, pt), fontsize=10, fontweight="bold")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(2))

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            lines1 + lines2,
            labels1 + labels2,
            fontsize=8,
            frameon=False,
            loc="lower left",
        )

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

    exp_df = results_df[results_df["experiment"] == experiment]
    row_match = exp_df[
        (exp_df["pitch_type"] == pitch_type) & (exp_df["outcome"] == outcome)
    ]
    if row_match.empty:
        print(f"  No results for {pitch_type} {outcome} [{experiment}] — skipping")
        return

    row = row_match.iloc[0]
    color = PITCH_COLORS.get(pitch_type, "steelblue")
    pt_raw = raw_df[raw_df["pitch_type"] == pitch_type].dropna(subset=[outcome, "age"])
    naive = _naive_means(pt_raw, outcome)
    ages, ages_c = _age_grid(age_mean)
    y_shifted = _shift_curve(row, ages_c, pt_raw[outcome].mean())

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(
        f"Survivorship Bias — {PITCH_LABELS.get(pitch_type, pitch_type)} "
        f"{OUTCOME_LABELS.get(outcome, outcome)}",
        fontsize=12,
        fontweight="bold",
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
            fontsize=8,
            color="gray",
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


def plot_ext_loo_heatmap(
    results_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Heatmap of PSIS-LOO delta (with_ext − base) per pitch type × outcome.
    Positive = extension improves fit.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric = "loo"
    metric_label = "ΔLOO"
    base = results_df[results_df["experiment"] == "base"].set_index(
        ["pitch_type", "outcome"]
    )[metric]
    ext = results_df[results_df["experiment"] == "with_ext"].set_index(
        ["pitch_type", "outcome"]
    )[metric]
    delta = (ext - base).unstack("outcome")

    pt_order = ["FF", "SL", "SI", "CH", "CU", "FC"]
    oc_order = [
        o
        for o in [
            "mean_velo",
            "mean_spin_rate",
            "mean_pfx_z",
            "mean_pfx_x_norm",
            "mean_pfx_x",
            "mean_spin_axis",
        ]
        if o in delta.columns
    ]

    delta = delta.reindex(index=pt_order, columns=oc_order)
    col_labels = [OUTCOME_LABELS.get(c, c) for c in delta.columns]
    row_labels = [PITCH_LABELS.get(r, r) for r in delta.index]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.heatmap(
        delta,
        ax=ax,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn_r",
        center=0,
        linewidths=0.5,
        linecolor="white",
        xticklabels=col_labels,
        yticklabels=row_labels,
        cbar_kws={"label": f"{metric_label} (with_ext − base)", "shrink": 0.8},
    )
    ax.set_title(
        f"Extension Covariate — {metric_label} Improvement over Base Model",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    fname = out_dir / "ext_loo_heatmap.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname.name}")


def plot_pareto_k_heatmap(
    results_df: pd.DataFrame,
    out_dir: Path,
    experiment: str = "base",
    threshold: float = 0.7,
) -> None:
    """
    Heatmap of % observations with Pareto k > threshold per pitch type × outcome.
    Highlights models where PSIS-LOO importance sampling is unreliable.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = results_df[results_df["experiment"] == experiment].copy()
    if "pct_high_pareto_k" not in df.columns or df["pct_high_pareto_k"].isna().all():
        print(f"  Skipping Pareto k heatmap — no pct_high_pareto_k data for {experiment}")
        return

    pivot = df.pivot_table(
        index="pitch_type", columns="outcome", values="pct_high_pareto_k"
    ) * 100  # convert to percent

    pt_order = ["FF", "SL", "SI", "CH", "CU", "FC"]
    oc_order = [
        o
        for o in [
            "mean_velo",
            "mean_spin_rate",
            "mean_pfx_z",
            "mean_pfx_x_norm",
            "mean_pfx_x",
            "mean_spin_axis",
        ]
        if o in pivot.columns
    ]
    pivot = pivot.reindex(index=pt_order, columns=oc_order)
    col_labels = [OUTCOME_LABELS.get(c, c) for c in pivot.columns]
    row_labels = [PITCH_LABELS.get(r, r) for r in pivot.index]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        vmin=0,
        linewidths=0.5,
        linecolor="white",
        xticklabels=col_labels,
        yticklabels=row_labels,
        cbar_kws={"label": f"% obs with Pareto k > {threshold}", "shrink": 0.8},
    )
    ax.set_title(
        f"PSIS-LOO Reliability — % Observations with Pareto k > {threshold} ({experiment})",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    fname = out_dir / f"pareto_k_heatmap_{experiment}.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname.name}")


def plot_peak_age_ci(
    ci_df: pd.DataFrame,
    out_dir: Path,
    outcome: str = "mean_spin_rate",
    experiment: str = "base",
) -> None:
    """Forest plot of peak ages with 95% confidence intervals."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_df = ci_df.copy()
    if "outcome" in plot_df.columns:
        plot_df = plot_df[plot_df["outcome"] == outcome]

    plot_df = plot_df.dropna(subset=["peak_age", "ci_lo", "ci_hi"]).copy()
    if plot_df.empty:
        print(f"  No CI rows for outcome={outcome}; skipping peak-age forest plot")
        return

    # Sort by estimated peak age so the ordering is visually interpretable.
    plot_df = plot_df.sort_values("peak_age").reset_index(drop=True)
    labels = [PITCH_LABELS.get(pt, pt) for pt in plot_df["pitch_type"]]
    y = np.arange(len(plot_df))

    x = plot_df["peak_age"].to_numpy(dtype=float)
    xerr = np.vstack(
        [
            x - plot_df["ci_lo"].to_numpy(dtype=float),
            plot_df["ci_hi"].to_numpy(dtype=float) - x,
        ]
    )

    fig_h = max(3.5, 0.7 * len(plot_df) + 1.8)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))

    ax.errorbar(
        x,
        y,
        xerr=xerr,
        fmt="o",
        color="#1f2937",
        ecolor="#6b7280",
        elinewidth=2,
        capsize=4,
        markersize=6,
        zorder=3,
    )

    # Add lightweight, per-pitch coloring accents for readability.
    for i, row in plot_df.iterrows():
        color = PITCH_COLORS.get(row["pitch_type"], "#1f77b4")
        ax.plot(row["peak_age"], i, "o", color=color, markersize=7, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Estimated peak age (years)")
    ax.set_ylabel("Pitch type")
    outcome_label = OUTCOME_LABELS.get(outcome, outcome)
    ax.set_title(
        f"Peak age estimates with 95% CIs — {outcome_label}",
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)

    x_lo = float(plot_df["ci_lo"].min())
    x_hi = float(plot_df["ci_hi"].max())
    pad = max(0.5, 0.08 * (x_hi - x_lo))
    ax.set_xlim(x_lo - pad, x_hi + pad)

    for i, row in plot_df.iterrows():
        ax.text(
            row["ci_hi"] + 0.05,
            i,
            f"{row['peak_age']:.1f} [{row['ci_lo']:.1f}, {row['ci_hi']:.1f}]",
            va="center",
            fontsize=8,
            color="#374151",
        )

    fig.tight_layout()
    fname = out_dir / f"peak_age_ci_{outcome}_{experiment}.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname.name}")


def plot_decline_rate_ci(
    decline_df: pd.DataFrame,
    out_dir: Path,
    outcome: str,
    eval_age: float = 28,
    experiment: str = "base",
) -> None:
    """Forest plot of decline rates at a given age with 95% confidence intervals."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_df = decline_df[
        (decline_df["outcome"] == outcome) & (decline_df["eval_age"] == eval_age)
    ].copy()
    plot_df = plot_df.dropna(subset=["rate", "ci_lo", "ci_hi"])
    if plot_df.empty:
        print(
            f"  No decline-rate rows for outcome={outcome}, age={eval_age}; "
            "skipping forest plot"
        )
        return

    plot_df = plot_df.sort_values("rate").reset_index(drop=True)
    labels = [PITCH_LABELS.get(pt, pt) for pt in plot_df["pitch_type"]]
    y = np.arange(len(plot_df))

    x = plot_df["rate"].to_numpy(dtype=float)
    xerr = np.vstack(
        [
            x - plot_df["ci_lo"].to_numpy(dtype=float),
            plot_df["ci_hi"].to_numpy(dtype=float) - x,
        ]
    )

    fig_h = max(3.5, 0.7 * len(plot_df) + 1.8)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))

    ax.errorbar(
        x,
        y,
        xerr=xerr,
        fmt="o",
        color="#1f2937",
        ecolor="#6b7280",
        elinewidth=2,
        capsize=4,
        markersize=6,
        zorder=3,
    )

    for i, row in plot_df.iterrows():
        color = PITCH_COLORS.get(row["pitch_type"], "#1f77b4")
        ax.plot(row["rate"], i, "o", color=color, markersize=7, zorder=4)

    ax.axvline(0.0, color="#9ca3af", lw=1.2, ls="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Decline rate per year")
    ax.set_ylabel("Pitch type")
    outcome_label = OUTCOME_LABELS.get(outcome, outcome)
    ax.set_title(
        f"Decline rates at age {eval_age:.0f} with 95% CIs — {outcome_label}",
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)

    x_lo = float(plot_df["ci_lo"].min())
    x_hi = float(plot_df["ci_hi"].max())
    pad = max(0.05, 0.08 * (x_hi - x_lo))
    ax.set_xlim(x_lo - pad, x_hi + pad)

    for i, row in plot_df.iterrows():
        ax.text(
            row["ci_hi"] + 0.01 * max(1.0, x_hi - x_lo),
            i,
            f"{row['rate']:.3f} [{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]",
            va="center",
            fontsize=8,
            color="#374151",
        )

    fig.tight_layout()
    fname = out_dir / f"decline_rate_ci_{outcome}_{experiment}.png"
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
            n_pt,
            1,
            figsize=(10, 3.2 * n_pt),
        )
        if n_pt == 1:
            axes = [axes]

        ylabel = OUTCOME_LABELS.get(outcome, outcome)
        fig.suptitle(
            f"Delta Method vs Mixed-Effects — {ylabel}  [{experiment}]",
            fontsize=13,
            fontweight="bold",
            y=1.01,
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
                delta["age"],
                delta["mean"],
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
                ages,
                y_shifted,
                color=color,
                lw=2.5,
                zorder=3,
                label="Mixed-effects",
            )

            peak_age = row.get("peak_age_median", row.get("peak_age"))
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
                    xytext=(4, 6),
                    textcoords="offset points",
                    fontsize=8,
                    color=color,
                )

            ax.axvspan(
                33, 38, alpha=0.05, color="red", label="High survivorship bias region"
            )

            decline = row.get("decline_at_mean", row.get("decline_rate_at_mean"))
            if pd.notna(decline):
                sign = "+" if decline > 0 else ""
                ax.annotate(
                    f"Mixed-effects decline: {sign}{decline:.3f}/yr at age {age_mean:.0f}",
                    xy=(0.02, 0.05),
                    xycoords="axes fraction",
                    fontsize=7.5,
                    color=color,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
                )

            ax.set_title(
                PITCH_LABELS.get(pt, pt),
                fontsize=10,
                loc="left",
                fontweight="bold",
                color=color,
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




def plot_bivariate_correlation(
    results: dict,
    out_dir: Path,
) -> None:
    """
    Posterior histogram of velo/spin random effect correlation for each pitch type.
    One panel per pitch type with mean, HDI shading, and ρ=0 reference.
    """
    import arviz as az

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pitch_types = list(results.keys())
    fig, axes = plt.subplots(1, len(pitch_types), figsize=(5.5 * len(pitch_types), 4))
    if len(pitch_types) == 1:
        axes = [axes]

    fig.suptitle(
        "Bivariate Velo/Spin Random Effect Correlation",
        fontsize=12,
        fontweight="bold",
    )

    for ax, pt in zip(axes, pitch_types):
        idata = results[pt]
        color = PITCH_COLORS.get(pt, "steelblue")
        rho = idata.posterior["chol_corr"].values[:, :, 0, 1].flatten()
        hdi = az.hdi(rho, hdi_prob=0.95)

        ax.hist(rho, bins=60, color=color, alpha=0.65, density=True, zorder=2)
        ax.axvspan(
            hdi[0],
            hdi[1],
            alpha=0.18,
            color=color,
            zorder=1,
            label=f"95% HDI [{hdi[0]:.3f}, {hdi[1]:.3f}]",
        )
        ax.axvline(
            rho.mean(), color=color, lw=2, zorder=3, label=f"mean = {rho.mean():.3f}"
        )
        ax.axvline(
            0,
            color="#374151",
            lw=1.2,
            ls="--",
            alpha=0.6,
            zorder=3,
            label="ρ = 0 (independence)",
        )

        ax.set_xlabel("Correlation (ρ)")
        ax.set_ylabel("Density")
        ax.set_title(
            PITCH_LABELS.get(pt, pt), fontsize=10, fontweight="bold", color=color
        )
        ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    fname = out_dir / "bivariate_correlation.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname.name}")


def plot_bivariate_peak_comparison(
    results: dict,
    univariate_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Side-by-side comparison of univariate vs bivariate peak age estimates.
    Bivariate posteriors shown as HDI bars; univariate as point + CI from delta method.
    """
    import arviz as az

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pitch_types = list(results.keys())
    outcomes = ["mean_velo", "mean_spin_rate"]
    uni_base = univariate_df[univariate_df["experiment"] == "base"]

    fig, axes = plt.subplots(
        len(outcomes),
        len(pitch_types),
        figsize=(5 * len(pitch_types), 3.5 * len(outcomes)),
        sharey=False,
    )
    if len(pitch_types) == 1:
        axes = [[ax] for ax in axes]

    for row_i, outcome in enumerate(outcomes):
        var_name = "peak_age_velo" if outcome == "mean_velo" else "peak_age_spin"
        ylabel = OUTCOME_LABELS.get(outcome, outcome)

        for col_i, pt in enumerate(pitch_types):
            ax = axes[row_i][col_i]
            idata = results[pt]
            color = PITCH_COLORS.get(pt, "steelblue")

            # Bivariate posterior
            samples = idata.posterior[var_name].values.flatten()
            hdi = az.hdi(samples, hdi_prob=0.95)
            biv_mean = samples.mean()

            ax.hist(
                samples,
                bins=60,
                color=color,
                alpha=0.55,
                density=True,
                zorder=2,
                label=f"Bivariate  {biv_mean:.1f} [{hdi[0]:.1f}, {hdi[1]:.1f}]",
            )

            # Univariate point estimate — support both old and new schema column names
            uni_row = uni_base[
                (uni_base["pitch_type"] == pt) & (uni_base["outcome"] == outcome)
            ]
            if not uni_row.empty:
                uni_peak_val = uni_row.iloc[0].get(
                    "peak_age_median", uni_row.iloc[0].get("peak_age")
                )
                if pd.notna(uni_peak_val):
                    ax.axvline(
                        uni_peak_val,
                        color="#374151",
                        lw=2,
                        ls="--",
                        zorder=3,
                        label=f"Univariate  {uni_peak_val:.1f}",
                    )

            ax.axvline(biv_mean, color=color, lw=1.8, zorder=3)
            xlim = (25, 45) if outcome == "mean_spin_rate" else (20, 40)
            ax.set_xlim(*xlim)
            ax.set_xlabel("Peak age (years)")
            ax.set_ylabel("Density" if col_i == 0 else "")
            ax.set_title(
                f"{PITCH_LABELS.get(pt, pt)} — {ylabel}",
                fontsize=9,
                fontweight="bold",
                color=color,
            )
            ax.legend(fontsize=7.5, frameon=False)

    fig.suptitle(
        "Univariate vs Bivariate Peak Age Estimates",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fname = out_dir / "bivariate_peak_comparison.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname.name}")


def plot_scg_comparison(
    uni_scg_df: pd.DataFrame,
    biv_scg_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Grouped horizontal bar chart comparing univariate vs bivariate SCG per pitch type.
    Bivariate bars include posterior HDI whiskers.
    """
    from matplotlib.lines import Line2D

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    biv = biv_scg_df.set_index("pitch_type")
    uni = uni_scg_df.set_index("pitch_type")
    pitch_types = biv.index.tolist()

    # Sort by bivariate SCG ascending so largest gap is at top
    pitch_types = sorted(pitch_types, key=lambda pt: biv.loc[pt, "scg"])

    n = len(pitch_types)
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.9 * n + 1.8)))

    offset = 0.22
    for i, pt in enumerate(pitch_types):
        color = PITCH_COLORS.get(pt, "steelblue")

        # Bivariate bar (bottom slot)
        biv_scg = float(biv.loc[pt, "scg"])
        biv_hdi_lo = float(biv.loc[pt, "scg_hdi_lo"])
        biv_hdi_hi = float(biv.loc[pt, "scg_hdi_hi"])
        bar_color = "#22c55e" if biv_scg >= 0 else "#ef4444"

        ax.barh(i - offset, biv_scg, color=color, alpha=0.80, height=0.35, zorder=2)
        ax.errorbar(
            biv_scg, i - offset,
            xerr=[[biv_scg - biv_hdi_lo], [biv_hdi_hi - biv_scg]],
            fmt="none", color=color, elinewidth=1.8, capsize=4, zorder=3,
        )
        sign = "+" if biv_scg >= 0 else ""
        # Place text below the bivariate bar, anchored right of the HDI upper bound
        ax.text(
            biv_hdi_hi + 0.25,
            i - offset - 0.18,
            f"{sign}{biv_scg:.1f} ★  [{biv_hdi_lo:.1f}, {biv_hdi_hi:.1f}]",
            va="top", ha="left",
            fontsize=8, color=color,
        )

        # Univariate bar (top slot)
        if pt in uni.index:
            uni_scg = float(uni.loc[pt, "scg"])
            ax.barh(i + offset, uni_scg, color=color, alpha=0.35, height=0.35, zorder=2)
            sign_u = "+" if uni_scg >= 0 else ""
            ax.text(
                max(uni_scg, 0) + 0.25 if uni_scg >= 0 else min(uni_scg, 0) - 0.25,
                i + offset,
                f"{sign_u}{uni_scg:.1f}",
                va="center", ha="left" if uni_scg >= 0 else "right",
                fontsize=8, color=color, alpha=0.7,
            )

    ax.axvline(0, color="#374151", lw=1.2, alpha=0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([PITCH_LABELS.get(pt, pt) for pt in pitch_types])
    ax.set_xlabel("Stuff Compensation Gap (spin peak age − velocity peak age, years)")
    ax.set_title(
        "Stuff Compensation Gap: Univariate vs Bivariate Estimates",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.grid(axis="x", alpha=0.2, zorder=0)
    ax.set_axisbelow(True)

    all_scg = [float(biv.loc[pt, "scg"]) for pt in pitch_types]
    if any(pt in uni.index for pt in pitch_types):
        all_scg += [float(uni.loc[pt, "scg"]) for pt in pitch_types if pt in uni.index]
    x_min = min(min(all_scg) - 1.5, -1)
    x_max = max(all_scg) + 5
    ax.set_xlim(x_min, x_max)

    legend_handles = [
        Line2D([0], [0], color="gray", lw=6, alpha=0.80, label="Bivariate (★ = posterior mean, whiskers = 95% HDI)"),
        Line2D([0], [0], color="gray", lw=6, alpha=0.35, label="Univariate"),
    ]
    ax.legend(handles=legend_handles, fontsize=8.5, frameon=False, loc="upper right")

    fig.tight_layout()
    fname = out_dir / "scg_comparison.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname.name}")


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
        pitch_rows = sig_df[sig_df["outcome"] == outcome]
        pitch_types = pitch_rows["pitch_type"].tolist()
        if not pitch_types:
            continue

        n_pt = len(pitch_types)
        fig, axes = plt.subplots(
            n_pt,
            1,
            figsize=(10, 3.5 * n_pt),
            sharex=False,
        )
        if n_pt == 1:
            axes = [axes]

        ylabel = OUTCOME_LABELS.get(outcome, outcome)
        fig.suptitle(
            f"Spaghetti Plot — {ylabel}  [{experiment}]  "
            f"(≥{min_seasons} seasons, n={n_sample} sampled)",
            fontsize=12,
            fontweight="bold",
            y=1.01,
        )

        for ax, pt in zip(axes, pitch_types):
            row = pitch_rows[pitch_rows["pitch_type"] == pt].iloc[0]
            color = PITCH_COLORS.get(pt, "steelblue")

            pt_raw = raw_df[raw_df["pitch_type"] == pt].dropna(
                subset=[outcome, "age", "age_c"]
            )
            season_counts = pt_raw.groupby("pitcher")["age"].count()
            eligible = season_counts[season_counts >= min_seasons].index.tolist()

            if not eligible:
                ax.set_title(
                    f"{PITCH_LABELS.get(pt, pt)} — insufficient data", color=color
                )
                continue

            sampled = rng.choice(
                eligible,
                size=min(n_sample, len(eligible)),
                replace=False,
            ).tolist()

            pitcher_means = (
                pt_raw[pt_raw["pitcher"].isin(sampled)]
                .groupby("pitcher")[outcome]
                .mean()
                .sort_values()
            )
            n_cs = min(3, len(pitcher_means))
            idx = [0, len(pitcher_means) // 2, len(pitcher_means) - 1][:n_cs]
            case_study_ids = set(pitcher_means.iloc[idx].index.tolist())

            obs_mean = pt_raw[outcome].mean()
            y_pop = _shift_curve(row, ages_c, obs_mean)

            for pid in sampled:
                p_data = pt_raw[pt_raw["pitcher"] == pid].sort_values("age")
                if len(p_data) < 2:
                    continue

                is_case = pid in case_study_ids
                age_vals = p_data["age"].to_numpy(dtype=float)
                y_vals = p_data[outcome].to_numpy(dtype=float)
                unique_ages = np.unique(age_vals)

                ax.scatter(
                    age_vals,
                    y_vals,
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
                    age_dense,
                    y_dense,
                    color=color,
                    alpha=0.65 if is_case else 0.12,
                    lw=1.8 if is_case else 0.9,
                    zorder=4 if is_case else 2,
                )

                pitcher_mean = p_data[outcome].mean()
                y_pitcher = _shift_curve(row, ages_c, pitcher_mean)
                age_lo = p_data["age"].min()
                age_hi = p_data["age"].max()
                mask = (ages >= age_lo) & (ages <= age_hi)
                ax.plot(
                    ages[mask],
                    y_pitcher[mask],
                    color=color,
                    alpha=0.5 if is_case else 0.05,
                    lw=1.5 if is_case else 0.6,
                    ls="--",
                    zorder=3 if is_case else 1,
                )

                if is_case:
                    name = (
                        p_data["player_name"].iloc[0]
                        if "player_name" in p_data
                        else str(pid)
                    )
                    last_age = p_data["age"].iloc[-1]
                    last_val = p_data[outcome].iloc[-1]
                    ax.annotate(
                        name.split(",")[0] if "," in name else name,
                        xy=(last_age, last_val),
                        xytext=(4, 0),
                        textcoords="offset points",
                        fontsize=7,
                        color=color,
                        alpha=0.85,
                        va="center",
                    )

            ax.plot(
                ages,
                y_pop,
                color="black",
                lw=3,
                zorder=5,
                label="Population curve (mixed-effects)",
            )

            peak_age = row.get("peak_age_median", row.get("peak_age"))
            if pd.notna(peak_age) and 21 <= peak_age <= 38:
                pa_c = peak_age - age_mean
                y_pk = float(
                    _predict_curve(row, np.array([pa_c]))[0]
                    - _predict_curve(row, ages_c).mean()
                    + obs_mean
                )
                ax.axvline(peak_age, color="black", lw=1, ls=":", alpha=0.5)
                ax.annotate(
                    f"Peak {peak_age:.1f}",
                    xy=(peak_age, y_pk),
                    xytext=(5, 8),
                    textcoords="offset points",
                    fontsize=8.5,
                    color="black",
                )

            ax.set_title(
                f"{PITCH_LABELS.get(pt, pt)}  "
                f"(n={len(sampled)} pitchers, {len(eligible)} eligible)",
                fontsize=10,
                loc="left",
                fontweight="bold",
                color=color,
            )
            ax.set_xlabel("Age")
            ax.set_ylabel(ylabel, fontsize=9)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
            ax.legend(fontsize=8, frameon=False, loc="upper right")

            from matplotlib.lines import Line2D

            handles, labels = ax.get_legend_handles_labels()
            handles += [
                Line2D(
                    [0],
                    [0],
                    color=color,
                    alpha=0.4,
                    lw=0.8,
                    label=f"Individual aging curves (sampled, n={len(sampled)})",
                ),
                Line2D(
                    [0],
                    [0],
                    color=color,
                    alpha=0.4,
                    lw=1.2,
                    ls="--",
                    label="Per-pitcher model fit",
                ),
            ]
            ax.legend(handles=handles, fontsize=7.5, frameon=False, loc="upper right")

        fig.tight_layout()
        fname = out_dir / f"spaghetti_{outcome}_{experiment}.png"
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname.name}")


def plot_scg_dumbbell(
    scg_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Dumbbell plot: velo peak age (orange) vs spin peak age (blue) per pitch type.
    The gap between them is the Stuff Compensation Gap. HDI shown as thin bars.
    Pitch types sorted by SCG descending.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = scg_df.sort_values("scg", ascending=True).reset_index(drop=True)
    n = len(df)

    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.7 * n + 1.5)))

    VELO_COLOR = "#e07b39"
    SPIN_COLOR = "#3b82f6"
    GAP_COLOR_POS = "#22c55e"
    GAP_COLOR_NEG = "#ef4444"

    for i, row in df.iterrows():
        pt = row["pitch_type"]
        vp = row["velo_peak_age"]
        sp = row["spin_peak_age"]
        scg = row["scg"]

        gap_color = GAP_COLOR_POS if scg >= 0 else GAP_COLOR_NEG

        # Connecting line (compensation window)
        ax.plot(
            [min(vp, sp), max(vp, sp)],
            [i, i],
            color=gap_color,
            lw=4,
            alpha=0.35,
            solid_capstyle="round",
            zorder=1,
        )

        # HDI bars
        if pd.notna(row.get("velo_hdi_lo")) and pd.notna(row.get("velo_hdi_hi")):
            ax.plot(
                [row["velo_hdi_lo"], row["velo_hdi_hi"]],
                [i, i],
                color=VELO_COLOR,
                lw=1.5,
                alpha=0.5,
                zorder=2,
            )
        if pd.notna(row.get("spin_hdi_lo")) and pd.notna(row.get("spin_hdi_hi")):
            ax.plot(
                [row["spin_hdi_lo"], row["spin_hdi_hi"]],
                [i, i],
                color=SPIN_COLOR,
                lw=1.5,
                alpha=0.5,
                zorder=2,
            )

        # Peak dots
        ax.scatter(vp, i, color=VELO_COLOR, s=90, zorder=4, linewidths=0)
        ax.scatter(sp, i, color=SPIN_COLOR, s=90, zorder=4, linewidths=0)

        # SCG label
        sign = "+" if scg >= 0 else ""
        ax.text(
            max(vp, sp) + 0.4,
            i,
            f"{sign}{scg:.1f} yr",
            va="center",
            fontsize=9,
            color=gap_color,
            fontweight="bold",
        )

    ax.set_yticks(range(n))
    ax.set_yticklabels([PITCH_LABELS.get(row["pitch_type"], row["pitch_type"]) for _, row in df.iterrows()])
    ax.set_xlabel("Age (years)")
    ax.set_xlim(14, 42)
    ax.set_title(
        "Stuff Compensation Gap — Spin Peak Age vs Velocity Peak Age",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.grid(axis="x", alpha=0.2, zorder=0)
    ax.set_axisbelow(True)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=VELO_COLOR, markersize=9, label="Velocity peak age"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=SPIN_COLOR, markersize=9, label="Spin rate peak age"),
        Line2D([0], [0], color=GAP_COLOR_POS, lw=4, alpha=0.5, label="Compensation window (SCG > 0)"),
        Line2D([0], [0], color=GAP_COLOR_NEG, lw=4, alpha=0.5, label="No compensation (SCG < 0)"),
    ]
    ax.legend(handles=legend_handles, fontsize=8.5, frameon=False, loc="lower right")

    fig.tight_layout()
    fname = out_dir / "scg_dumbbell.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname.name}")


def plot_scg_bars(
    scg_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Horizontal bar chart of SCG per pitch type, sorted descending.
    Green = spin compensates velo decline; red = no compensation.
    Annotated with exact SCG and peak age breakdown.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = scg_df.sort_values("scg", ascending=True).reset_index(drop=True)
    n = len(df)

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.65 * n + 1.5)))

    for i, row in df.iterrows():
        scg = row["scg"]
        color = "#22c55e" if scg >= 0 else "#ef4444"
        ax.barh(i, scg, color=color, alpha=0.75, height=0.55)

        sign = "+" if scg >= 0 else ""
        label = (
            f"{sign}{scg:.1f} yr  "
            f"(velo peak {row['velo_peak_age']:.1f}, spin peak {row['spin_peak_age']:.1f})"
        )
        x_pos = max(scg, 0) + 0.15
        ha = "left"
        ax.text(x_pos, i, label, va="center", fontsize=8.5, color="#1f2937", ha=ha)

    ax.axvline(0, color="#374151", lw=1.2, ls="-", alpha=0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([PITCH_LABELS.get(row["pitch_type"], row["pitch_type"]) for _, row in df.iterrows()])
    ax.set_xlabel("Stuff Compensation Gap (years)")
    ax.set_title(
        "Stuff Compensation Gap by Pitch Type\n"
        "SCG = spin peak age − velocity peak age",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.grid(axis="x", alpha=0.2, zorder=0)
    ax.set_axisbelow(True)

    x_min = min(df["scg"].min() - 1.5, -1)
    x_max = df["scg"].max() + 4
    ax.set_xlim(x_min, x_max)

    fig.tight_layout()
    fname = out_dir / "scg_bars.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname.name}")
