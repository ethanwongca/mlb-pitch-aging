"""
EDA for MLB Pitch Aging (Rest of EDA is in notebooks/EDA.ipynb since but these plots are repetitive it's better off here)

Generate two sets of plots for each metric (velocity, spin rate, horizontal/vertical break, extension, and effective speed)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "master_data"
PLOTS_DIR = BASE_DIR / "plots"

VALID_PITCH_TYPES = ["FF", "SL", "SI", "CH", "CU", "FC"]

METRICS = [
    ("mean_velo", "std_velo", "Velocity (mph)", "Mean Velocity"),
    ("mean_spin_rate", "std_spin_rate", "Spin Rate (rpm)", "Mean Spin Rate"),
    ("mean_pfx_x", "std_pfx_x", "Horizontal Break (in)", "Mean H-Break"),
    ("mean_pfx_z", "std_pfx_z", "Vertical Break (in)", "Mean V-Break"),
    ("mean_ext", "std_ext", "Extension (ft)", "Mean Extension"),
    ("mean_eff_speed", "std_eff_speed", "Eff. Speed (mph)", "Mean Eff. Speed"),
]

def save_plot(title, suffix):
    filename = f"{title.lower().replace(' ', '_')}_{suffix}.png"
    plt.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    print(f"Saved {filename}")

def plot_by_age(df, metric, std_col, ylabel, title):
    _, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, pt in zip(axes.flatten(), VALID_PITCH_TYPES):
        agg = (
            df[df["pitch_type"] == pt]
            .groupby("age")
            .agg(mean=(metric, "mean"), std=(std_col, "mean"), n=("pitcher", "count"))
            .reset_index()
            .query("n >= 10")
        )
        ax.plot(agg["age"], agg["mean"], marker="o", linewidth=2)
        ax.fill_between(agg["age"], agg["mean"] - agg["std"], agg["mean"] + agg["std"], alpha=0.2)
        ax.set_title(pt)
        ax.set_xlabel("Age")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"{title} by Age", fontsize=13)
    plt.tight_layout()
    save_plot(title, "by_age")
    plt.show()

def plot_by_year(df, metric, ylabel, title):
    _, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, pt in zip(axes.flatten(), VALID_PITCH_TYPES):
        agg = (
            df[df["pitch_type"] == pt]
            .groupby("year")[metric]
            .mean()
            .reset_index()
        )
        ax.plot(agg["year"], agg[metric], marker="o", linewidth=2)
        ax.axvline(x=2021, color="red", linestyle="--", alpha=0.7, label="2021 crackdown")
        ax.set_title(pt)
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"{title} by Year", fontsize=13)
    plt.tight_layout()
    save_plot(title, "by_year")
    plt.show()

if __name__ == "__main__":
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    df = pd.read_csv(DATA_DIR / "pitching_master.csv")
    df = df[df["pitch_type"].isin(VALID_PITCH_TYPES)]

    for metric, std_col, ylabel, title in METRICS:
        plot_by_age(df, metric, std_col, ylabel, title)
        plot_by_year(df, metric, ylabel, title)

    if "mean_spin_axis" in df.columns:
        spin_df = df[df["year"] >= 2020]
        plot_by_age(spin_df, "mean_spin_axis", "std_spin_axis", "Spin Axis (deg)", "Spin Axis")
        plot_by_year(spin_df, "mean_spin_axis", "Spin Axis (deg)", "Spin Axis")