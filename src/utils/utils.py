import logging
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "master_data" / "pitching_master.csv"

DEFAULT_OUTCOMES = [
    "mean_velo",
    "mean_spin_rate",
    "mean_pfx_x",
    "mean_pfx_x_norm",
    "mean_pfx_z",
]

PITCH_COLORS = {
    "FF": "#E63946",
    "SL": "#457B9D",
    "SI": "#F4A261",
    "CH": "#2A9D8F",
    "CU": "#9B5DE5",
    "FC": "#F72585",
}
PITCH_ORDER = ["FF", "SL", "SI", "CH", "CU", "FC"]
PITCH_LABELS = {
    "FF": "4-Seam FB",
    "SL": "Slider",
    "SI": "Sinker",
    "CH": "Changeup",
    "CU": "Curveball",
    "FC": "Cutter",
}
OUTCOME_LABELS = {
    "mean_velo": "Velocity (mph)",
    "mean_spin_rate": "Spin Rate (rpm)",
    "mean_pfx_x": "Horizontal Break (ft)",
    "mean_pfx_x_norm": "Horizontal Break, norm (ft)",
    "mean_pfx_z": "Vertical Break (ft)",
}


def setup_logger(name: str, log_file: Path) -> logging.Logger:
    """Create a console+file logger with a consistent format."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def load_data(data: Path = DATA_PATH) -> pd.DataFrame:
    """Load the prepared master dataset and drop effective-speed columns."""
    return pd.read_csv(data).drop(
        columns=["mean_eff_speed", "std_eff_speed"], errors="ignore"
    )


def get_data_pitch_type_dict(
    df: pd.DataFrame, pitch_types: list[str]
) -> dict[str, pd.DataFrame]:
    """Return one dataframe per pitch type."""
    return {
        pitch_type: df[df["pitch_type"] == pitch_type] for pitch_type in pitch_types
    }


def get_valid_pitch_types() -> list[str]:
    """Return pitch types that have enough data to be analyzed."""
    return ["FF", "SL", "SI", "CH", "CU", "FC"]


def get_default_outcomes() -> list[str]:
    """Return outcomes to evaluate across pitch types."""
    return DEFAULT_OUTCOMES.copy()


def get_age_mean(df: pd.DataFrame) -> float:
    """Calculate mean age for centering in mixed models."""
    return df["age"].mean()


def ensure_mean_pfx_x_norm(df: pd.DataFrame) -> pd.DataFrame:
    """Create normalized horizontal break if required columns are present."""
    if "mean_pfx_x_norm" in df.columns:
        return df
    if {"mean_pfx_x", "p_throws"}.issubset(df.columns):
        df = df.copy()
        df["mean_pfx_x_norm"] = df["mean_pfx_x"].where(
            df["p_throws"] != "L", -df["mean_pfx_x"]
        )
    return df


def filter_pitchers_by_min_distinct_seasons(
    data: pd.DataFrame,
    min_seasons: int,
    pitcher_col: str = "pitcher",
    season_col: str = "year",
) -> pd.DataFrame:
    """Keep only pitchers with at least min_seasons distinct season values."""
    if data.empty:
        return data
    season_counts = data.groupby(pitcher_col)[season_col].nunique(dropna=True)
    keep_pitchers = season_counts[season_counts >= min_seasons].index
    return data[data[pitcher_col].isin(keep_pitchers)].copy()
