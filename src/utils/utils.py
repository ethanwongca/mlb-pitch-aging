import logging
from pathlib import Path

import pandas as pd

# Path variables used across scripts.
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "master_data" / "pitching_master.csv"

DEFAULT_OUTCOMES = [
    "mean_velo",
    "mean_spin_rate",
    "mean_pfx_x",
    "mean_pfx_x_norm",
    "mean_pfx_z",
    "mean_spin_axis",
]


def setup_logger(name: str, log_file: Path) -> logging.Logger:
    """Create a console+file logger with a consistent format."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def load_data(data: Path = DATA_PATH) -> pd.DataFrame:
    """Load the prepared master dataset and dropped effective speed
    This is due to it's high colinearity with velo visible in notebooks/EDA.ipynb in our correlation charts
    """
    df = pd.read_csv(data)
    df = df.drop(columns=["mean_eff_speed", "std_eff_speed"], errors="ignore")
    return df 


def get_data_pitch_type_dict(df: pd.DataFrame, pitch_types: list[str]) -> dict[str, pd.DataFrame]:
    """Return one dataframe per pitch type."""
    return {pitch_type: df[df["pitch_type"] == pitch_type] for pitch_type in pitch_types}


def get_valid_pitch_types() -> list[str]:
    """Return pitch types that have enough data to be analyzed."""
    return ["FF", "SL", "SI", "CH", "CU", "FC"]


def build_univariate_equation(outcome: str) -> str:
    """Build the fixed-effects formula for univariate mixed models."""
    return f"{outcome} ~ age_c + age_c_sq + C(year)"


def build_univariate_equation_with_ext(outcome: str) -> str:
    """Build the fixed-effects formula with extension as a covariate."""
    return f"{outcome} ~ age_c + age_c_sq + C(year) + mean_ext"


def get_n_groups(result) -> int:
    """Return number of groups across statsmodels versions."""
    if hasattr(result, "ngroups"):
        return int(result.ngroups)

    model = result.model
    if hasattr(model, "group_labels"):
        return int(len(model.group_labels))
    if hasattr(model, "groups"):
        return int(pd.Series(model.groups).nunique())
    return 0


def get_default_outcomes() -> list[str]:
    """Return outcomes to evaluate across pitch types."""
    return DEFAULT_OUTCOMES.copy()


def get_age_mean(df: pd.DataFrame) -> float:
    """Calculate mean age for centering in mixed models."""
    return df["age"].mean()