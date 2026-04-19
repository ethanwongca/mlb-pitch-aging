from pathlib import Path
import pandas as pd

# Path variables used across scripts.
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "master_data" / "pitching_master.csv"


DEFAULT_OUTCOMES = [
    "mean_velo",
    "mean_spin_rate",
    "mean_pfx_x",
    "mean_pfx_z",
    "mean_ext",
    "mean_spin_axis",
]

def load_data(data: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(data)
    return df

def get_data_pitch_type_dict(df: pd.DataFrame, pitch_types: list[str]) -> dict[str, pd.DataFrame]:
    """
    Return one dataframe per pitch type.
    """
    dict_type_dict = {}
    for pitch_type in pitch_types:
        dict_type_dict[pitch_type] = df[df["pitch_type"] == pitch_type]

    return dict_type_dict


def get_valid_pitch_types() -> list[str]:
    """
    Return pitch types that have enough data to be analyzed.
    """

    return ["FF", "SL", "SI", "CH", "CU", "FC"]


def build_univariate_equation(outcome: str) -> str:
    """Build the fixed-effects formula for univariate mixed models."""
    return f"{outcome} ~ age_c + age_c_sq + C(year)"


def get_default_outcomes() -> list[str]:
    """Return outcomes to evaluate across pitch types."""
    return DEFAULT_OUTCOMES.copy()


def is_realistic_peak_age(peak_age: float | None, low: float = 22, high: float = 35) -> bool:
    """Flag peak ages that fall in a realistic MLB aging window."""
    if peak_age is None:
        return False
    return low <= peak_age <= high

def get_age_mean(df: pd.DataFrame) -> float:
    """Calculate mean age for centering in mixed models."""
    return df["age"].mean()