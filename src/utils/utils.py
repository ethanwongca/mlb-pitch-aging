from pathlib import Path

import pandas as pd


def get_pitch_type(data: str | Path, pitch_type: str) -> pd.DataFrame:
    """
    Return rows matching the requested pitch type.
    """
    df = pd.read_csv(data)
    return df[df["pitch_type"] == pitch_type]