import pybaseball
from pybaseball import statcast 
import pandas as pd
from tqdm import tqdm
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

def get_pitching_stats_year(year: int) -> None:
    # Skipping if intial pull crashes
    path = f"{DATA_DIR}/pitching_stats_{year}.parquet"
    if os.path.exists(path):
        print(f"{year} already finished")
        return
    
    # MLB Season is March - September 
    df = statcast(f'{year}-03-01', f'{year}-09-30')

    stats_record = ['pitcher', 'player_name', 'p_throws', 'pitch_type', 'pitch_name', 
    'release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z', 'release_extension',
    'release_pos_x', 'release_pos_z', 'effective_speed', 'game_type']

    # spin axis was introduced in 2020 
    if 'spin_axis' in df.columns: 
        stats_record += ['spin_axis']

    # Only Regular Season + Pitching Reqs we need 
    pitching_df = df[df['game_type'] == "R"][stats_record].copy()

    pitching_df = pitching_df.drop(columns=['game_type'])

    # Adding p_throws for Pat Venditte (MLB's only switch pitcher)
    groupby_cols = ["pitcher", "player_name", "pitch_type", "pitch_name", "p_throws"]

    agg_dict = {
        "mean_velo": ("release_speed", "mean"),
        "std_velo": ("release_speed", "std"),
        "mean_spin_rate": ("release_spin_rate", "mean"),
        "std_spin_rate": ("release_spin_rate", "std"),
        "mean_pfx_x": ("pfx_x", "mean"),
        "std_pfx_x": ("pfx_x", "std"),
        "mean_pfx_z": ("pfx_z", "mean"),
        "std_pfx_z": ("pfx_z", "std"),
        "mean_ext": ("release_extension", "mean"),
        "std_ext": ("release_extension", "std"),
        "mean_eff_speed": ("effective_speed", "mean"),
        "std_eff_speed": ("effective_speed", "std"), 
        "n_pitches": ("release_speed", "count")
    }

    if 'spin_axis' in pitching_df.columns:
        agg_dict["mean_spin_axis"] = ("spin_axis", "mean")
        agg_dict["std_spin_axis"] = ("spin_axis", "std")

    agg = (
        pitching_df.groupby(groupby_cols)
        .agg(**agg_dict)
        .reset_index()
    )

    # Removing uncommon position players pitching 
    agg = agg[agg["n_pitches"] > 150]
    agg["year"] = year

    os.makedirs(DATA_DIR, exist_ok=True)
    agg.to_parquet(f"{DATA_DIR}/pitching_stats_{year}.parquet", index=False)

    print(f"Finished {year} with {len(agg)} rows")

if __name__ == "__main__":
    pybaseball.cache.enable()
    # Pulling from 2015 with Statcast's introduction 
    for year in tqdm(range(2015, 2026)):
        print(f"Processing year: {year}")
        get_pitching_stats_year(year)