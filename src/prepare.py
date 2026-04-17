"""Concat Stats and adding age via playerid_reverse_lookup"""

import pybaseball
from pybaseball import chadwick_register
import pandas as pd
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

def build_id_map() -> pd.DataFrame:
    """Bridge MLBAM IDs to birth years via Chadwick + Lahman People.csv"""
    print("Loading Chadwick register!")
    chadwick = (
        chadwick_register()[["key_mlbam", "key_bbref"]]
        .dropna(subset=["key_mlbam", "key_bbref"])
        .assign(key_mlbam=lambda x: x["key_mlbam"].astype(int))
    )

    print("Loading Lahman People.csv...")
    people = pd.read_csv(DATA_DIR / "People.csv")[["bbrefID", "birthYear"]]

    id_map = (
        chadwick
        .merge(people, left_on="key_bbref", right_on="bbrefID", how="left")
        [["key_mlbam", "birthYear"]]
        .dropna()
    )
    return id_map

def load_statcast() -> pd.DataFrame:
    """Concat all per-year parquets into one DataFrame"""
    print("Loading per-year parquets!")
    dfs = []
    for year in tqdm(range(2015, 2026)):
        path = DATA_DIR / f"pitching_stats_{year}.parquet"
        if not path.exists():
            print(f"Warning: {year} parquet missing, skipping")
            continue
        dfs.append(pd.read_parquet(path))
    return pd.concat(dfs, ignore_index=True)

if __name__ == "__main__":
    pybaseball.cache.enable()

    df = load_statcast()
    print(f"Loaded {len(df)} rows across {df['year'].nunique()} seasons")

    id_map = build_id_map()

    # Join age
    df = df.merge(id_map, left_on="pitcher", right_on="key_mlbam", how="left")
    df["age"] = df["year"] - df["birthYear"]
    df["age_sq"] = df["age"] ** 2
    df = df.drop(columns=["key_mlbam", "birthYear"])

    missing = df["age"].isna().sum()
    print(f"Missing age: {missing} rows dropped")
    df = df.dropna(subset=["age"])
    df["age"] = df["age"].astype(int)

    # Save master
    out_path = DATA_DIR / "pitching_master.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path.name} with {len(df)} rows")
    print(f"Age range: {df['age'].min()}–{df['age'].max()}")
    print(df[["pitcher", "player_name", "year", "age", "pitch_type", "mean_velo"]].head(10))

