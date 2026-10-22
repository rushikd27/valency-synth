"""Assemble model-ready feature tables from final_data.csv."""
import warnings

import numpy as np
import pandas as pd

from composition import describe, reduced_key

warnings.filterwarnings("ignore")

DFT_COLS = ["Decomposition Energy", "Band Gap"]


def load_data(path="data/raw/final_data.csv"):
    df = pd.read_csv(path)
    df["y"] = (~df["theoretical"].astype(bool)).astype(int)
    df["comp_key"] = df["Name"].map(lambda n: str(reduced_key(n)))
    return df


def composition_features(df):
    D = pd.DataFrame([describe(n) for n in df["Name"]], index=df.index)
    return D.drop(columns=["parse_ok"]).astype(float), D["parse_ok"].astype(bool)


def structure_onehot(df):
    return pd.get_dummies(df["Crystal Structure"], prefix="struct").astype(float)


def feature_sets(df):
    comp, ok = composition_features(df)
    sets = {
        "composition": comp,
        "composition+structure": pd.concat([comp, structure_onehot(df)], axis=1),
        "composition+structure+DFT": pd.concat([comp, structure_onehot(df), df[DFT_COLS]], axis=1),
        "DFT only": pd.concat([structure_onehot(df), df[DFT_COLS]], axis=1),
    }
    return sets, ok
