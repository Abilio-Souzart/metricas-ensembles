"""Leitura da base PEJOTA e divisão treino/teste."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_PATH, FEATURES, SEED

COLS_COM_BRANCOS = ["IFIN1", "IFIN2", "IFIN4", "IFIN5", "IFIN7", "IFIN8"]


def load_pejota() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH, sheet_name="PEJOTA")
    df.columns = df.columns.str.strip()                      # "IDADE " -> "IDADE"
    df[COLS_COM_BRANCOS] = (df[COLS_COM_BRANCOS]
                            .replace(r"^\s*$", np.nan, regex=True)
                            .apply(pd.to_numeric, errors="coerce"))
    df["DEFAULT"] = (df["STATUS"] == "MAU").astype(int)
    return df


@dataclass
class Split:
    X_tr: pd.DataFrame
    X_te: pd.DataFrame
    y_tr: pd.Series
    y_te: pd.Series


def make_split(df: pd.DataFrame, test_size: float, seed: int = SEED) -> Split:
    X, y = df[FEATURES], df["DEFAULT"]
    return Split(*train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y))
