# src/preprocess.py

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.replace("?", np.nan)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["weight"])
    df["max_glu_serum"] = df["max_glu_serum"].fillna("Not Measured")
    df["A1Cresult"] = df["A1Cresult"].fillna("Not Measured")

    df["readmitted_binary"] = df["readmitted"].apply(
        lambda x: 1 if x == "<30" else 0
    )

    return df.drop(columns=["readmitted"])

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(include="object").columns
    numerical_cols = X.select_dtypes(exclude="object").columns

    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numerical_cols),
        ]
    )
