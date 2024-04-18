import os
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_data(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
       return pd.read_csv(
            path,
            encoding="cp1251",
            delimiter=";",
            header=None,
            low_memory=False,
        )
    elif path.endswith(".parquet"):
        return pd.read_parquet(path)
    else:
        raise Exception("unknown format")

def filter_features(data: pd.DataFrame, drop_features: list) -> pd.DataFrame:
    """Filters the input data based on the usage flag.
    """
    if not len(drop_features):
        return data

    if not all([d in data.columns for d in drop_features]):
        raise Exception('not all ffatures in dataframe')

    return data.drop(columns=drop_features)


def trnsfrm(x: pd.DataFrame,
            scale: bool,
            norm: bool,
            scale_params=None,
            norm_params=None) -> Tuple[np.ndarray, dict]:
    """
    Transforms the input data using either StandardScaler or MinMaxScaler based on user input.

    Args:
        x (np.ndarray): Input data as a numpy array.
        scale (bool): Flag to apply StandardScaler.
        norm (bool): Flag to apply MinMaxScaler.

    Returns:
        Tuple[np.ndarray, Dict]: Tuple containing the transformed data as a numpy array and a dictionary with information on the applied scaler (mean and standard deviation for StandardScaler, and minimum and maximum values for MinMaxScaler).

    """

    xx = x.copy()
    s = StandardScaler() # with_mean=scale_params["mean"], with_std=scale_params["std"]) if scale_params else StandardScaler()
    n = MinMaxScaler()
    # fix here
    #   with_mean=norm_params["min_"], with_std=norm_params["std"]) if scale_params else MinMaxScaler()
    info = {}
    if scale:
        xx = s.transform(xx) if scale_params else s.fit_transform(xx)
        info["scale"] = {"mean": s.mean_.tolist(), "std": s.scale_.tolist()}
    if norm:
        xx = n.transform(xx) if norm_params else n.fit_transform(xx)
        info["norm"] = {"min_": n.data_min_.tolist(), "max_": n.data_max_.tolist()}
    return xx, info
