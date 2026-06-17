import numpy as np
from config import HORIZON


def add_target(df, horizon=HORIZON):
    df = df.copy()

    vol_backwards = df["log_return"].rolling(window=horizon).std()

    df["target"] = np.log(vol_backwards.shift(-horizon) + 1e-8)

    return df.dropna()