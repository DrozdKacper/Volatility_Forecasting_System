import numpy as np
import pandas as pd

def ewma_baseline(returns, span=24):
    ewma = pd.Series(returns).ewm(span=span, adjust=False).std()
    return np.log(ewma.shift(1) + 1e-8).values


def persistence_baseline(targets):
    baseline = np.roll(targets, 1)
    baseline[0] = np.nan
    return baseline


def rolling_std_baseline(returns, window=24):
    vol = pd.Series(returns).rolling(window).std()
    return np.log(vol.shift(1) + 1e-8).values