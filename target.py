import numpy as np
from config import HORIZON

def add_target(df, horizon=HORIZON):
    df = df.copy()
    # Obliczamy zmienność kroczącą
    vol_realized = df["log_return"].rolling(window=horizon).std()

    # KLUCZ: shift(-horizon - 1)
    # To zapewnia, że żadna świeca użyta do obliczenia cech
    # nie jest użyta do obliczenia targetu.
    df["target"] = np.log(vol_realized.shift(-horizon - 1) + 1e-8)

    return df.dropna()