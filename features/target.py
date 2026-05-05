import numpy as np
from config import HORIZON


def add_target(df, horizon=HORIZON):
    df = df.copy()

    # 1. Obliczamy zmienność 'w tył' (standardowy rolling)
    # Ta wartość w wierszu 'T' oznacza zmienność z ostatnich H godzin.
    vol_backwards = df["log_return"].rolling(window=horizon).std()

    # 2. Przesuwamy o 'horizon' w górę (shift ujemny)
    # Teraz w wierszu 'T' mamy zmienność, która DOPIERO SIĘ WYDARZY
    # w oknie od T+1 do T+horizon.
    df["target"] = np.log(vol_backwards.shift(-horizon) + 1e-8)

    return df.dropna()