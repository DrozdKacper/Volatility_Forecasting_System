import numpy as np


def add_features(df):
    # 1. Logarytmiczne stopy zwrotu (Log Returns)
    # Kluczowe dla stacjonarności szeregu czasowego
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # 2. Log-Volatility (Target)
    # Obliczamy zmienność jako kroczące odchylenie standardowe log-zwrotów
    # Okno 24 (dla 1h) oznacza zmienność z ostatniej doby
    window = 24
    df['vol_raw'] = df['log_return'].rolling(window=window).std()

    # Przekształcenie logarytmiczne zmienności (to, o czym pisałeś 🔥)
    # Dodajemy małą stałą 1e-8, aby uniknąć log(0)
    df['log_vol'] = np.log(df['vol_raw'] + 1e-8)

    # 3. Dodatkowe cechy (Features) wspierające model
    # Logarytm wolumenu (wolumen często rośnie wykładniczo)
    df['log_volume'] = np.log(df['volume'] + 1)

    # Zmienność wolumenu (pomaga wykryć anomalie/skoki aktywności)
    df['vol_change'] = df['log_volume'].diff()

    # Range (High-Low) w skali logarytmicznej
    df['log_range'] = np.log(df['high'] / df['low'])

    # 4. Target: Co chcemy przewidzieć?
    # Przesuwamy log_vol o 1 interwał w tył, aby model uczył się przewidywać przyszłość
    df['target_log_vol'] = df['log_vol'].shift(-1)

    # Czyszczenie danych z NaN (powstałych przez rolling i shift)
    df.dropna(inplace=True)

    return df