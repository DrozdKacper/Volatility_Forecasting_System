import numpy as np


def add_features(df):
    df = df.copy()

    # --- 1. PODSTAWY ---
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['log_range'] = np.log(df['high'] / df['low'])
    df['log_volume'] = np.log1p(df['volume'])

    # --- 2. ZMIENNOŚĆ KRÓTKOTERMINOWA (Bez wygładzania 24h) ---
    # Patrzymy tylko na ostatnie 5 i 10 interwałów
    df['vol_5'] = df['log_return'].rolling(5).std()
    df['vol_10'] = df['log_return'].rolling(10).std()

    # --- 3. ANOMALIE I SKOKI (Surge Features) ---
    # Czy obecny ruch/wolumen jest większy niż średnia z ostatniej doby?
    df['vol_surge'] = df['vol_5'] / (df['log_return'].rolling(24).std() + 1e-8)
    df['volume_surge'] = df['log_volume'] / (df['log_volume'].rolling(24).mean() + 1e-8)
    df['range_surge'] = df['log_range'] / (df['log_range'].rolling(24).mean() + 1e-8)

    # --- 4. POZYCJA CENY (Intraday Intensity) ---
    # Gdzie zamknęliśmy się w relacji do High-Low (0 = na dnie, 1 = na szczycie)
    df['close_pos'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)

    # Odległość od średniej kroczącej (czy rynek jest "rozciągnięty")
    df['dist_ma'] = df['close'] / df['close'].rolling(20).mean()

    # --- 5. LAGI SUROWYCH ZWROTÓW (Dla RF) ---
    df['log_ret_lag1'] = df['log_return'].shift(1)
    df['log_ret_lag2'] = df['log_return'].shift(2)

    return df.dropna()


# --- NOWA LISTA CECH ---
FEATURE_COLUMNS = [
    "log_return",
    "log_range",
    "vol_5",
    "vol_10",
    "vol_surge",
    "volume_surge",
    "range_surge",
    "close_pos",
    "dist_ma",
    "log_ret_lag1",
    "log_ret_lag2"
]