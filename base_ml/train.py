import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from features.features import add_features, FEATURE_COLUMNS
from features.target import add_target
from config import*



def load_data(path):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def rolling_split(df, train_size=TRAIN_SIZE, test_size=TEST_SIZE, step=STEP):
    splits = []
    n = len(df)

    for start in range(0, n - train_size - test_size + 1, step):
        train = df.iloc[start:start + train_size]
        test = df.iloc[start + train_size:start + train_size + test_size]
        splits.append((train, test))

    return splits


def train_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=30,
        min_samples_split=50,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    print("MAE:", mae)


def evaluate_global(all_preds, all_targets, all_naive):
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    naive = np.concatenate(all_naive)

    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    naive_mae = mean_absolute_error(targets, naive)
    improvement = (naive_mae - mae) / naive_mae * 100
    print("\n===== GLOBAL EVALUATION =====")
    # --- SEKCOJA DEBUG ---
    print(f"DEBUG SKALI:")
    print(f"Średnia Target: {np.mean(targets):.4f} (Standardowe odchylenie: {np.std(targets):.4f})")
    print(f"Średnia Naive:  {np.mean(naive):.4f}")
    print(f"Średnia Preds:  {np.mean(preds):.4f}")
    # ---------------------
    print("\nImprovement vs naive:")
    print(f"{improvement:.2f}%")
    print(f"\nMAE:        {mae:.6f}")
    print(f"RMSE:       {rmse:.6f}")
    print(f"Naive MAE:  {naive_mae:.6f}")



if __name__ == "__main__":
    df = load_data("../data/btc_1h.csv")

    df = add_features(df)
    df = add_target(df)

    df = df.dropna()

    splits = rolling_split(df)

    all_preds = []
    all_targets = []
    all_naive = []

    for i, (train_df, test_df) in enumerate(splits):
        # Przygotowanie danych treningowych
        X_train = train_df[FEATURE_COLUMNS]
        y_train = train_df["target"]

        # Przygotowanie danych testowych
        X_test = test_df[FEATURE_COLUMNS]
        y_test = test_df["target"]

        # Trenowanie modelu
        model = train_model(X_train, y_train)

        # Predykcja modelu RF
        preds = model.predict(X_test)

        # --- MODEL NAIWNY (Benchmark) ---
        # Musi liczyć dokładnie to samo co target, ale na danych bieżących (bez shiftu)
        # Zakładając, że Twój target to rolling(24).std()
        vol_now_raw = test_df["log_return"].rolling(window=HORIZON).std()
        naive_series = np.log(vol_now_raw + 1e-8)

        # --- MASKOWANIE (Usuwanie NaN z początku okna rolling) ---
        # Tworzymy maskę, która jest True tylko tam, gdzie naiwny nie jest NaN
        mask = ~np.isnan(naive_series.values)

        # Filtrujemy wszystkie tablice tą samą maską, aby zachować spójność
        clean_preds = preds[mask]
        clean_targets = y_test.values[mask]
        clean_naive = naive_series.values[mask]

        # Sprawdzenie czy po filtrowaniu mamy dane
        if len(clean_targets) > 0:
            all_preds.append(clean_preds)
            all_targets.append(clean_targets)
            all_naive.append(clean_naive)

            if i % 50 == 0:
                print(f"Split {i} przetworzony pomyślnie...")
        else:
            print(f"Split {i} pominięty (brak danych po nałożeniu maski).")

    # Wywołanie ewaluacji globalnej
    if all_preds:
        evaluate_global(all_preds, all_targets, all_naive)
    else:
        print("Błąd: Nie zebrano żadnych danych do ewaluacji!")

importances = model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.title('Które cechy najbardziej wpływają na zmienność BTC?')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [FEATURE_COLUMNS[i] for i in indices])
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.show()