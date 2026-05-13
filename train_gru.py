import pandas as pd
import numpy as np

from baselines.baselines import ewma_baseline
from evaluation.evaluator import evaluate_model
from features.features import add_features, FEATURE_COLUMNS
from model.gru_model import GRUModel
from features.target import add_target
from config import*
from training.trainer_gru import train_model
from etl.load import load_data
from utils.seed import set_seed
from utils.sequences import create_sequences
from utils.split import rolling_split
from utils.dataset import create_dataloader
import torch
from sklearn.preprocessing import StandardScaler
import mlflow



if __name__ == "__main__":

    set_seed(SEED)

    mlflow.set_experiment("GRU_model")

    df = load_data("data/processed/btc_features.csv")

    splits = rolling_split(df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("CUDA AVAILABLE:", torch.cuda.is_available())
    print("DEVICE:", device)
    print(torch.__version__)
    print(torch.version.cuda)




    with mlflow.start_run(run_name="GRU_volatility_rolling"):
        model = GRUModel().to(device)
        results = []
        mlflow.log_param("model", "GRU")
        mlflow.log_param("hidden_size", model.hidden_size)
        mlflow.log_param("num_layers", model.num_layers)
        mlflow.log_param("input_size", len(FEATURE_COLUMNS))
        mlflow.log_param("sequence_length", SEQUENCE_LENGTH)
        mlflow.log_param("horizon", HORIZON)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("dropout", DROPOUT)
        mlflow.log_param("seed", SEED)

        for train_df, test_df in splits:

            scaler = StandardScaler()
            scaler.fit(train_df[FEATURE_COLUMNS])

            train_scaled = train_df.copy()
            test_scaled = test_df.copy()

            train_scaled[FEATURE_COLUMNS] = scaler.transform(train_scaled[FEATURE_COLUMNS])
            test_scaled[FEATURE_COLUMNS] = scaler.transform(test_scaled[FEATURE_COLUMNS])


            X_train, y_train = create_sequences(train_scaled, SEQUENCE_LENGTH)

            test_extended = pd.concat([
                train_scaled.tail(SEQUENCE_LENGTH),
                test_scaled
            ])

            X_test, y_test = create_sequences(test_extended, SEQUENCE_LENGTH)

            train_loader = create_dataloader(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = create_dataloader(X_test, y_test, batch_size=BATCH_SIZE, shuffle=False)

            train_model(
                model,
                train_loader,
                device,
                lr=LEARNING_RATE,
                epochs=EPOCHS
            )
            print(next(model.parameters()).device)

            returns = test_scaled["log_return"]
            baseline = ewma_baseline(returns, HORIZON)

            res = evaluate_model(
                model,
                test_loader,
                device,
                baseline_series=baseline
            )

            results.append(res)

        mse_list = [r["mse"] for r in results]
        mae_list = [r["mae"] for r in results]
        skill_list = [r.get("skill", 0) for r in results]
        corr_list = [r.get("corr", 0) for r in results]

        mlflow.log_metric("mse_mean", np.mean(mse_list))
        mlflow.log_metric("mse_std", np.std(mse_list))

        mlflow.log_metric("mae_mean", np.mean(mae_list))
        mlflow.log_metric("mae_std", np.std(mae_list))

        mlflow.log_metric("skill_mean", np.mean(skill_list))
        mlflow.log_metric("skill_std", np.std(skill_list))

        mlflow.log_metric("corr_mean", np.mean(corr_list))
        mlflow.pytorch.log_model(model, "model")




