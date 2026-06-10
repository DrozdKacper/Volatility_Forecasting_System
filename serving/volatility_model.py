import joblib
import mlflow
import pandas as pd
import torch

from config import SEQUENCE_LENGTH
from features.features import add_features, FEATURE_COLUMNS
from ge.validation import prepare_validator, validate_training_data, validate_inference_data
from model.gru_model import GRUModel
from utils.sequences import create_inference_sequences


class VolatilityForecastModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.scaler = joblib.load(
            context.artifacts["scaler"]
        )

        self.model = GRUModel()  # <-- KLUCZOWE

        state = torch.load(
            context.artifacts["gru_model"],
            map_location=self.device
        )

        self.model.load_state_dict(state)

        self.model.to(self.device)
        self.model.eval()

    def predict(self, context, model_input):
        df = model_input.copy()

        df = add_features(df)

        df = df.dropna()

        validate_inference_data(df)

        scaled = self.scaler.transform(
            df[FEATURE_COLUMNS]
        )

        sequence = create_inference_sequences(
            scaled,
            SEQUENCE_LENGTH
        )

        tensor = torch.tensor(
            sequence,
            dtype=torch.float32,
            device=self.device
        )

        with torch.no_grad():
            prediction = self.model(tensor)

        return pd.DataFrame({
            "prediction": [float(prediction.item())]
        })





