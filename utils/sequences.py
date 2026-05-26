import numpy as np

from features.features import FEATURE_COLUMNS

def create_sequences(df, seq_length):
    data = df[FEATURE_COLUMNS].values
    target = df["target"].values

    xs = np.array([
        data[i:i+seq_length]
        for i in range(len(df) - seq_length)
    ])

    ys = np.array([
        target[i+seq_length]
        for i in range(len(df) - seq_length)
    ])

    return xs, ys


def create_inference_sequences(data, seq_length):
    return np.expand_dims(data[-seq_length:], axis=0)