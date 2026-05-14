import pytest
import pandas as pd

from etl.transform import transform_data


@pytest.fixture
def sample_ohlcv_df():

    return pd.DataFrame({
        "timestamp": pd.date_range("2021-01-01", periods=100, freq="h"),
        "open": range(100),
        "high": range(100),
        "low": range(100),
        "close": range(100),
        "volume": range(100),
    })

@pytest.fixture
def transformed_df(sample_ohlcv_df):

    return transform_data(sample_ohlcv_df)