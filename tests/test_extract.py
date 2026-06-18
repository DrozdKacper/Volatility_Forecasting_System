import pytest
import pandas as pd
from etl.extract import fetch_all_ohlcv

@pytest.mark.skip(reason="temporary - CI unsafe live API")
def test_extract_return_df():

    df = fetch_all_ohlcv("BTC/USDT", "1h", "2021-01-01T00:00:00Z")

    assert isinstance(df, pd.DataFrame)

@pytest.mark.skip(reason="temporary - CI unsafe live API")
def test_extract_columns():

    df = fetch_all_ohlcv("BTC/USDT", "1h", "2021-01-01T00:00:00Z")

    required = ["timestamp", "open", "high", "low", "close", "volume"]

    for col in required:
        assert col in df.columns