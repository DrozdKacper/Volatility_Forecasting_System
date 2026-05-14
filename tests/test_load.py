import pandas as pd
from etl.load import load_data


def test_load_returns_dataframe(tmp_path):

    file = tmp_path / "data.csv"

    df = pd.DataFrame({
        "timestamp": ["2021-01-02", "2021-01-01"],
        "value": [1, 2]
    })

    df.to_csv(file, index=False)

    result = load_data(file)

    assert isinstance(result, pd.DataFrame)

def test_timestamp_converted(tmp_path):

    file = tmp_path / "data.csv"

    df = pd.DataFrame({
        "timestamp": ["2021-01-01", "2021-01-02"],
        "value": [1, 2]
    })

    df.to_csv(file, index=False)

    result = load_data(file)

    assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])


def test_sorted_by_timestamp(tmp_path):

    file = tmp_path / "data.csv"

    df = pd.DataFrame({
        "timestamp": ["2021-01-03", "2021-01-01", "2021-01-02"],
        "value": [1, 2, 3]
    })

    df.to_csv(file, index=False)

    result = load_data(file)

    assert result["timestamp"].is_monotonic_increasing