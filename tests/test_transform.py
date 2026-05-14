import pytest
import pandas as pd

from etl.transform import transform_data


def test_transform_run(transformed_df):

    assert isinstance(transformed_df, pd.DataFrame)

def test_add_features(transformed_df, sample_ohlcv_df):


    assert transformed_df.shape[1] > sample_ohlcv_df.shape[1]

def test_no_nans(transformed_df):


    assert transformed_df.isna().sum().sum() == 0

