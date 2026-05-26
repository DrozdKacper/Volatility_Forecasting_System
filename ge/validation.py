import great_expectations as gx

from config import SEQUENCE_LENGTH
from features.features import FEATURE_COLUMNS


def prepare_validator(df):

    context = gx.get_context()

    data_source = context.data_sources.add_pandas(
        name="processed_data"
    )

    data_asset = data_source.add_dataframe_asset(
        name="btc_features"
    )

    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        name="btc_batch"
    )

    batch = batch_definition.get_batch(
        batch_parameters={"dataframe": df}
    )

    validator = context.get_validator(batch=batch)

    return validator


def validate_training_data(validator):

    required_columns = FEATURE_COLUMNS + ["target"]

    for col in required_columns:
        validator.expect_column_values_to_not_be_null(col)


    validator.expect_column_values_to_be_unique(
        "timestamp"
    )


    validator.expect_column_values_to_be_between(
        "log_return",
        min_value=-0.5,
        max_value=0.5
    )

    validator.expect_column_values_to_be_between(
        "vol_5",
        min_value=0,
        max_value=1
    )

    validator.expect_column_values_to_be_between(
        "vol_10",
        min_value=0,
        max_value=1
    )

    validator.expect_column_values_to_be_between(
        "close_pos",
        min_value=0,
        max_value=1
    )

    validator.expect_column_values_to_be_between(
        "target",
        min_value=-10,
        max_value=0
    )



    validator.expect_column_mean_to_be_between(
        "target",
        min_value=-10,
        max_value=0
    )



    result = validator.validate()

    return result.success


def validate_inference_data(df):

    required_columns = FEATURE_COLUMNS


    if len(df) < SEQUENCE_LENGTH:
        raise ValueError(f"Need at least {SEQUENCE_LENGTH} rows")

    if df[required_columns].isnull().any().any():
        raise ValueError("Null values detected")

    if not df["timestamp"].is_unique:
        raise ValueError("Timestamp not unique")

    if (df["log_return"] < -0.5).any() or (df["log_return"] > 0.5).any():
        raise ValueError("log_return out of bounds")

    return True


