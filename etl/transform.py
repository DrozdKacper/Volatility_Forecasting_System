import logging

from features.features import add_features
from features.target import add_target



logger = logging.getLogger(__name__)


def transform_data(df):

    logger.info(f"Starting feature engineering | input shape={df.shape}")

    df = add_features(df)
    logger.info(f"After features | shape={df.shape}")

    df = add_target(df)
    logger.info(f"After target | shape={df.shape}")

    before_dropna = df.shape[0]
    df = df.dropna()
    after_dropna = df.shape[0]

    logger.info(
        f"Dropped NA rows | before={before_dropna}, after={after_dropna}, "
        f"removed={before_dropna - after_dropna}"
    )

    if df.empty:
        raise ValueError("Transform produced empty dataset after dropna")

    logger.info(f"Transformation completed | final shape={df.shape}")

    return df

