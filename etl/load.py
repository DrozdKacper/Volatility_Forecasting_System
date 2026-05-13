import logging

import pandas as pd

logger = logging.getLogger(__name__)

def load_data(path):

    logger.info(f"Loading data from: {path}")

    try:

        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        logger.info(
            f"Data successfully processed | final shape={df.shape} | "
            f"date range={df['timestamp'].min()} -> {df['timestamp'].max()}"
        )

        return df

    except FileNotFoundError as e:
        logger.exception(f"File not found: {path}")
        raise


    except pd.errors.EmptyDataError:

        logger.exception(f"Empty CSV file: {path}")

        raise


    except Exception as e:

        logger.exception(f"Unexpected error while loading data from {path}")

        raise