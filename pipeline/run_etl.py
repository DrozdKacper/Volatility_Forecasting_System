import logging

from etl.extract import fetch_all_ohlcv
from etl.transform import transform_data
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":

    try:

        logging.info("Starting ETL pipeline")

        raw_df = fetch_all_ohlcv(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date="2021-01-01T00:00:00Z"
        )

        logging.info(f"Raw data fetched | shape={raw_df.shape}")

        (DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)
        raw_df.to_csv(DATA_DIR / "raw/btc_1h.csv", index=False)

        logging.info("Raw data saved to raw/btc_1h.csv")

        processed_df = transform_data(raw_df)

        logging.info(f"Processed data ready | shape={processed_df.shape}")

        (DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)
        processed_df.to_csv(DATA_DIR / "processed/btc_features.csv", index=False)

        logging.info("Processed data saved to processed/btc_features.csv")

    except Exception:
        logging.exception("ETL pipeline failed")
        raise