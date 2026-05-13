import logging

import ccxt
import pandas as pd
import time
import os


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def fetch_all_ohlcv(symbol, timeframe, start_date):

    exchange = ccxt.binance()

    limit = 1000
    all_data = []

    since = exchange.parse8601(start_date)

    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000

    logging.info(
        f"Starting OHLCV download | "
        f"symbol={symbol}, timeframe={timeframe}, start_date={start_date}"
    )

    try:

        while True:

            logging.info(
                f"Fetching batch | since={pd.to_datetime(since, unit='ms')}"
            )

            ohlcv = exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since,
                limit=limit
            )

            if not ohlcv:
                logging.warning("No more data returned from exchange.")
                break

            all_data.extend(ohlcv)

            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + timeframe_ms

            logging.info(
                f"Fetched candles: {len(ohlcv)} | "
                f"Total candles: {len(all_data)}"
            )

            time.sleep(exchange.rateLimit / 1000)

            if len(ohlcv) < limit:
                logging.info(
                    "Received fewer candles than limit. "
                    "Reached end of available history."
                )
                break

        df = pd.DataFrame(
            all_data,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]
        )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            unit="ms"
        )

        logging.info(
            f"Successfully created DataFrame | shape={df.shape}"
        )

        return df

    except ccxt.NetworkError as e:
        logging.exception(
            f"Network error while downloading {symbol}: {e}"
        )

    except ccxt.ExchangeError as e:
        logging.exception(
            f"Exchange error while downloading {symbol}: {e}"
        )

    except Exception as e:
        logging.exception(
            f"Unexpected error while downloading {symbol}: {e}"
        )

    return pd.DataFrame()

