
import ccxt
import pandas as pd
import time
import os

def fetch_all_ohlcv(symbol, timeframe, start_date):
    exchange = ccxt.binance()

    limit = 1000
    all_data = []

    since = exchange.parse8601(start_date)

    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000

    while True:
        ohlcv = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            since=since,
            limit=limit
        )

        if not ohlcv:
            break

        all_data.extend(ohlcv)

        last_timestamp = ohlcv[-1][0]
        since = last_timestamp + timeframe_ms

        print(f"Pobrano: {len(all_data)}")

        time.sleep(exchange.rateLimit / 1000)


        if len(ohlcv) < limit:
            break

    df = pd.DataFrame(
        all_data,
        columns=["timestamp","open","high","low","close","volume"]
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    return df


symbol = "BTC/USDT"
timeframe = "1h"
start_date = "2021-01-01T00:00:00Z"

df = fetch_all_ohlcv(symbol, timeframe, start_date)

print(df.info())

os.makedirs("data", exist_ok=True)

df.to_csv("data/btc_1h.csv", index=False)