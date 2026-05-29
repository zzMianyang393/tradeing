import unittest

import pandas as pd

from data.fetcher import DataFetcher


class MemoryStorage:
    def __init__(self):
        self.latest = None
        self.saved = []

    def get_latest_timestamp(self, symbol, timeframe):
        return self.latest

    def save_klines(self, symbol, timeframe, df):
        self.saved.append(df)
        self.latest = int(df["timestamp"].max())

    def load_klines(self, symbol, timeframe):
        if not self.saved:
            return pd.DataFrame()
        return pd.concat(self.saved, ignore_index=True)


class FlakyFetcher(DataFetcher):
    def __init__(self):
        self.storage = MemoryStorage()
        self.calls = 0

    def fetch_ohlcv(self, symbol, timeframe="15m", since=None, limit=100):
        self.calls += 1
        if self.calls == 2:
            return pd.DataFrame()
        if self.calls > 3:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "timestamp": since + 900_000,
                "open": 1,
                "high": 1,
                "low": 1,
                "close": 1,
                "volume": 1,
            }
        ])


class FetcherSyncTests(unittest.TestCase):
    def test_sync_retries_empty_batch_before_stopping(self):
        fetcher = FlakyFetcher()

        df = fetcher.sync_klines("BTC/USDT:USDT", "15m", days=1)

        self.assertEqual(fetcher.calls, 6)
        self.assertEqual(len(df), 2)


if __name__ == "__main__":
    unittest.main()
