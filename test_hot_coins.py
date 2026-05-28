import unittest

from data.hot_coins import HotCoinSelector


class FakeFetcher:
    def fetch_all_swap_tickers(self):
        return [
            {
                "symbol": "BAD/USDT:USDT",
                "last": 1,
                "quoteVolume": None,
                "percentage": None,
                "baseVolume": 0,
            },
            {
                "symbol": "BTC/USDT:USDT",
                "last": 100,
                "quoteVolume": 2_000_000,
                "percentage": 1.5,
                "baseVolume": 10,
            },
            {
                "symbol": "ETH/USDT:USDT",
                "last": 50,
                "quoteVolume": 1_500_000,
                "percentage": -2.0,
                "baseVolume": 20,
            },
        ]


class HotCoinSelectorTests(unittest.TestCase):
    def test_skips_tickers_with_missing_quote_volume(self):
        selector = HotCoinSelector(FakeFetcher())

        coins = selector.get_top_coins(top_n=2, min_volume_usdt=1_000_000)

        self.assertIn("BTC/USDT:USDT", coins)
        self.assertIn("ETH/USDT:USDT", coins)
        self.assertNotIn("BAD/USDT:USDT", coins)


if __name__ == "__main__":
    unittest.main()
