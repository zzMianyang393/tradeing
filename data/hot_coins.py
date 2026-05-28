"""热门币种动态筛选模块"""

from __future__ import annotations

import time
from typing import Optional

from loguru import logger

from .fetcher import DataFetcher



class HotCoinSelector:
    def __init__(self, fetcher: Optional[DataFetcher] = None):
        self.fetcher = fetcher or DataFetcher()
        self._cache = {}
        self._last_refresh = 0
        self.refresh_interval = 3600  # 1小时刷新

    def get_top_coins(
        self,
        top_n: int = 18,
        min_volume_usdt: float = 1_000_000,
        blacklist: list = None,
    ) -> list[str]:
        now = time.time()
        if (
            self._cache
            and (now - self._last_refresh) < self.refresh_interval
        ):
            return self._cache.get("coins", [])

        if blacklist is None:
            blacklist = ["USDC", "BUSD", "TUSD", "DAI"]

        tickers = self.fetcher.fetch_all_swap_tickers()
        if not tickers:
            logger.warning("无法获取tickers，使用缓存")
            return self._cache.get("coins", [])

        filtered = []
        for t in tickers:
            symbol = t["symbol"]
            base = symbol.split("/")[0]
            if base in blacklist:
                continue
            quote_volume = float(t.get("quoteVolume") or 0)
            if quote_volume < min_volume_usdt:
                continue
            t["quoteVolume"] = quote_volume
            t["percentage"] = float(t.get("percentage") or 0)
            filtered.append(t)

        for t in filtered:
            vol_rank = 0
            price_rank = 0
            sorted_by_vol = sorted(filtered, key=lambda x: x["quoteVolume"], reverse=True)
            sorted_by_price = sorted(filtered, key=lambda x: abs(x["percentage"] or 0), reverse=True)
            for i, s in enumerate(sorted_by_vol):
                if s["symbol"] == t["symbol"]:
                    vol_rank = i
                    break
            for i, s in enumerate(sorted_by_price):
                if s["symbol"] == t["symbol"]:
                    price_rank = i
                    break
            t["score"] = vol_rank * 0.6 + price_rank * 0.4

        filtered.sort(key=lambda x: x["score"])

        result = [t["symbol"] for t in filtered[:top_n]]

        mandatory = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
        for m in mandatory:
            if m not in result:
                result.insert(0, m)
                if len(result) > top_n:
                    result.pop()

        self._cache["coins"] = result
        self._last_refresh = now

        logger.info(f"热门币种更新: {len(result)} 个")
        for i, coin in enumerate(result[:5]):
            logger.debug(f"  Top {i+1}: {coin}")

        return result

    def is_hot_coin(self, symbol: str, hot_list: list = None) -> bool:
        if hot_list is None:
            hot_list = self.get_top_coins()
        return symbol in hot_list
