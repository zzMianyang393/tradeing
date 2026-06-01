"""数据获取模块 - 通过ccxt获取OKX公开K线数据"""

from __future__ import annotations

import time
from typing import Optional
from datetime import datetime, timedelta

import ccxt
import pandas as pd
from loguru import logger

from .storage import DataStorage



# OKX 时间框架映射
TIMEFRAME_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


class DataFetcher:
    def __init__(self, storage: Optional[DataStorage] = None):
        self.exchange = ccxt.okx({"enableRateLimit": True})
        self.storage = storage or DataStorage()

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "15m",
        since: Optional[int] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol, timeframe, since=since, limit=limit
            )
        except Exception as e:
            logger.error(f"获取 {symbol} {timeframe} K线失败: {e}")
            return pd.DataFrame()

        if not ohlcv:
            return pd.DataFrame()

        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = df["timestamp"].astype(int)
        return df

    def sync_klines(
        self,
        symbol: str,
        timeframe: str = "15m",
        days: int = 90,
    ) -> pd.DataFrame:
        tf_ms = TIMEFRAME_MS.get(timeframe, 900_000)
        end_ts = int(time.time() * 1000)
        start_ts = end_ts - (days * 24 * 3600 * 1000)

        latest = self.storage.get_latest_timestamp(symbol, timeframe)
        if latest is not None and latest > start_ts:
            start_ts = latest + tf_ms

        all_data = []
        current = int(start_ts)
        empty_retries = 0

        while current < end_ts:
            batch = self.fetch_ohlcv(symbol, timeframe, since=int(current), limit=300)
            if batch.empty:
                empty_retries += 1
                if empty_retries <= 2:
                    self.exchange = ccxt.okx({"enableRateLimit": True})
                    time.sleep(1)
                    continue
                break
            empty_retries = 0

            all_data.append(batch)
            last_ts = batch["timestamp"].max()
            if last_ts <= current:
                break
            current = last_ts + tf_ms
            time.sleep(0.5)  # 避免频率限制

        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            df.drop_duplicates(subset=["timestamp"], inplace=True)
            df.sort_values("timestamp", inplace=True)
            self.storage.save_klines(symbol, timeframe, df)
            logger.info(f"同步 {symbol} {timeframe}: {len(df)} 条K线")

        # Always return all historical data (new + existing)
        return self.storage.load_klines(symbol, timeframe)

    def fetch_ticker(self, symbol: str) -> dict:
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            logger.error(f"获取 {symbol} ticker 失败: {e}")
            return {}

    def fetch_all_swap_tickers(self) -> list:
        try:
            tickers = self.exchange.fetch_tickers(params={"instType": "SWAP"})
            return [
                {
                    "symbol": sym,
                    "last": t.get("last", 0),
                    "quoteVolume": t.get("quoteVolume", 0),
                    "percentage": t.get("percentage", 0),
                    "baseVolume": t.get("baseVolume", 0),
                }
                for sym, t in tickers.items()
                if "/USDT:USDT" in sym
            ]
        except Exception as e:
            logger.error(f"获取全部tickers失败: {e}")
            return []

    def fetch_multi_timeframe(
        self, symbol: str, timeframes: list = None
    ) -> dict[str, pd.DataFrame]:
        if timeframes is None:
            timeframes = ["15m", "1h", "4h"]

        result = {}
        for tf in timeframes:
            result[tf] = self.sync_klines(symbol, tf, days=30)
        return result

    def fetch_funding_rate_history(
        self, symbol: str, days: int = 90
    ) -> pd.DataFrame:
        """获取资金费率历史（每8h一次）"""
        try:
            # OKX资金费率历史
            all_rates = []
            end_ts = int(time.time() * 1000)
            start_ts = end_ts - (days * 24 * 3600 * 1000)

            # ccxt的fetchFundingRateHistory
            rates = self.exchange.fetch_funding_rate_history(
                symbol, since=start_ts, limit=100
            )
            if rates:
                for r in rates:
                    all_rates.append({
                        "timestamp": r["timestamp"],
                        "funding_rate": r.get("fundingRate", 0),
                        "next_funding_time": r.get("nextFundingTimestamp", 0),
                    })

            if all_rates:
                df = pd.DataFrame(all_rates)
                df.drop_duplicates(subset=["timestamp"], inplace=True)
                df.sort_values("timestamp", inplace=True)
                logger.info(f"获取 {symbol} 资金费率: {len(df)} 条")
                return df
            else:
                logger.warning(f"{symbol} 无资金费率数据")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"获取 {symbol} 资金费率失败: {e}")
            return pd.DataFrame()
