"""技术指标计算模块"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pandas_ta as ta


class TechnicalIndicators:
    def __init__(self, config: dict = None):
        if config is None:
            config = {}
        self.ema_fast = config.get("ema", {}).get("fast", 9)
        self.ema_medium = config.get("ema", {}).get("medium", 21)
        self.ema_slow = config.get("ema", {}).get("slow", 55)
        self.rsi_period = config.get("rsi", {}).get("period", 14)
        self.rsi_overbought = config.get("rsi", {}).get("overbought", 70)
        self.rsi_oversold = config.get("rsi", {}).get("oversold", 30)
        self.macd_fast = config.get("macd", {}).get("fast", 12)
        self.macd_slow = config.get("macd", {}).get("slow", 26)
        self.macd_signal = config.get("macd", {}).get("signal", 9)
        self.bb_period = config.get("bollinger", {}).get("period", 20)
        self.bb_std = config.get("bollinger", {}).get("std", 2)
        self.atr_period = config.get("atr", {}).get("period", 14)
        self.vol_sma_period = config.get("volume_sma", {}).get("period", 20)
        self.vol_spike = config.get("volume_sma", {}).get("spike_threshold", 1.5)
        self.adx_period = config.get("adx", {}).get("period", 14)
        self.adx_threshold = config.get("adx", {}).get("threshold", 20)

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 60:
            return df

        # 如果指标已计算过，跳过（避免回测循环中重复计算）
        if "ema_fast" in df.columns and "adx" in df.columns:
            return df

        df = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        df["ema_fast"] = ta.ema(close, length=self.ema_fast)
        df["ema_medium"] = ta.ema(close, length=self.ema_medium)
        df["ema_slow"] = ta.ema(close, length=self.ema_slow)

        df["rsi"] = ta.rsi(close, length=self.rsi_period)

        macd = ta.macd(
            close,
            fast=self.macd_fast,
            slow=self.macd_slow,
            signal=self.macd_signal,
        )
        if macd is not None and not macd.empty:
            df["macd"] = macd.iloc[:, 0]
            df["macd_signal"] = macd.iloc[:, 2]
            df["macd_hist"] = macd.iloc[:, 1]
        else:
            df["macd"] = 0.0
            df["macd_signal"] = 0.0
            df["macd_hist"] = 0.0

        bbands = ta.bbands(close, length=self.bb_period, std=self.bb_std)
        if bbands is not None and not bbands.empty:
            df["bb_upper"] = bbands.iloc[:, 2]
            df["bb_middle"] = bbands.iloc[:, 1]
            df["bb_lower"] = bbands.iloc[:, 0]
        else:
            df["bb_upper"] = close
            df["bb_middle"] = close
            df["bb_lower"] = close

        df["atr"] = ta.atr(high, low, close, length=self.atr_period)

        adx_result = ta.adx(high, low, close, length=self.adx_period)
        if adx_result is not None and not adx_result.empty:
            df["adx"] = adx_result.iloc[:, 0]
            df["adx_dmp"] = adx_result.iloc[:, 1]
            df["adx_dmn"] = adx_result.iloc[:, 2]
        else:
            df["adx"] = 0.0
            df["adx_dmp"] = 0.0
            df["adx_dmn"] = 0.0

        df["volume_sma"] = ta.sma(volume, length=self.vol_sma_period)
        df["volume_ratio"] = volume / df["volume_sma"].replace(0, np.nan)

        df["obv"] = ta.obv(close, volume)

        stoch_rsi = ta.stochrsi(
            close, length=self.rsi_period, rsi_length=self.rsi_period
        )
        if stoch_rsi is not None and not stoch_rsi.empty:
            df["stoch_rsi_k"] = stoch_rsi.iloc[:, 0]
            df["stoch_rsi_d"] = stoch_rsi.iloc[:, 1]
        else:
            df["stoch_rsi_k"] = 50.0
            df["stoch_rsi_d"] = 50.0

        df["price_change"] = close.pct_change()
        df["volatility"] = close.rolling(window=20).std() / close.rolling(window=20).mean()

        return df

    def calculate_multi_tf(
        self, data: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        result = {}
        for tf, df in data.items():
            result[tf] = self.calculate(df)
        return result
