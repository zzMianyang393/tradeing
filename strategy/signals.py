"""规则信号生成器 - 趋势跟踪策略"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd
import numpy as np


class SignalType(Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass
class Signal:
    type: SignalType
    strength: float  # 0-1
    conditions: dict
    reason: str


class SignalGenerator:
    def __init__(self, config: dict = None):
        if config is None:
            config = {}
        self.direction_mode = config.get("rules", {}).get("direction_mode", "both")
        self.adx_threshold = config.get("adx", {}).get("threshold", 20)

    def generate(self, df: pd.DataFrame) -> Optional[Signal]:
        if df.empty or len(df) < 3:
            return None

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        adx = latest.get("adx", 0)
        if adx < self.adx_threshold:
            return None

        # 先检查做多（要求更严格：7/8条件）
        if self.direction_mode != "short_only":
            long_conditions = self._check_long_conditions(latest, prev, df)
            long_score = sum(long_conditions.values())
            total = len(long_conditions)
            if long_score >= 7:
                strength = long_score / total
                reasons = [k for k, v in long_conditions.items() if v]
                return Signal(
                    type=SignalType.LONG,
                    strength=strength,
                    conditions=long_conditions,
                    reason=f"趋势做多({long_score}/{total}): {', '.join(reasons)}",
                )

        # 再检查做空
        if self.direction_mode != "long_only":
            short_conditions = self._check_short_conditions(latest, prev, df)
            short_score = sum(short_conditions.values())
            total = len(short_conditions)
            if short_score >= 4:
                strength = short_score / total
                reasons = [k for k, v in short_conditions.items() if v]
                return Signal(
                    type=SignalType.SHORT,
                    strength=strength,
                    conditions=short_conditions,
                    reason=f"趋势做空({short_score}/{total}): {', '.join(reasons)}",
                )

        return None

    def _check_long_conditions(self, latest: pd.Series, prev: pd.Series, df: pd.DataFrame) -> dict:
        """做多条件检查 - 只在确认的上涨趋势中买入"""
        conditions = {}

        ema_fast = latest.get("ema_fast", 0)
        ema_medium = latest.get("ema_medium", 0)
        ema_slow = latest.get("ema_slow", 0)
        price = latest.get("close", 0)

        # 条件1: EMA多头排列（必须）
        conditions["EMA多头排列"] = ema_fast > ema_medium > ema_slow

        # 条件2: 价格必须在慢线之上（趋势确认）
        conditions["价格在慢线之上"] = price > ema_slow * 1.01 if ema_slow > 0 else False

        # 条件3: 价格回踩EMA（买入时机）
        if ema_fast > 0:
            ema9_dist = (price - ema_fast) / ema_fast
            conditions["价格回踩EMA"] = -0.003 <= ema9_dist <= 0.003
        else:
            conditions["价格回踩EMA"] = False

        # 条件4: RSI在健康区间（40-65）
        rsi = latest.get("rsi", 50)
        conditions["RSI健康区间"] = 40 < rsi < 65

        # 条件5: 成交量确认
        vol_ratio = latest.get("volume_ratio", 1)
        conditions["成交量确认"] = vol_ratio > 1.0  # 放量

        # 条件6: 前K线收阳
        prev_close = prev.get("close", 0)
        prev_open = prev.get("open", 0)
        conditions["前K线收阳"] = prev_close > prev_open

        # 条件7: DI确认（多头力量）
        adx_dmp = latest.get("adx_dmp", 0)
        adx_dmn = latest.get("adx_dmn", 0)
        conditions["DI确认"] = adx_dmp > adx_dmn

        # 条件8: ADX趋势强度
        adx = latest.get("adx", 0)
        conditions["ADX趋势"] = adx > 20

        return conditions

    def _check_short_conditions(self, latest: pd.Series, prev: pd.Series, df: pd.DataFrame) -> dict:
        """做空条件检查 - 做多的镜像"""
        conditions = {}

        ema_fast = latest.get("ema_fast", 0)
        ema_medium = latest.get("ema_medium", 0)
        ema_slow = latest.get("ema_slow", 0)
        conditions["EMA空头排列"] = ema_fast < ema_medium < ema_slow

        price = latest.get("close", 0)
        if ema_fast > 0:
            ema9_dist = (price - ema_fast) / ema_fast
            conditions["价格反弹EMA"] = -0.005 <= ema9_dist <= 0.005
        else:
            conditions["价格反弹EMA"] = False

        rsi = latest.get("rsi", 50)
        conditions["RSI未超卖"] = rsi > 35

        vol_ratio = latest.get("volume_ratio", 1)
        conditions["成交量确认"] = vol_ratio > 0.8

        prev_close = prev.get("close", 0)
        prev_open = prev.get("open", 0)
        conditions["前K线收阴"] = prev_close < prev_open

        adx_dmp = latest.get("adx_dmp", 0)
        adx_dmn = latest.get("adx_dmn", 0)
        conditions["DI确认"] = adx_dmn > adx_dmp

        return conditions
