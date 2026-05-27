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

        long_conditions = self._check_long_conditions(latest, prev, df)

        long_score = sum(long_conditions.values())
        total = len(long_conditions)

        if long_score >= 4 and self.direction_mode != "short_only":
            strength = long_score / total
            reasons = [k for k, v in long_conditions.items() if v]
            return Signal(
                type=SignalType.LONG,
                strength=strength,
                conditions=long_conditions,
                reason=f"趋势做多({long_score}/{total}): {', '.join(reasons)}",
            )

        return None

    def _check_long_conditions(
        self, latest: pd.Series, prev: pd.Series, df: pd.DataFrame
    ) -> dict:
        conditions = {}

        ema_fast = latest.get("ema_fast", 0)
        ema_medium = latest.get("ema_medium", 0)
        ema_slow = latest.get("ema_slow", 0)
        conditions["EMA多头排列"] = ema_fast > ema_medium > ema_slow

        price = latest.get("close", 0)
        if ema_fast > 0:
            ema9_dist = (price - ema_fast) / ema_fast
            conditions["价格回踩EMA"] = -0.005 <= ema9_dist <= 0.005
        else:
            conditions["价格回踩EMA"] = False

        rsi = latest.get("rsi", 50)
        conditions["RSI未超买"] = rsi < 65

        vol_ratio = latest.get("volume_ratio", 1)
        conditions["成交量确认"] = vol_ratio > 0.8

        prev_close = prev.get("close", 0)
        prev_open = prev.get("open", 0)
        conditions["前K线收阳"] = prev_close > prev_open

        adx_dmp = latest.get("adx_dmp", 0)
        adx_dmn = latest.get("adx_dmn", 0)
        conditions["DI确认"] = adx_dmp > adx_dmn

        return conditions
