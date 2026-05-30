"""规则信号生成器 - 支持趋势跟踪和均值回归两种策略"""

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
    metadata: dict = None  # 额外数据（不计入信号评分）


class SignalGenerator:
    def __init__(self, config: dict = None):
        if config is None:
            config = {}
        self.direction_mode = config.get("rules", {}).get("direction_mode", "both")
        self.min_conditions = config.get("rules", {}).get("min_conditions", 4)
        self.min_conditions_strict = config.get("rules", {}).get("min_conditions_strict", 6)
        self.long_conditions_min = config.get("rules", {}).get("long_conditions_min", 0)  # 0=use min_conditions
        self.adx_threshold = config.get("adx", {}).get("threshold", 20)
        self.strategy_mode = config.get("rules", {}).get("strategy_mode", "trend")

    def generate(self, df: pd.DataFrame, htf_df: Optional[pd.DataFrame] = None) -> Optional[Signal]:
        if self.strategy_mode == "mean_reversion":
            return self._generate_mean_reversion(df)
        if self.strategy_mode == "momentum":
            return self._generate_momentum(df)
        if self.strategy_mode == "trend_4h_filter":
            return self._generate_trend_4h_filter(df, htf_df)
        return self._generate_trend(df)

    def _generate_trend(self, df: pd.DataFrame) -> Optional[Signal]:
        """趋势跟踪信号"""
        if df.empty or len(df) < 3:
            return None

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        adx = latest.get("adx", 0)
        if adx < self.adx_threshold:
            return None

        ema_slow = latest.get("ema_slow", 0)
        price = latest.get("close", 0)
        trend = "neutral"
        if ema_slow > 0:
            if price > ema_slow * 1.005:
                trend = "bullish"
            elif price < ema_slow * 0.995:
                trend = "bearish"

        # 做多条件
        if trend != "bearish":
            long_conditions = self._check_long_conditions(latest, prev, df)
            long_score = sum(long_conditions.values())
            total = len(long_conditions)
            # 使用专用做多阈值（如果设置），否则用通用阈值
            if self.long_conditions_min > 0:
                long_threshold = self.long_conditions_min
            elif self.direction_mode == "short_only":
                long_threshold = self.min_conditions_strict
            else:
                long_threshold = self.min_conditions
            if long_score >= long_threshold:
                strength = long_score / total
                reasons = [k for k, v in long_conditions.items() if v]
                return Signal(
                    type=SignalType.LONG,
                    strength=strength,
                    conditions=long_conditions,
                    reason=f"趋势做多({long_score}/{total}): {', '.join(reasons)}",
                )

        # 做空条件
        if trend != "bullish":
            short_conditions = self._check_short_conditions(latest, prev, df)
            short_score = sum(short_conditions.values())
            total = len(short_conditions)
            short_threshold = len(short_conditions)  # 要求所有条件都满足
            if short_score >= short_threshold:
                strength = short_score / total
                reasons = [k for k, v in short_conditions.items() if v]
                return Signal(
                    type=SignalType.SHORT,
                    strength=strength,
                    conditions=short_conditions,
                    reason=f"趋势做空({short_score}/{total}): {', '.join(reasons)}",
                )

        return None

    def _generate_mean_reversion(self, df: pd.DataFrame) -> Optional[Signal]:
        """均值回归信号 - 深度RSI反转 + BB确认（高选择性）"""
        if df.empty or len(df) < 3:
            return None

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        price = latest.get("close", 0)
        bb_upper = latest.get("bb_upper", 0)
        bb_lower = latest.get("bb_lower", 0)
        bb_middle = latest.get("bb_middle", 0)
        rsi = latest.get("rsi", 50)
        stoch_k = latest.get("stoch_rsi_k", 50)
        stoch_d = latest.get("stoch_rsi_d", 50)
        prev_stoch_k = prev.get("stoch_rsi_k", 50)
        prev_stoch_d = prev.get("stoch_rsi_d", 50)
        vol_ratio = latest.get("volume_ratio", 1)
        atr = latest.get("atr", 0)
        adx_dmp = latest.get("adx_dmp", 0)
        adx_dmn = latest.get("adx_dmn", 0)

        if bb_upper <= 0 or bb_lower <= 0 or atr <= 0:
            return None

        candle_low = latest.get("low", price)
        candle_close = latest.get("close", price)
        candle_open = latest.get("open", price)
        candle_high = latest.get("high", price)

        # ========== 做多：深度超卖反转 ==========
        long_conditions = {}

        # 条件1: RSI深度超卖（<30）
        long_conditions["RSI深度超卖"] = rsi < 30

        # 条件2: 价格跌破BB下轨
        long_conditions["跌破BB下轨"] = price < bb_lower

        # 条件3: StochRSI深度超卖或金叉
        stoch_cross_up = prev_stoch_k <= prev_stoch_d and stoch_k > stoch_d
        long_conditions["StochRSI反转"] = stoch_cross_up or stoch_k < 20

        # 条件4: 下影线支撑
        lower_wick = min(candle_open, candle_close) - candle_low
        long_conditions["下影线支撑"] = lower_wick > atr * 0.3

        # 条件5: 放量
        long_conditions["放量确认"] = vol_ratio > 1.3

        # 条件6: 实体确认（K线实体需有一定大小，排除十字星）
        candle_body = abs(candle_close - candle_open)
        long_conditions["实体确认"] = candle_body > atr * 0.3

        # 条件7: DI多头
        long_conditions["DI确认"] = adx_dmp > adx_dmn

        long_score = sum(long_conditions.values())
        long_total = len(long_conditions)
        long_threshold = self.min_conditions_strict if self.direction_mode == "short_only" else self.min_conditions

        if long_score >= long_threshold:
            strength = long_score / long_total
            if rsi < 20:
                strength = min(strength + 0.15, 1.0)

            # === Bonus条件（不计入阈值，但增加信号强度）===
            bonus_score = 0
            bonus_total = 0
            bonus_reasons = []

            # Bonus1: MACD柱状图方向（负→正 或 收窄 = 下跌动能减弱）
            macd_hist = latest.get("macd_hist", 0)
            prev_macd_hist = prev.get("macd_hist", 0)
            # 更敏感：柱状图在增大（负值减小或正值增大）
            if macd_hist > prev_macd_hist:
                bonus_score += 1
                bonus_reasons.append("MACD转多")
            bonus_total += 1

            # Bonus2: RSI底背离（价格接近近期低点但RSI更高 = 卖压衰竭）
            lookback_n = 20
            if len(df) >= lookback_n + 2:
                recent = df.iloc[-(lookback_n + 1):-1]
                prev_price_low = recent["low"].min()
                low_idx = recent["low"].idxmin()
                if low_idx in df.index:
                    prev_rsi_at_low = df.loc[low_idx].get("rsi", 50)
                else:
                    prev_rsi_at_low = 50
                # 放宽条件：价格在近期低点1.5%范围内，RSI更高
                if (candle_low <= prev_price_low * 1.015) and (rsi > prev_rsi_at_low + 3):
                    bonus_score += 1
                    bonus_reasons.append("RSI底背离")
            bonus_total += 1

            # Bonus加分
            if bonus_total > 0:
                strength = min(strength + 0.1 * (bonus_score / bonus_total), 1.0)

            reasons = [k for k, v in long_conditions.items() if v]
            reasons.extend(bonus_reasons)
            return Signal(
                type=SignalType.LONG,
                strength=strength,
                conditions=long_conditions,
                reason=f"深度反转做多({long_score}/{long_total}): {', '.join(reasons)}",
                metadata={"bb_middle": bb_middle},
            )

        # ========== 做空：深度超买回落 ==========
        short_conditions = {}

        # 条件1: RSI深度超买（>70）
        short_conditions["RSI深度超买"] = rsi > 70

        # 条件2: 价格突破BB上轨
        short_conditions["突破BB上轨"] = price > bb_upper

        # 条件3: StochRSI深度超买或死叉
        stoch_cross_down = prev_stoch_k >= prev_stoch_d and stoch_k < stoch_d
        short_conditions["StochRSI反转"] = stoch_cross_down or stoch_k > 80

        # 条件4: 上影线压力
        upper_wick = candle_high - max(candle_open, candle_close)
        short_conditions["上影线压力"] = upper_wick > atr * 0.3

        # 条件5: 放量
        short_conditions["放量确认"] = vol_ratio > 1.3

        # 条件6: 实体确认（K线实体需有一定大小，排除十字星）
        candle_body = abs(candle_close - candle_open)
        short_conditions["实体确认"] = candle_body > atr * 0.3

        # 条件7: DI空头
        short_conditions["DI确认"] = adx_dmn > adx_dmp

        short_score = sum(short_conditions.values())
        short_total = len(short_conditions)
        short_threshold = self.min_conditions_strict if self.direction_mode == "long_only" else self.min_conditions

        if short_score >= short_threshold:
            strength = short_score / short_total
            if rsi > 80:
                strength = min(strength + 0.15, 1.0)

            # === Bonus条件（不计入阈值，但增加信号强度）===
            bonus_score = 0
            bonus_total = 0
            bonus_reasons = []

            # Bonus1: MACD柱状图方向（正→负 或 收窄 = 上涨动能减弱）
            macd_hist = latest.get("macd_hist", 0)
            prev_macd_hist = prev.get("macd_hist", 0)
            # 更敏感：柱状图在减小（正值减小或负值增大）
            if macd_hist < prev_macd_hist:
                bonus_score += 1
                bonus_reasons.append("MACD转空")
            bonus_total += 1

            # Bonus2: RSI顶背离（价格接近近期高点但RSI更低 = 买压衰竭）
            lookback_n = 20
            if len(df) >= lookback_n + 2:
                recent = df.iloc[-(lookback_n + 1):-1]
                prev_price_high = recent["high"].max()
                high_idx = recent["high"].idxmax()
                if high_idx in df.index:
                    prev_rsi_at_high = df.loc[high_idx].get("rsi", 50)
                else:
                    prev_rsi_at_high = 50
                # 放宽条件：价格在近期高点1.5%范围内，RSI更低
                if (candle_high >= prev_price_high * 0.985) and (rsi < prev_rsi_at_high - 3):
                    bonus_score += 1
                    bonus_reasons.append("RSI顶背离")
            bonus_total += 1

            # Bonus加分
            if bonus_total > 0:
                strength = min(strength + 0.1 * (bonus_score / bonus_total), 1.0)

            reasons = [k for k, v in short_conditions.items() if v]
            reasons.extend(bonus_reasons)
            return Signal(
                type=SignalType.SHORT,
                strength=strength,
                conditions=short_conditions,
                reason=f"深度反转做空({short_score}/{short_total}): {', '.join(reasons)}",
                metadata={"bb_middle": bb_middle},
            )

        return None

    def _generate_momentum(self, df: pd.DataFrame) -> Optional[Signal]:
        """动量突破策略 — 顺势交易，捕捉趋势延续"""
        if df.empty or len(df) < 25:
            return None

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        price = latest.get("close", 0)
        adx = latest.get("adx", 0)
        if adx < self.adx_threshold:
            return None

        vol_ratio = latest.get("volume_ratio", 1)
        rsi = latest.get("rsi", 50)
        prev_rsi = prev.get("rsi", 50)
        ema_fast = latest.get("ema_fast", 0)
        ema_medium = latest.get("ema_medium", 0)
        ema_slow = latest.get("ema_slow", 0)
        macd_hist = latest.get("macd_hist", 0)
        prev_macd_hist = prev.get("macd_hist", 0)
        atr = latest.get("atr", 0)

        if ema_fast <= 0 or ema_medium <= 0 or atr <= 0:
            return None

        # 计算近20根K线的最高/最低价（不含当前K线）
        lookback = df.iloc[-21:-1]
        recent_high = lookback["high"].max()
        recent_low = lookback["low"].min()

        # DI方向
        adx_dmp = latest.get("adx_dmp", 0)
        adx_dmn = latest.get("adx_dmn", 0)

        # ========== 做多：向上突破 ==========
        long_conditions = {}

        # 条件1: 价格突破近20根K线最高价（需超过0.2%避免假突破）
        long_conditions["向上突破"] = price > recent_high * 1.002

        # 条件2: 成交量放大（突破需要量能确认）
        long_conditions["放量确认"] = vol_ratio > 1.3

        # 条件3: RSI > 50 且上升（多头动量）
        long_conditions["RSI多头"] = rsi > 50 and rsi > prev_rsi

        # 条件4: EMA快线 > 中线（趋势向上）
        long_conditions["EMA多头排列"] = ema_fast > ema_medium

        # 条件5: MACD柱状图为正且增长
        long_conditions["MACD动量"] = macd_hist > 0 and macd_hist > prev_macd_hist

        # 条件6: DI确认（多头力量）
        long_conditions["DI确认"] = adx_dmp > adx_dmn

        long_score = sum(long_conditions.values())
        long_total = len(long_conditions)
        long_threshold = self.min_conditions_strict if self.direction_mode == "short_only" else self.min_conditions

        if long_score >= long_threshold:
            strength = long_score / long_total
            if adx > 30:
                strength = min(strength + 0.1, 1.0)
            reasons = [k for k, v in long_conditions.items() if v]
            return Signal(
                type=SignalType.LONG,
                strength=strength,
                conditions=long_conditions,
                reason=f"动量做多({long_score}/{long_total}): {', '.join(reasons)}",
                metadata={"breakout_high": recent_high},
            )

        # ========== 做空：向下突破 ==========
        short_conditions = {}

        # 条件1: 价格跌破近20根K线最低价（需跌破0.2%避免假突破）
        short_conditions["向下突破"] = price < recent_low * 0.998

        # 条件2: 成交量放大
        short_conditions["放量确认"] = vol_ratio > 1.3

        # 条件3: RSI < 50 且下降（空头动量）
        short_conditions["RSI空头"] = rsi < 50 and rsi < prev_rsi

        # 条件4: EMA快线 < 中线（趋势向下）
        short_conditions["EMA空头排列"] = ema_fast < ema_medium

        # 条件5: MACD柱状图为负且下降
        short_conditions["MACD动量"] = macd_hist < 0 and macd_hist < prev_macd_hist

        # 条件6: DI确认（空头力量）
        short_conditions["DI确认"] = adx_dmn > adx_dmp

        short_score = sum(short_conditions.values())
        short_total = len(short_conditions)
        short_threshold = self.min_conditions_strict if self.direction_mode == "long_only" else self.min_conditions

        if short_score >= short_threshold:
            strength = short_score / short_total
            if adx > 30:
                strength = min(strength + 0.1, 1.0)
            reasons = [k for k, v in short_conditions.items() if v]
            return Signal(
                type=SignalType.SHORT,
                strength=strength,
                conditions=short_conditions,
                reason=f"动量做空({short_score}/{short_total}): {', '.join(reasons)}",
                metadata={"breakout_low": recent_low},
            )

        return None

    def _generate_trend_4h_filter(self, df: pd.DataFrame, htf_df: Optional[pd.DataFrame] = None) -> Optional[Signal]:
        """4小时趋势过滤策略 — 用4h EMA55判断趋势方向，15m RSI+成交量入场"""
        if df.empty or len(df) < 3:
            return None

        latest = df.iloc[-1]

        # ========== 4小时趋势判断 ==========
        if htf_df is not None and not htf_df.empty and len(htf_df) >= 60:
            # 使用提供的4h数据
            htf_close = htf_df["close"]
            htf_ema55 = htf_close.ewm(span=55, adjust=False).mean()
            htf_price = float(htf_close.iloc[-1])
            htf_ema55_val = float(htf_ema55.iloc[-1])
        else:
            # 从15m数据重采样到4h
            if "timestamp" in df.columns:
                tmp = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
                tmp.index = pd.to_datetime(tmp["timestamp"], unit="ms")
            else:
                tmp = df[["open", "high", "low", "close", "volume"]].copy()
                tmp.index = df.index

            df_4h = tmp.resample("4h").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna()

            if len(df_4h) < 60:
                return None

            htf_close = df_4h["close"]
            htf_ema55 = htf_close.ewm(span=55, adjust=False).mean()
            htf_price = float(htf_close.iloc[-1])
            htf_ema55_val = float(htf_ema55.iloc[-1])

        # 4h趋势判断
        if htf_price > htf_ema55_val * 1.01:
            htf4_trend = "bullish"
        elif htf_price < htf_ema55_val * 0.99:
            htf4_trend = "bearish"
        else:
            htf4_trend = "neutral"

        # ========== 15m入场条件 ==========
        price = latest.get("close", 0)
        rsi = latest.get("rsi", 50)
        vol_ratio = latest.get("volume_ratio", 1)

        # LONG entry (only in bullish 4h trend)
        if htf4_trend == "bullish":
            long_conditions = {
                "RSI超卖": rsi < 40,
                "放量确认": vol_ratio > 1.0,
            }
            long_score = sum(long_conditions.values())
            long_total = len(long_conditions)

            if long_score >= long_total:  # 所有条件都满足
                strength = long_score / long_total
                reasons = [k for k, v in long_conditions.items() if v]
                return Signal(
                    type=SignalType.LONG,
                    strength=strength,
                    conditions=long_conditions,
                    reason=f"4H趋势做多({htf4_trend})({long_score}/{long_total}): {', '.join(reasons)}",
                    metadata={"htf4_trend": htf4_trend, "htf_ema55": htf_ema55_val},
                )

        # SHORT entry (only in bearish 4h trend)
        if htf4_trend == "bearish":
            short_conditions = {
                "RSI超买": rsi > 60,
                "放量确认": vol_ratio > 1.0,
            }
            short_score = sum(short_conditions.values())
            short_total = len(short_conditions)

            if short_score >= short_total:  # 所有条件都满足
                strength = short_score / short_total
                reasons = [k for k, v in short_conditions.items() if v]
                return Signal(
                    type=SignalType.SHORT,
                    strength=strength,
                    conditions=short_conditions,
                    reason=f"4H趋势做空({htf4_trend})({short_score}/{short_total}): {', '.join(reasons)}",
                    metadata={"htf4_trend": htf4_trend, "htf_ema55": htf_ema55_val},
                )

        return None

    def _check_long_conditions(self, latest: pd.Series, prev: pd.Series, df: pd.DataFrame) -> dict:
        """做多条件检查 - 趋势跟踪：EMA排列 + ADX + 动量确认"""
        conditions = {}

        ema_fast = latest.get("ema_fast", 0)
        ema_medium = latest.get("ema_medium", 0)
        ema_slow = latest.get("ema_slow", 0)
        prev_ema_fast = prev.get("ema_fast", 0)
        prev_ema_medium = prev.get("ema_medium", 0)
        price = latest.get("close", 0)
        prev_price = prev.get("close", 0)

        # 条件1: EMA多头排列（fast > medium > slow）
        conditions["EMA多头排列"] = ema_fast > ema_medium > ema_slow if all([ema_fast, ema_medium, ema_slow]) else False

        # 条件2: 价格在EMA慢线之上（趋势确认）
        conditions["价格在慢线之上"] = price > ema_slow * 1.002 if ema_slow > 0 else False

        # 条件3: RSI动量（>50且上升）
        rsi = latest.get("rsi", 50)
        prev_rsi = prev.get("rsi", 50)
        conditions["RSI多头动量"] = rsi > 50 and rsi > prev_rsi

        # 条件4: 成交量确认（高于均量）
        vol_ratio = latest.get("volume_ratio", 1)
        conditions["放量确认"] = vol_ratio > 1.1

        # 条件5: DI确认（多头力量占优）
        adx_dmp = latest.get("adx_dmp", 0)
        adx_dmn = latest.get("adx_dmn", 0)
        conditions["DI多头"] = adx_dmp > adx_dmn

        # 条件6: MACD柱状图为正或转正
        macd_hist = latest.get("macd_hist", 0)
        prev_macd_hist = prev.get("macd_hist", 0)
        conditions["MACD多头"] = macd_hist > 0 or (macd_hist > prev_macd_hist and prev_macd_hist < 0)

        # 条件7: 前K线收阳（动量确认）
        prev_close = prev.get("close", 0)
        prev_open = prev.get("open", 0)
        conditions["前K线收阳"] = prev_close > prev_open

        return conditions

    def _check_short_conditions(self, latest: pd.Series, prev: pd.Series, df: pd.DataFrame) -> dict:
        """做空条件检查 - 趋势跟踪：EMA排列 + ADX + 动量确认"""
        conditions = {}

        ema_fast = latest.get("ema_fast", 0)
        ema_medium = latest.get("ema_medium", 0)
        ema_slow = latest.get("ema_slow", 0)
        prev_ema_fast = prev.get("ema_fast", 0)
        prev_ema_medium = prev.get("ema_medium", 0)
        price = latest.get("close", 0)
        prev_price = prev.get("close", 0)

        # 条件1: EMA空头排列（fast < medium < slow）
        conditions["EMA空头排列"] = ema_fast < ema_medium < ema_slow if all([ema_fast, ema_medium, ema_slow]) else False

        # 条件2: 价格在EMA慢线之下（趋势确认）
        conditions["价格在慢线之下"] = price < ema_slow * 0.998 if ema_slow > 0 else False

        # 条件3: RSI动量（<50且下降）
        rsi = latest.get("rsi", 50)
        prev_rsi = prev.get("rsi", 50)
        conditions["RSI空头动量"] = rsi < 50 and rsi < prev_rsi

        # 条件4: 成交量确认（高于均量）
        vol_ratio = latest.get("volume_ratio", 1)
        conditions["放量确认"] = vol_ratio > 1.1

        # 条件5: DI确认（空头力量占优）
        adx_dmp = latest.get("adx_dmp", 0)
        adx_dmn = latest.get("adx_dmn", 0)
        conditions["DI空头"] = adx_dmn > adx_dmp

        # 条件6: MACD柱状图为负或转负
        macd_hist = latest.get("macd_hist", 0)
        prev_macd_hist = prev.get("macd_hist", 0)
        conditions["MACD空头"] = macd_hist < 0 or (macd_hist < prev_macd_hist and prev_macd_hist > 0)

        # 条件7: 前K线收阴（动量确认）
        prev_close = prev.get("close", 0)
        prev_open = prev.get("open", 0)
        conditions["前K线收阴"] = prev_close < prev_open

        return conditions
