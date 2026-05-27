"""动量回踩策略 - 简单高盈亏比"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np


@dataclass
class MomentumSignal:
    direction: str
    strength: float
    entry_price: float
    stop_loss: float
    take_profit: float
    reason: str


class MomentumStrategy:
    def __init__(self, config: dict = None):
        if config is None:
            config = {}

        # 趋势识别参数
        self.ema_fast = config.get("indicators", {}).get("ema", {}).get("fast", 9)
        self.ema_slow = config.get("indicators", {}).get("ema", {}).get("slow", 55)

        # 动量参数
        self.min_candle_pct = 0.005   # 最小K线幅度 0.5%
        self.volume_spike = 1.5       # 放量倍数
        self.pullback_pct = 0.3       # 回踩到K线的30%-70%位置
        self.min_atr_mult = 1.5       # 最小止损距离 (ATR倍数)
        self.rr_ratio = 2.0           # 盈亏比 2:1

    def analyze(self, df: pd.DataFrame) -> Optional[MomentumSignal]:
        """分析是否有动量回踩信号"""
        if df.empty or len(df) < 20:
            return None

        # 计算指标
        df = self._calculate_indicators(df)

        # 检查最近5根K线是否有信号
        for i in range(max(0, len(df)-5), len(df)):
            signal = self._check_signal(df, i)
            if signal is not None:
                return signal

        return None

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算基础指标"""
        if 'ema_fast' not in df.columns:
            df['ema_fast'] = df['close'].ewm(span=self.ema_fast, adjust=False).mean()
        if 'ema_slow' not in df.columns:
            df['ema_slow'] = df['close'].ewm(span=self.ema_slow, adjust=False).mean()
        if 'atr' not in df.columns:
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = tr.rolling(14).mean()
        if 'volume_sma' not in df.columns:
            df['volume_sma'] = df['volume'].rolling(20).mean()

        return df

    def _check_signal(self, df: pd.DataFrame, idx: int) -> Optional[MomentumSignal]:
        """检查单根K线是否有信号"""
        if idx < 3:
            return None

        # 获取K线数据
        candle = df.iloc[idx]      # 当前K线（回踩K线）
        prev = df.iloc[idx-1]      # 前一根K线（动量K线）
        prev2 = df.iloc[idx-2]     # 更前一根

        price = float(candle['close'])
        atr = float(candle.get('atr', price * 0.01))

        if atr <= 0:
            return None

        # 条件1: 前一根是强势阳线
        prev_body = float(prev['close']) - float(prev['open'])
        prev_range = float(prev['high']) - float(prev['low'])
        prev_body_pct = prev_body / float(prev['open'])

        if prev_body <= 0:  # 必须是阳线
            return None
        if prev_body_pct < self.min_candle_pct:  # 幅度不够
            return None
        if prev_range <= 0:
            return None

        # 条件2: 前一根放量
        prev_volume = float(prev['volume'])
        volume_sma = float(prev.get('volume_sma', prev_volume))
        if volume_sma > 0 and prev_volume < volume_sma * self.volume_spike:
            return None

        # 条件3: 当前K线是回踩（阴线或小阳线，收盘低于前一根高点）
        curr_open = float(candle['open'])
        curr_close = float(candle['close'])
        curr_high = float(candle['high'])
        curr_low = float(candle['low'])
        prev_high = float(prev['high'])

        # 回踩条件：当前收盘 < 前一根最高价（有回调）
        if curr_close >= prev_high:
            return None

        # 回踩幅度：在前一根K线的30%-70%位置
        prev_body_top = float(prev['close'])
        prev_body_bottom = float(prev['open'])
        pullback_level = (prev_body_top - prev_body_bottom) * self.pullback_pct

        # 当前价格应该在回踩区域内
        if curr_close > prev_body_top - pullback_level:
            return None  # 回踩不够
        if curr_close < prev_body_bottom + pullback_level:
            return None  # 回踩太深

        # 条件4: 趋势向上（EMA快线 > 慢线）
        ema_fast = float(candle.get('ema_fast', 0))
        ema_slow = float(candle.get('ema_slow', 0))
        if ema_fast <= ema_slow:
            return None

        # 条件5: 价格在EMA快线附近（±1%）
        ema_distance = abs(curr_close - ema_fast) / ema_fast
        if ema_distance > 0.01:
            return None

        # 计算止损和止盈
        # 止损：回踩低点下方 0.5 ATR
        stop_loss = curr_low - atr * 0.5
        stop_distance = curr_close - stop_loss
        stop_pct = stop_distance / curr_close

        # 止盈：2:1 盈亏比
        take_profit = curr_close + stop_distance * self.rr_ratio

        # 确保止损距离合理（至少 1.5 ATR）
        if stop_distance < atr * self.min_atr_mult:
            stop_loss = curr_close - atr * self.min_atr_mult
            stop_distance = curr_close - stop_loss
            take_profit = curr_close + stop_distance * self.rr_ratio

        # 计算信号强度
        strength = 0.5
        if prev_body_pct > self.min_candle_pct * 2:
            strength += 0.1  # 大阳线加分
        if prev_volume > volume_sma * 2:
            strength += 0.1  # 大幅放量加分
        if ema_distance < 0.005:
            strength += 0.1  # 精准回踩加分

        strength = min(strength, 1.0)

        return MomentumSignal(
            direction="long",
            strength=strength,
            entry_price=curr_close,
            stop_loss=round(stop_loss, 8),
            take_profit=round(take_profit, 8),
            reason=f"动量回踩: 阳线{prev_body_pct*100:.2f}%+放量{prev_volume/volume_sma:.1f}x+回踩企稳"
        )
