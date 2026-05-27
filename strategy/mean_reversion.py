"""极端超卖反弹策略 - 捕捉超跌反弹"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np


@dataclass
class ReversionSignal:
    direction: str
    strength: float
    entry_price: float
    stop_loss: float
    take_profit: float
    reason: str


class MeanReversionStrategy:
    def __init__(self, config: dict = None):
        if config is None:
            config = {}

        # RSI 参数
        self.rsi_period = config.get("indicators", {}).get("rsi", {}).get("period", 14)
        self.rsi_oversold = 25  # 极度超卖
        self.rsi_overbought = 75

        # 价格跌幅参数
        self.min_drop_pct = 0.03   # 最近3根K线累计跌幅 > 3%
        self.lookback = 3          # 回看3根K线

        # 止盈止损
        self.sl_atr_mult = 1.5
        self.tp_atr_mult = 2.0     # 目标: 回撤50%

    def analyze(self, df: pd.DataFrame) -> Optional[ReversionSignal]:
        """分析是否有超卖反弹信号"""
        if df.empty or len(df) < 20:
            return None

        # 计算指标
        df = self._calculate_indicators(df)

        # 检查信号
        return self._check_signal(df)

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算指标"""
        if 'rsi' not in df.columns:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

        if 'atr' not in df.columns:
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = tr.rolling(14).mean()

        if 'ema_fast' not in df.columns:
            df['ema_fast'] = df['close'].ewm(span=9, adjust=False).mean()

        return df

    def _check_signal(self, df: pd.DataFrame) -> Optional[ReversionSignal]:
        """检查超卖反弹信号"""
        if len(df) < self.lookback + 1:
            return None

        latest = df.iloc[-1]
        price = float(latest['close'])
        atr = float(latest.get('atr', price * 0.01))

        if atr <= 0:
            return None

        rsi = float(latest.get('rsi', 50))

        # 条件1: RSI 极度超卖
        if rsi > self.rsi_oversold:
            return None

        # 条件2: 最近N根K线累计跌幅 > 3%
        lookback_candles = df.iloc[-(self.lookback+1):-1]
        if len(lookback_candles) < self.lookback:
            return None

        start_price = float(lookback_candles.iloc[0]['open'])
        end_price = float(latest['close'])
        drop_pct = (start_price - end_price) / start_price

        if drop_pct < self.min_drop_pct:
            return None

        # 条件3: 当前K线是企稳信号（阳线或十字星）
        curr_open = float(latest['open'])
        curr_close = float(latest['close'])
        curr_body = abs(curr_close - curr_open)
        curr_range = float(latest['high']) - float(latest['low'])

        # 阳线或十字星（body < 30% of range）
        is_bullish = curr_close > curr_open
        is_doji = curr_body < curr_range * 0.3 if curr_range > 0 else False

        if not (is_bullish or is_doji):
            return None

        # 条件4: 收盘价高于最低价（有下影线）
        curr_low = float(latest['low'])
        lower_shadow = min(curr_open, curr_close) - curr_low
        if lower_shadow < curr_range * 0.2:
            return None  # 没有下影线，说明没有买盘支撑

        # 条件5: 价格在EMA下方（超跌）
        ema_fast = float(latest.get('ema_fast', price))
        if price > ema_fast:
            return None  # 价格在EMA上方，不算超跌

        # 计算止损止盈
        # 止损: 当前低点下方 1 ATR
        sl = curr_low - atr * self.sl_atr_mult
        stop_distance = price - sl

        # 止盈: 50% 回撤 或 2 ATR（取较小值）
        target_1 = price + stop_distance * 1.5  # 1.5:1 R:R
        target_2 = price + atr * self.tp_atr_mult
        tp = min(target_1, target_2)

        # 计算信号强度
        strength = 0.5
        if rsi < 20:
            strength += 0.15  # RSI 极度超卖
        if drop_pct > 0.05:
            strength += 0.1   # 大幅下跌
        if is_doji:
            strength += 0.1   # 十字星企稳
        if lower_shadow > curr_range * 0.4:
            strength += 0.1   # 长下影线

        strength = min(strength, 1.0)

        return ReversionSignal(
            direction="long",
            strength=strength,
            entry_price=price,
            stop_loss=round(sl, 8),
            take_profit=round(tp, 8),
            reason=f"超跌反弹: RSI={rsi:.1f}, 跌幅={drop_pct*100:.2f}%, {'十字星' if is_doji else '阳线'}企稳"
        )
