"""混合策略模块 - 结合规则信号和ML确认"""

from __future__ import annotations

from typing import Optional
from dataclasses import dataclass

import pandas as pd
from loguru import logger

from .indicators import TechnicalIndicators
from .signals import SignalGenerator, Signal, SignalType
from .ml_model import MLModel



@dataclass
class TradeSignal:
    symbol: str
    direction: str  # "long" or "short"
    strength: float  # 0-1, 决定杠杆
    rule_signal: Signal
    ml_prediction: dict
    reason: str


class HybridStrategy:
    def __init__(self, config: dict = None):
        if config is None:
            config = {}
        self.config = config
        self.indicators = TechnicalIndicators(config)
        self.signal_gen = SignalGenerator(config)
        self.ml_model = MLModel(config)

    def analyze(
        self, symbol: str, df: pd.DataFrame, htf_df: Optional[pd.DataFrame] = None
    ) -> Optional[TradeSignal]:
        df = self.indicators.calculate(df)
        if df.empty or len(df) < 60:
            return None

        rule_signal = self.signal_gen.generate(df)
        if rule_signal is None:
            return None

        ml_pred = self.ml_model.predict(df)

        strength = rule_signal.strength
        
        # ML确认：如果ML预测方向一致，增强信号强度
        ml_confirms = False
        if rule_signal.type == SignalType.LONG and ml_pred["long_prob"] > 0.55:
            ml_confirms = True
            strength = min(strength + 0.1, 1.0)
        elif rule_signal.type == SignalType.SHORT and ml_pred["short_prob"] > 0.55:
            ml_confirms = True
            strength = min(strength + 0.1, 1.0)

        signal = TradeSignal(
            symbol=symbol,
            direction=rule_signal.type.value,
            strength=strength,
            rule_signal=rule_signal,
            ml_prediction=ml_pred,
            reason=(
                f"{rule_signal.reason} | "
                f"ML: prob={ml_pred['long_prob'] if rule_signal.type == SignalType.LONG else ml_pred['short_prob']:.3f}"
                f"{' ✓确认' if ml_confirms else ''}"
            ),
        )

        logger.info(
            f"交易信号: {symbol} {signal.direction.upper()} "
            f"强度={signal.strength:.3f} | {signal.reason}"
        )

        return signal

    def train_ml(self, data: dict[str, pd.DataFrame], force: bool = False) -> dict:
        all_dfs = []
        for symbol, df in data.items():
            df = self.indicators.calculate(df)
            if not df.empty and len(df) >= 100:
                df["symbol"] = symbol
                all_dfs.append(df)

        if not all_dfs:
            logger.warning("没有足够的训练数据")
            return {}

        combined = pd.concat(all_dfs, ignore_index=True)
        return self.ml_model.train(combined, force=force)

    def analyze_multi(
        self, data: dict[str, pd.DataFrame]
    ) -> list[TradeSignal]:
        signals = []
        for symbol, df in data.items():
            signal = self.analyze(symbol, df)
            if signal is not None:
                signals.append(signal)

        signals.sort(key=lambda s: s.strength, reverse=True)
        return signals
