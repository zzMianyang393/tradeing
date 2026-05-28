"""Hybrid strategy: rule signals first, ML as an optional trade filter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from loguru import logger

from .indicators import TechnicalIndicators
from .ml_model import MLModel, NEUTRAL_PREDICTION
from .signals import Signal, SignalGenerator


@dataclass
class TradeSignal:
    symbol: str
    direction: str
    strength: float
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

        ml_cfg = config.get("ml", {})
        self.ml_enabled = bool(ml_cfg.get("enabled", False))
        self.ml_mode = ml_cfg.get("mode", "filter")
        self.min_expected_return = ml_cfg.get("min_expected_return", 0.001)
        self.min_win_prob = ml_cfg.get("min_win_prob", 0.55)
        self.take_profit_capture = ml_cfg.get("take_profit_capture", 0.8)
        self.reversal_exit_prob = ml_cfg.get("reversal_exit_prob", 0.55)
        self.protect_profit_pct = ml_cfg.get("protect_profit_pct", 0.003)
        self.stop_loss_pct = config.get("stop_loss", {}).get("fixed_pct", 0.008)
        self.take_profit_pct = config.get("take_profit", {}).get("fixed_pct", 0.005)
        self.ml_model = MLModel(config)

    def _ml_allows_signal(self, rule_signal: Signal, prediction: dict) -> bool:
        if not self.ml_enabled:
            return True
        if self.ml_mode != "filter":
            return True
        return self.ml_model.should_confirm(rule_signal.type.value, prediction)

    def _build_entry_advice(self, direction: str, prediction: dict) -> dict:
        expected_return = abs(float(prediction.get("expected_return", 0.0)))
        take_profit_pct = max(
            expected_return * self.take_profit_capture,
            self.take_profit_pct,
        )

        if direction == "long":
            adverse_move = abs(float(prediction.get("max_down_return", 0.0)))
            win_prob = float(prediction.get("long_win_prob", prediction.get("long_prob", 0.0)))
        else:
            adverse_move = abs(float(prediction.get("max_up_return", 0.0)))
            win_prob = float(prediction.get("short_win_prob", prediction.get("short_prob", 0.0)))

        stop_loss_pct = max(adverse_move * 1.1, self.stop_loss_pct * 0.5)
        return {
            "direction": direction,
            "expected_return": float(prediction.get("expected_return", 0.0)),
            "win_prob": win_prob,
            "take_profit_pct": round(take_profit_pct, 6),
            "stop_loss_pct": round(stop_loss_pct, 6),
            "reason": (
                "ML advisor: expected={:.3%}, win_prob={:.3f}, tp={:.3%}, sl={:.3%}"
                .format(
                    prediction.get("expected_return", 0.0),
                    win_prob,
                    take_profit_pct,
                    stop_loss_pct,
                )
            ),
        }

    def _advise_open_position(
        self,
        side: str,
        entry_price: float,
        current_price: float,
        prediction: dict,
    ) -> dict:
        if entry_price <= 0 or current_price <= 0:
            return {"action": "hold", "reason": "invalid price"}

        if side == "long":
            pnl_pct = current_price / entry_price - 1
            continuation_prob = float(prediction.get("long_win_prob", 0.0))
            reversal_prob = float(prediction.get("short_win_prob", 0.0))
            predicted_against = prediction.get("direction") == "short"
        else:
            pnl_pct = entry_price / current_price - 1
            continuation_prob = float(prediction.get("short_win_prob", 0.0))
            reversal_prob = float(prediction.get("long_win_prob", 0.0))
            predicted_against = prediction.get("direction") == "long"

        if pnl_pct > 0 and (
            reversal_prob >= self.reversal_exit_prob or predicted_against
        ):
            return {
                "action": "close",
                "reason": (
                    "ML reversal risk: pnl={:.3%}, continuation={:.3f}, reversal={:.3f}"
                    .format(pnl_pct, continuation_prob, reversal_prob)
                ),
            }

        if pnl_pct >= self.protect_profit_pct and continuation_prob < self.min_win_prob:
            return {
                "action": "protect",
                "stop_price": entry_price,
                "reason": (
                    "ML trend weakening: pnl={:.3%}, continuation={:.3f}"
                    .format(pnl_pct, continuation_prob)
                ),
            }

        return {
            "action": "hold",
            "reason": (
                "ML hold: pnl={:.3%}, continuation={:.3f}, reversal={:.3f}"
                .format(pnl_pct, continuation_prob, reversal_prob)
            ),
        }

    def advise_position(
        self,
        symbol: str,
        df: pd.DataFrame,
        side: str,
        entry_price: float,
        current_price: float,
    ) -> dict:
        if not self.ml_enabled or self.ml_mode != "advisor":
            return {"action": "hold", "reason": "ML advisor disabled"}

        df = self.indicators.calculate(df)
        prediction = self.ml_model.predict(df)
        advice = self._advise_open_position(
            side=side,
            entry_price=entry_price,
            current_price=current_price,
            prediction=prediction,
        )
        advice["prediction"] = prediction
        advice["symbol"] = symbol
        return advice

    def analyze(
        self, symbol: str, df: pd.DataFrame, htf_df: Optional[pd.DataFrame] = None
    ) -> Optional[TradeSignal]:
        df = self.indicators.calculate(df)
        if df.empty or len(df) < 60:
            return None

        rule_signal = self.signal_gen.generate(df)
        if rule_signal is None:
            return None

        ml_pred = dict(NEUTRAL_PREDICTION)
        reason_suffix = "ML: disabled"
        signal_direction = rule_signal.type.value
        if self.ml_enabled:
            ml_pred = self.ml_model.predict(df)
            if not self._ml_allows_signal(rule_signal, ml_pred):
                expected_key = f"{rule_signal.type.value}_expected_return"
                win_key = f"{rule_signal.type.value}_win_prob"
                logger.info(
                    "ML filtered signal: {} {} expected={:.4%} win_prob={:.3f}".format(
                        symbol,
                        rule_signal.type.value.upper(),
                        ml_pred.get(expected_key, 0.0),
                        ml_pred.get(win_key, 0.0),
                    )
                )
                return None

            expected_key = f"{rule_signal.type.value}_expected_return"
            win_key = f"{rule_signal.type.value}_win_prob"
            reason_suffix = (
                "ML filter passed: expected={:.3%}, win_prob={:.3f}".format(
                    ml_pred.get(expected_key, 0.0),
                    ml_pred.get(win_key, 0.0),
                )
            )
            if self.ml_mode == "advisor":
                ml_direction = ml_pred.get("direction")
                ml_expected = abs(float(ml_pred.get("expected_return", 0.0)))
                ml_win_prob = float(ml_pred.get(f"{ml_direction}_win_prob", 0.0))
                if (
                    ml_direction in ("long", "short")
                    and ml_expected >= self.min_expected_return
                    and ml_win_prob >= self.min_win_prob
                ):
                    signal_direction = ml_direction
                advice = self._build_entry_advice(signal_direction, ml_pred)
                reason_suffix = advice["reason"]

        signal = TradeSignal(
            symbol=symbol,
            direction=signal_direction,
            strength=rule_signal.strength,
            rule_signal=rule_signal,
            ml_prediction=ml_pred,
            reason=f"{rule_signal.reason} | {reason_suffix}",
        )

        logger.info(
            "Trade signal: {} {} strength={:.3f} | {}".format(
                symbol, signal.direction.upper(), signal.strength, signal.reason
            )
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
            logger.warning("Not enough data to train ML model")
            return {}

        combined = pd.concat(all_dfs, ignore_index=True)
        return self.ml_model.train(combined, force=force)

    def analyze_multi(self, data: dict[str, pd.DataFrame]) -> list[TradeSignal]:
        signals = []
        for symbol, df in data.items():
            signal = self.analyze(symbol, df)
            if signal is not None:
                signals.append(signal)

        signals.sort(key=lambda s: s.strength, reverse=True)
        return signals
