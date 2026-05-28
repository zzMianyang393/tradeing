import unittest

import pandas as pd

from strategy.ml_model import MLModel
from strategy.hybrid import HybridStrategy
from strategy.signals import Signal, SignalType


class TradeOutcomeLabelTests(unittest.TestCase):
    def _model(self):
        return MLModel({
            "ml": {
                "enabled": False,
                "predict_horizon": 3,
                "fee_rate": 0.0,
                "slippage": 0.0,
                "min_expected_return": 0.001,
                "model_path": "data/test_ml_model.pkl",
            },
            "stop_loss": {"fixed_pct": 0.01},
            "take_profit": {"fixed_pct": 0.02},
        })

    def test_long_label_uses_take_profit_before_future_close(self):
        df = pd.DataFrame([
            {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
            {"open": 100, "high": 103, "low": 100, "close": 100.5, "volume": 1},
            {"open": 100.5, "high": 101, "low": 100, "close": 100.2, "volume": 1},
            {"open": 100.2, "high": 101, "low": 99, "close": 100, "volume": 1},
        ])

        labels = self._model().prepare_trade_outcome_labels(df)

        self.assertAlmostEqual(labels.loc[0, "long_net_return"], 0.02)
        self.assertEqual(labels.loc[0, "action"], 2)

    def test_long_label_uses_stop_loss_when_stop_hits_first(self):
        df = pd.DataFrame([
            {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
            {"open": 100, "high": 100.5, "low": 98.5, "close": 100.3, "volume": 1},
            {"open": 100.3, "high": 103, "low": 100, "close": 102, "volume": 1},
            {"open": 102, "high": 103, "low": 101, "close": 102, "volume": 1},
        ])

        labels = self._model().prepare_trade_outcome_labels(df)

        self.assertAlmostEqual(labels.loc[0, "long_net_return"], -0.01)
        self.assertNotEqual(labels.loc[0, "action"], 2)

    def test_advisor_labels_include_future_return_and_excursions(self):
        df = pd.DataFrame([
            {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
            {"open": 100, "high": 104, "low": 98, "close": 101, "volume": 1},
            {"open": 101, "high": 103, "low": 97, "close": 99, "volume": 1},
            {"open": 99, "high": 102, "low": 96, "close": 98, "volume": 1},
        ])

        labels = self._model().prepare_advisor_labels(df)

        self.assertAlmostEqual(labels.loc[0, "future_return"], -0.02)
        self.assertAlmostEqual(labels.loc[0, "max_up_return"], 0.04)
        self.assertAlmostEqual(labels.loc[0, "max_down_return"], -0.04)


class HybridMlFilterTests(unittest.TestCase):
    def test_ml_filter_blocks_rule_signal_with_negative_expected_return(self):
        strategy = HybridStrategy({
            "ml": {
                "enabled": True,
                "mode": "filter",
                "min_expected_return": 0.001,
                "min_win_prob": 0.55,
                "model_path": "data/test_ml_model.pkl",
            }
        })
        signal = Signal(
            type=SignalType.LONG,
            strength=0.8,
            conditions={},
            reason="rule signal",
        )
        prediction = {
            "long_expected_return": -0.002,
            "long_win_prob": 0.40,
            "short_expected_return": 0.0,
            "short_win_prob": 0.50,
        }

        self.assertFalse(strategy._ml_allows_signal(signal, prediction))

    def test_ml_filter_allows_rule_signal_with_positive_expected_return(self):
        strategy = HybridStrategy({
            "ml": {
                "enabled": True,
                "mode": "filter",
                "min_expected_return": 0.001,
                "min_win_prob": 0.55,
                "model_path": "data/test_ml_model.pkl",
            }
        })
        signal = Signal(
            type=SignalType.LONG,
            strength=0.8,
            conditions={},
            reason="rule signal",
        )
        prediction = {
            "long_expected_return": 0.003,
            "long_win_prob": 0.62,
            "short_expected_return": 0.0,
            "short_win_prob": 0.50,
        }

        self.assertTrue(strategy._ml_allows_signal(signal, prediction))

    def test_advisor_builds_short_entry_from_expected_drop(self):
        strategy = HybridStrategy({
            "ml": {
                "enabled": True,
                "mode": "advisor",
                "min_expected_return": 0.001,
                "min_win_prob": 0.55,
                "take_profit_capture": 0.8,
                "model_path": "data/test_ml_model.pkl",
            },
            "stop_loss": {"fixed_pct": 0.008},
            "take_profit": {"fixed_pct": 0.005},
        })
        prediction = {
            "direction": "short",
            "expected_return": -0.012,
            "short_win_prob": 0.64,
            "max_up_return": 0.004,
            "max_down_return": -0.018,
        }

        advice = strategy._build_entry_advice("short", prediction)

        self.assertEqual(advice["direction"], "short")
        self.assertAlmostEqual(advice["take_profit_pct"], 0.0096)
        self.assertGreaterEqual(advice["stop_loss_pct"], 0.004)

    def test_position_advisor_closes_profitable_short_when_reversal_risk_is_high(self):
        strategy = HybridStrategy({
            "ml": {
                "enabled": True,
                "mode": "advisor",
                "model_path": "data/test_ml_model.pkl",
                "reversal_exit_prob": 0.55,
            }
        })
        prediction = {
            "direction": "long",
            "expected_return": 0.006,
            "long_win_prob": 0.62,
            "short_win_prob": 0.30,
        }

        advice = strategy._advise_open_position(
            side="short",
            entry_price=100,
            current_price=99,
            prediction=prediction,
        )

        self.assertEqual(advice["action"], "close")


if __name__ == "__main__":
    unittest.main()
