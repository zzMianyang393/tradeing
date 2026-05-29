"""Machine-learning model for trade-outcome filtering."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier, XGBRegressor


FEATURE_COLUMNS = [
    "ema_fast", "ema_medium", "ema_slow",
    "rsi", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_middle", "bb_lower",
    "atr", "volume_ratio", "obv",
    "stoch_rsi_k", "stoch_rsi_d",
    "price_change", "volatility",
]


NEUTRAL_PREDICTION = {
    "direction": "flat",
    "expected_return": 0.0,
    "max_up_return": 0.0,
    "max_down_return": 0.0,
    "long_prob": 0.5,
    "short_prob": 0.5,
    "flat_prob": 1.0,
    "confidence": 0.0,
    "long_win_prob": 0.5,
    "short_win_prob": 0.5,
    "long_expected_return": 0.0,
    "short_expected_return": 0.0,
    "suggested_take_profit_pct": 0.0,
    "suggested_stop_loss_pct": 0.0,
}


class MLModel:
    def __init__(self, config: dict = None):
        if config is None:
            config = {}

        ml_cfg = config.get("ml", {})
        self.predict_horizon = ml_cfg.get("predict_horizon", 4)
        self.confidence_threshold = ml_cfg.get("confidence_threshold", 0.6)
        self.min_expected_return = ml_cfg.get("min_expected_return", 0.001)
        self.min_win_prob = ml_cfg.get("min_win_prob", 0.55)
        self.take_profit_capture = ml_cfg.get("take_profit_capture", 0.8)
        self.fee_rate = ml_cfg.get("fee_rate", 0.0005)
        self.slippage = ml_cfg.get(
            "slippage", config.get("general", {}).get("slippage", 0.001)
        )
        self.stop_loss_pct = config.get("stop_loss", {}).get("fixed_pct", 0.008)
        self.take_profit_pct = config.get("take_profit", {}).get("fixed_pct", 0.005)
        self.model_path = Path(ml_cfg.get("model_path", "data/ml_model.pkl"))

        self.model: Optional[XGBClassifier] = None
        self.return_models: dict[str, XGBRegressor] = {}
        self.action_classes: list[int] = []
        self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            return

        try:
            with open(self.model_path, "rb") as f:
                saved = pickle.load(f)
            if isinstance(saved, dict) and "classifier" in saved:
                self.model = saved.get("classifier")
                self.return_models = saved.get("return_models", {})
                self.action_classes = saved.get("action_classes", [])
            else:
                self.model = saved
                self.action_classes = []
            logger.info(f"Loaded ML model: {self.model_path}")
        except Exception as e:
            logger.warning(f"Failed to load ML model: {e}")
            self.model = None
            self.return_models = {}

    def _save_model(self):
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump({
                "classifier": self.model,
                "return_models": self.return_models,
                "action_classes": self.action_classes,
            }, f)
        logger.info(f"Saved ML model: {self.model_path}")

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in FEATURE_COLUMNS if c in df.columns]
        features = df[available].copy()
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill().bfill()
        return features

    def _net_cost(self) -> float:
        return 2 * (self.fee_rate + self.slippage)

    def _simulate_trade_return(
        self, entry_price: float, future: pd.DataFrame, direction: str
    ) -> float:
        if future.empty or entry_price <= 0:
            return np.nan

        if direction == "long":
            stop_price = entry_price * (1 - self.stop_loss_pct)
            target_price = entry_price * (1 + self.take_profit_pct)
            for _, row in future.iterrows():
                if float(row["low"]) <= stop_price:
                    return -self.stop_loss_pct - self._net_cost()
                if float(row["high"]) >= target_price:
                    return self.take_profit_pct - self._net_cost()
            exit_price = float(future.iloc[-1]["close"])
            return (exit_price / entry_price - 1) - self._net_cost()

        stop_price = entry_price * (1 + self.stop_loss_pct)
        target_price = entry_price * (1 - self.take_profit_pct)
        for _, row in future.iterrows():
            if float(row["high"]) >= stop_price:
                return -self.stop_loss_pct - self._net_cost()
            if float(row["low"]) <= target_price:
                return self.take_profit_pct - self._net_cost()
        exit_price = float(future.iloc[-1]["close"])
        return (entry_price / exit_price - 1) - self._net_cost()

    def prepare_trade_outcome_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        source = df.reset_index(drop=True)
        for i, row in source.iterrows():
            future = source.iloc[i + 1:i + 1 + self.predict_horizon]
            if len(future) < self.predict_horizon:
                rows.append({
                    "long_net_return": np.nan,
                    "short_net_return": np.nan,
                    "action": np.nan,
                })
                continue

            entry_price = float(row["close"])
            long_ret = self._simulate_trade_return(entry_price, future, "long")
            short_ret = self._simulate_trade_return(entry_price, future, "short")

            action = 1  # flat / no trade
            if long_ret >= short_ret and long_ret > self.min_expected_return:
                action = 2
            elif short_ret > long_ret and short_ret > self.min_expected_return:
                action = 0

            rows.append({
                "long_net_return": long_ret,
                "short_net_return": short_ret,
                "action": action,
            })

        return pd.DataFrame(rows, index=df.index)

    def prepare_advisor_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        source = df.reset_index(drop=True)
        for i, row in source.iterrows():
            future = source.iloc[i + 1:i + 1 + self.predict_horizon]
            if len(future) < self.predict_horizon:
                rows.append({
                    "future_return": np.nan,
                    "max_up_return": np.nan,
                    "max_down_return": np.nan,
                })
                continue

            entry_price = float(row["close"])
            final_close = float(future.iloc[-1]["close"])
            future_high = float(future["high"].max())
            future_low = float(future["low"].min())
            rows.append({
                "future_return": final_close / entry_price - 1,
                "max_up_return": future_high / entry_price - 1,
                "max_down_return": future_low / entry_price - 1,
            })

        return pd.DataFrame(rows, index=df.index)

    def prepare_labels(self, df: pd.DataFrame) -> pd.Series:
        return self.prepare_trade_outcome_labels(df)["action"]

    def _classifier(self) -> XGBClassifier:
        return XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        )

    def _regressor(self) -> XGBRegressor:
        return XGBRegressor(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )

    def train(self, df: pd.DataFrame, force: bool = False) -> dict:
        if self.model is not None and not force:
            logger.info("ML model already exists; use force=True to retrain")
            return {}

        features = self.prepare_features(df)
        outcome_labels = self.prepare_trade_outcome_labels(df)
        advisor_labels = self.prepare_advisor_labels(df)
        labels = outcome_labels["action"]

        valid_mask = (
            features.notna().all(axis=1)
            & labels.notna()
            & advisor_labels.notna().all(axis=1)
        )
        features = features[valid_mask]
        labels = labels[valid_mask].astype(int)
        outcome_labels = outcome_labels.loc[features.index]
        advisor_labels = advisor_labels.loc[features.index]

        if len(features) < 100:
            logger.warning(f"Not enough ML training rows: {len(features)}")
            return {}
        if labels.nunique() < 2:
            logger.warning("Not enough ML action diversity to train classifier")
            return {}

        self.action_classes = sorted(int(v) for v in labels.unique())
        action_to_encoded = {
            action: encoded for encoded, action in enumerate(self.action_classes)
        }
        encoded_labels = labels.map(action_to_encoded).astype(int)

        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        for train_idx, val_idx in tscv.split(features):
            x_train = features.iloc[train_idx]
            y_train = encoded_labels.iloc[train_idx]
            x_val = features.iloc[val_idx]
            y_val = encoded_labels.iloc[val_idx]
            if y_train.nunique() < 2:
                continue

            model = self._classifier()
            model.fit(x_train, y_train)
            pred = model.predict(x_val)
            scores.append(accuracy_score(y_val, pred))
        if not scores:
            logger.warning("Not enough action diversity in ML validation folds")
            return {}

        self.model = self._classifier()
        self.model.fit(features, encoded_labels)

        self.return_models = {}
        for target in (
            "long_net_return",
            "short_net_return",
            "future_return",
            "max_up_return",
            "max_down_return",
        ):
            label_source = (
                outcome_labels if target in outcome_labels.columns else advisor_labels
            )
            valid_reg_mask = label_source[target].notna()
            if valid_reg_mask.sum() < 100:
                continue

            reg_features = features[valid_reg_mask]
            reg_labels = label_source.loc[valid_reg_mask, target]

            # 时序交叉验证回归器
            reg_scores = []
            for train_idx, val_idx in tscv.split(reg_features):
                x_train = reg_features.iloc[train_idx]
                y_train = reg_labels.iloc[train_idx]
                x_val = reg_features.iloc[val_idx]
                y_val = reg_labels.iloc[val_idx]

                reg = self._regressor()
                reg.fit(x_train, y_train)
                score = reg.score(x_val, y_val)
                reg_scores.append(score)

            # 用最后一折的模型（最近的数据）
            reg = self._regressor()
            reg.fit(reg_features, reg_labels)
            self.return_models[target] = reg

            avg_score = np.mean(reg_scores) if reg_scores else 0
            logger.debug(
                f"Regressor {target}: R²={avg_score:.4f} (+/- {np.std(reg_scores):.4f})"
            )

        self._save_model()

        metrics = {
            "accuracy": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "train_size": int(len(features)),
            "action_distribution": labels.value_counts(normalize=True).to_dict(),
        }
        logger.info(
            "ML trained: accuracy={:.4f} (+/- {:.4f}), rows={}".format(
                metrics["accuracy"], metrics["std"], metrics["train_size"]
            )
        )
        return metrics

    def predict(self, df: pd.DataFrame) -> dict:
        if self.model is None:
            return dict(NEUTRAL_PREDICTION)

        features = self.prepare_features(df)
        if features.empty or features.isna().all().any():
            return dict(NEUTRAL_PREDICTION)

        latest = features.iloc[[-1]]
        proba = self.model.predict_proba(latest)[0]
        prob_by_class = {}
        for cls, prob in zip(self.model.classes_, proba):
            encoded = int(cls)
            action = (
                self.action_classes[encoded]
                if encoded < len(self.action_classes)
                else encoded
            )
            prob_by_class[int(action)] = float(prob)

        short_prob = prob_by_class.get(0, 0.0)
        flat_prob = prob_by_class.get(1, 0.0)
        long_prob = prob_by_class.get(2, 0.0)
        confidence = max(long_prob, short_prob) - flat_prob

        long_expected = 0.0
        short_expected = 0.0
        future_return = 0.0
        max_up_return = 0.0
        max_down_return = 0.0
        if "long_net_return" in self.return_models:
            long_expected = float(self.return_models["long_net_return"].predict(latest)[0])
        if "short_net_return" in self.return_models:
            short_expected = float(self.return_models["short_net_return"].predict(latest)[0])
        if "future_return" in self.return_models:
            future_return = float(self.return_models["future_return"].predict(latest)[0])
        if "max_up_return" in self.return_models:
            max_up_return = float(self.return_models["max_up_return"].predict(latest)[0])
        if "max_down_return" in self.return_models:
            max_down_return = float(self.return_models["max_down_return"].predict(latest)[0])

        direction = "flat"
        expected_return = future_return
        if future_return > self.min_expected_return:
            direction = "long"
        elif future_return < -self.min_expected_return:
            direction = "short"

        suggested_take_profit = abs(expected_return) * self.take_profit_capture
        suggested_take_profit = max(suggested_take_profit, self.take_profit_pct)
        suggested_stop_loss = self.stop_loss_pct
        if direction == "long":
            suggested_stop_loss = max(abs(max_down_return) * 1.1, self.stop_loss_pct * 0.5)
        elif direction == "short":
            suggested_stop_loss = max(abs(max_up_return) * 1.1, self.stop_loss_pct * 0.5)

        return {
            "direction": direction,
            "expected_return": expected_return,
            "max_up_return": max_up_return,
            "max_down_return": max_down_return,
            "long_prob": long_prob,
            "short_prob": short_prob,
            "flat_prob": flat_prob,
            "confidence": float(confidence),
            "long_win_prob": long_prob,
            "short_win_prob": short_prob,
            "long_expected_return": long_expected,
            "short_expected_return": short_expected,
            "suggested_take_profit_pct": float(suggested_take_profit),
            "suggested_stop_loss_pct": float(suggested_stop_loss),
        }

    def should_confirm(self, signal_type: str, prediction: dict) -> bool:
        if signal_type == "long":
            return (
                prediction.get("long_expected_return", 0.0) >= self.min_expected_return
                and prediction.get("long_win_prob", prediction.get("long_prob", 0.0))
                >= self.min_win_prob
            )
        if signal_type == "short":
            return (
                prediction.get("short_expected_return", 0.0) >= self.min_expected_return
                and prediction.get("short_win_prob", prediction.get("short_prob", 0.0))
                >= self.min_win_prob
            )
        return False
