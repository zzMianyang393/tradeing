"""ML模型模块 - XGBoost训练与预测"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


FEATURE_COLUMNS = [
    "ema_fast", "ema_medium", "ema_slow",
    "rsi", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_middle", "bb_lower",
    "atr", "volume_ratio", "obv",
    "stoch_rsi_k", "stoch_rsi_d",
    "price_change", "volatility",
]


class MLModel:
    def __init__(self, config: dict = None):
        if config is None:
            config = {}
        self.predict_horizon = config.get("ml", {}).get("predict_horizon", 4)
        self.confidence_threshold = config.get("ml", {}).get("confidence_threshold", 0.6)
        self.model_path = Path(config.get("ml", {}).get("model_path", "data/ml_model.pkl"))
        self.model: Optional[XGBClassifier] = None
        self._load_model()

    def _load_model(self):
        if self.model_path.exists():
            try:
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
                logger.info(f"加载ML模型: {self.model_path}")
            except Exception as e:
                logger.warning(f"加载模型失败: {e}")
                self.model = None

    def _save_model(self):
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"保存ML模型: {self.model_path}")

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in FEATURE_COLUMNS if c in df.columns]
        features = df[available].copy()
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill().bfill()
        return features

    def prepare_labels(self, df: pd.DataFrame) -> pd.Series:
        future_returns = (
            df["close"].shift(-self.predict_horizon) / df["close"] - 1
        )
        labels = (future_returns > 0).astype(int)
        return labels

    def train(self, df: pd.DataFrame, force: bool = False) -> dict:
        if self.model is not None and not force:
            logger.info("模型已存在，跳过训练（使用force=True强制重训练）")
            return {}

        features = self.prepare_features(df)
        labels = self.prepare_labels(df)

        valid_mask = features.notna().all(axis=1) & labels.notna()
        features = features[valid_mask]
        labels = labels[valid_mask]

        if len(features) < 100:
            logger.warning(f"训练数据不足: {len(features)} 条")
            return {}

        tscv = TimeSeriesSplit(n_splits=5)
        scores = []

        for train_idx, val_idx in tscv.split(features):
            X_train = features.iloc[train_idx]
            y_train = labels.iloc[train_idx]
            X_val = features.iloc[val_idx]
            y_val = labels.iloc[val_idx]

            model = XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            scores.append(accuracy_score(y_val, pred))

        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        self.model.fit(features, labels)
        self._save_model()

        metrics = {
            "accuracy": np.mean(scores),
            "std": np.std(scores),
            "train_size": len(features),
        }
        logger.info(
            f"ML模型训练完成: 准确率={metrics['accuracy']:.4f} "
            f"(±{metrics['std']:.4f}), 样本数={metrics['train_size']}"
        )
        return metrics

    def predict(self, df: pd.DataFrame) -> dict:
        if self.model is None:
            return {"long_prob": 0.5, "short_prob": 0.5, "confidence": 0}

        features = self.prepare_features(df)
        if features.empty or features.isna().all().any():
            return {"long_prob": 0.5, "short_prob": 0.5, "confidence": 0}

        latest = features.iloc[[-1]]
        proba = self.model.predict_proba(latest)[0]

        long_prob = proba[1]
        short_prob = proba[0]
        confidence = abs(long_prob - 0.5) * 2

        return {
            "long_prob": float(long_prob),
            "short_prob": float(short_prob),
            "confidence": float(confidence),
        }

    def should_confirm(self, signal_type: str, prediction: dict) -> bool:
        if signal_type == "long":
            return (
                prediction["long_prob"] >= self.confidence_threshold
                and prediction["confidence"] >= 0.2
            )
        elif signal_type == "short":
            return (
                prediction["short_prob"] >= self.confidence_threshold
                and prediction["confidence"] >= 0.2
            )
        return False
