"""回测引擎 - 本地模拟历史交易"""

from __future__ import annotations

from typing import Optional
from datetime import datetime

import pandas as pd
from loguru import logger

from strategy.hybrid import HybridStrategy, TradeSignal
from strategy.indicators import TechnicalIndicators
from risk.position_sizer import PositionSizer
from risk.stop_loss import StopLossManager, StopLossResult
from risk.take_profit import TakeProfitManager, TakeProfitResult
from execution.account import AccountManager, TradeRecord



class BacktestEngine:
    def __init__(self, config: dict):
        self.config = config
        self.account = AccountManager(
            initial_capital=config.get("general", {}).get("initial_capital", 10.0)
        )
        self.strategy = HybridStrategy(config)
        self.indicators = TechnicalIndicators(config)
        self.position_sizer = PositionSizer(config)
        self.stop_loss_mgr = StopLossManager(config)
        self.take_profit_mgr = TakeProfitManager(config)

        self.current_positions: dict[str, dict] = {}
        self.trade_log: list[dict] = []
        self.last_close_bar: dict[str, int] = {}  # 每个币种上次平仓的bar index
        self.cooldown_bars = 6  # 平仓后冷却6根K线（1.5小时）
        self.current_bar_index = 0

        # Per-bar portfolio tracking for Sharpe calculation
        self.portfolio_values: list[float] = []
        self.daily_returns: list[float] = []

    def run(
        self,
        symbol: str,
        df: pd.DataFrame,
        train_data: Optional[pd.DataFrame] = None,
        htf_data: Optional[pd.DataFrame] = None,
        htf4_data: Optional[pd.DataFrame] = None,
        full_df: Optional[pd.DataFrame] = None,
    ) -> dict:
        # 每个币种回测前重置每日盈亏计数器
        self.account.daily_pnl = 0.0
        self.portfolio_values = []
        self.daily_returns = []

        if train_data is not None and not train_data.empty:
            self.strategy.train_ml({symbol: train_data}, force=False)

        df = self.indicators.calculate(df)
        if df.empty or len(df) < 60:
            return {"error": "数据不足"}

        # 计算1小时指标（用于趋势过滤）
        htf_1h = None
        if htf_data is not None and not htf_data.empty:
            htf_1h = self.indicators.calculate(htf_data)
            if htf_1h.empty or len(htf_1h) < 20:
                htf_1h = None

        # 计算4小时指标（用于大趋势判断）
        htf_4h = None
        if htf4_data is not None and not htf4_data.empty:
            htf_4h = self.indicators.calculate(htf4_data)
            if htf_4h.empty or len(htf_4h) < 10:
                htf_4h = None

        # 存储4h数据供 trend_4h_filter 策略使用
        # 优先用 full_df 重采样（包含完整历史，EMA55计算准确）
        # 否则用当前 df 重采样（窗口数据，可能不够长）
        resample_source = full_df if full_df is not None and not full_df.empty else df
        if htf_4h is None and "timestamp" in resample_source.columns:
            tmp = resample_source[["timestamp", "open", "high", "low", "close", "volume"]].copy()
            tmp.index = pd.to_datetime(tmp["timestamp"], unit="ms")
            resampled = tmp.resample("4h").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna()
            if len(resampled) >= 10:
                resampled = resampled.reset_index()
                resampled["timestamp"] = (resampled["timestamp"].astype("int64") // 1_000_000).astype(int)
                htf_4h = resampled
                logger.info(f"  自动重采样4h: {len(htf_4h)} bars")

        self._htf_4h_df = htf_4h

        logger.info(f"回测 {symbol}: {len(df)} 根K线"
                     + (f", 1h={len(htf_1h)}条" if htf_1h is not None else "")
                     + (f", 4h={len(htf_4h)}条" if htf_4h is not None else ""))

        for i in range(60, len(df)):
            window = df.iloc[:i + 1]
            current = df.iloc[i]
            current_price = float(current["close"])
            if "timestamp" in df.columns:
                current_ts = current.get("timestamp", 0)
            elif hasattr(df.index[i], "timestamp"):
                current_ts = int(df.index[i].timestamp() * 1000)
            else:
                current_ts = 0
            current_time = datetime.utcfromtimestamp(current_ts / 1000) if current_ts else datetime.utcnow()

            # 用实际1h数据判断趋势（优先），否则重采样15m
            if htf_1h is not None:
                htf_trend = self._get_htf_trend_from_data(htf_1h, current_ts)
            else:
                htf_trend = self._get_htf_trend(window)

            # 用4h数据判断大趋势
            htf4_trend = "neutral"
            if htf_4h is not None:
                htf4_trend = self._get_htf_trend_from_data(htf_4h, current_ts)

            self.current_bar_index = i
            self._check_positions(symbol, current_price, current, current_time)
            self._check_signals(symbol, window, current_price, current_time, htf_trend, i, htf4_trend)

            # Track portfolio value per bar
            self.portfolio_values.append(self.account.balance)

        self._close_all_positions(symbol, df.iloc[-1], df.index[-1])

        # Calculate daily returns from portfolio values (96 bars per day on 15m)
        if len(self.portfolio_values) > 1:
            bars_per_day = 96
            for i in range(bars_per_day, len(self.portfolio_values), bars_per_day):
                prev = self.portfolio_values[i - bars_per_day]
                curr = self.portfolio_values[i]
                if prev > 0:
                    self.daily_returns.append((curr - prev) / prev)

        stats = self.account.get_stats()
        stats["symbol"] = symbol
        stats["trades"] = self.trade_log
        return stats

    def run_multi(
        self,
        data: dict[str, pd.DataFrame],
        train_data: Optional[dict[str, pd.DataFrame]] = None,
        htf_data: Optional[dict[str, pd.DataFrame]] = None,
        htf4_data: Optional[dict[str, pd.DataFrame]] = None,
    ) -> dict:
        all_stats = {}
        for symbol, df in data.items():
            train = train_data.get(symbol) if train_data else None
            htf = htf_data.get(symbol) if htf_data else None
            htf4 = htf4_data.get(symbol) if htf4_data else None
            all_stats[symbol] = self.run(symbol, df, train, htf, htf4)

        combined_stats = self.account.get_stats()
        combined_stats["by_symbol"] = all_stats
        combined_stats["trades"] = self.trade_log
        return combined_stats

    def _get_htf_trend_from_data(self, htf_df: pd.DataFrame, current_ts: int) -> str:
        """用实际1h/4h数据判断当前时间的趋势方向"""
        if htf_df is None or htf_df.empty or len(htf_df) < 60:
            return "neutral"

        # 取到当前时间为止的数据
        if current_ts and "timestamp" in htf_df.columns:
            mask = htf_df["timestamp"] <= current_ts
            available = htf_df[mask]
            if len(available) < 30:
                return "neutral"
        else:
            available = htf_df

        close = available["close"]
        ema9 = close.ewm(span=9, adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        ema55 = close.ewm(span=55, adjust=False).mean()

        ef = ema9.iloc[-1]
        em = ema21.iloc[-1]
        es = ema55.iloc[-1]
        price = close.iloc[-1]

        if ef > em > es and price > es:
            return "bullish"
        if ef < em < es and price < es:
            return "bearish"
        return "neutral"

    def _get_htf_trend(self, window_15m: pd.DataFrame) -> str:
        """将15m数据重采样为1h，用1h EMA判断趋势方向"""
        if window_15m is None or len(window_15m) < 240:  # 至少40根1h K线
            return "neutral"

        # 重采样15m -> 1h (每4根15m合成1根1h)
        tmp = window_15m[["open", "high", "low", "close", "volume"]].copy()
        if "timestamp" in window_15m.columns:
            tmp.index = pd.to_datetime(window_15m["timestamp"], unit="ms")
        else:
            tmp.index = window_15m.index

        df_1h = tmp.resample("1h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()

        if len(df_1h) < 60:
            return "neutral"

        close = df_1h["close"]
        ema9 = close.ewm(span=9, adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        ema55 = close.ewm(span=55, adjust=False).mean()

        ef = ema9.iloc[-1]
        em = ema21.iloc[-1]
        es = ema55.iloc[-1]
        price = close.iloc[-1]

        if ef > em > es and price > es:
            return "bullish"
        if ef < em < es and price < es:
            return "bearish"

        return "neutral"

    def _check_positions(
        self, symbol: str, price: float, row: pd.Series, time: datetime
    ):
        if symbol not in self.current_positions:
            return

        pos = self.current_positions[symbol]
        candle_high = float(row["high"])
        candle_low = float(row["low"])
        candle_open = float(row["open"])

        # 检查止损和止盈
        if pos["direction"] == "long":
            hit_sl = candle_low <= pos["stop_loss"]
            hit_tp = candle_high >= pos["take_profit"]
        else:
            hit_sl = candle_high >= pos["stop_loss"]
            hit_tp = candle_low <= pos["take_profit"]

        # 当同一根K线同时触及止损和止盈时，用开盘价方向判断先触及哪个
        if hit_sl and hit_tp:
            if pos["direction"] == "long":
                # 做多：开盘价偏向止盈方向→先触TP（赢），偏向止损方向→先触SL（亏）
                if candle_open >= pos["entry_price"]:
                    self._close_position(symbol, pos["take_profit"], "止盈", time)
                else:
                    self._close_position(symbol, pos["stop_loss"], "止损", time)
            else:
                # 做空：开盘价偏向止盈方向→先触TP（赢），偏向止损方向→先触SL（亏）
                if candle_open <= pos["entry_price"]:
                    self._close_position(symbol, pos["take_profit"], "止盈", time)
                else:
                    self._close_position(symbol, pos["stop_loss"], "止损", time)
            return

        if hit_sl:
            self._close_position(symbol, pos["stop_loss"], "止损", time)
            return

        # 保本止损：盈利超过breakeven_trigger时，将止损移至入场价
        breakeven_trigger = self.config.get("stop_loss", {}).get("breakeven_trigger", 0)
        if breakeven_trigger > 0:
            if pos["direction"] == "long":
                unrealized = (price - pos["entry_price"]) / pos["entry_price"]
                if unrealized >= breakeven_trigger and pos["stop_loss"] < pos["entry_price"]:
                    pos["stop_loss"] = pos["entry_price"]
            else:
                unrealized = (pos["entry_price"] - price) / pos["entry_price"]
                if unrealized >= breakeven_trigger and pos["stop_loss"] > pos["entry_price"]:
                    pos["stop_loss"] = pos["entry_price"]

        # 追踪止损（可配置仅对多头生效；callback=0时禁用）
        trailing_callback = self.config.get("stop_loss", {}).get("trailing_callback", 0.012)
        if trailing_callback > 0:
            trailing_long_only = self.config.get("stop_loss", {}).get("trailing_long_only", False)
            if not trailing_long_only or pos["direction"] == "long":
                new_stop = self.stop_loss_mgr.should_move_stop(
                    pos["entry_price"], price, pos["stop_loss"], pos["direction"]
                )
                if new_stop is not None:
                    pos["stop_loss"] = new_stop

        # 部分止盈：盈利达到partial_close_trigger时平仓50%
        partial_trigger = self.config.get("take_profit", {}).get("partial_close_trigger", 0)
        partial_pct = self.config.get("take_profit", {}).get("partial_close_pct", 0.5)
        if partial_trigger > 0 and not pos.get("partial_closed"):
            if pos["direction"] == "long":
                unrealized = (price - pos["entry_price"]) / pos["entry_price"]
            else:
                unrealized = (pos["entry_price"] - price) / pos["entry_price"]
            if unrealized >= partial_trigger:
                partial_size = pos["size"] * partial_pct
                self._close_position(symbol, price, "部分止盈", time, partial_size=partial_size)
                pos["size"] -= partial_size
                pos["partial_closed"] = True
                # 将止损移至保本
                pos["stop_loss"] = pos["entry_price"]

        if hit_tp and not pos.get("partial_closed"):
            self._close_position(symbol, pos["take_profit"], "止盈", time)
            return

        # 时间止损：持仓超过max_holding_bars根K线且亏损时平仓
        max_holding_bars = self.config.get("stop_loss", {}).get("max_holding_bars", 0)
        if max_holding_bars > 0:
            bars_held = self.current_bar_index - pos.get("open_bar", 0)
            if bars_held >= max_holding_bars:
                if pos["direction"] == "long":
                    unrealized = (price - pos["entry_price"]) / pos["entry_price"]
                else:
                    unrealized = (pos["entry_price"] - price) / pos["entry_price"]
                if unrealized < 0:
                    self._close_position(symbol, price, "时间止损", time)

    def _check_signals(
        self, symbol: str, df: pd.DataFrame, price: float, time: datetime,
        htf_trend: str = None, bar_index: int = 0, htf4_trend: str = "neutral",
    ):
        if symbol in self.current_positions:
            return

        # 冷却期检查：平仓后等待N根K线再开新仓
        last_close = self.last_close_bar.get(symbol, -999)
        if bar_index - last_close < self.cooldown_bars:
            return

        can_open, reason = self.position_sizer.can_open_position(
            self.account.balance,
            self.account.daily_pnl,
            self.account.open_positions,
        )
        if not can_open:
            return

        # 获取4h窗口数据（用于 trend_4h_filter 策略）
        htf4_window = None
        if hasattr(self, '_htf_4h_df') and self._htf_4h_df is not None and "timestamp" in df.columns:
            current_ts = df.iloc[-1].get("timestamp", 0)
            htf_mask = self._htf_4h_df["timestamp"] <= current_ts
            htf4_window = self._htf_4h_df[htf_mask]

        signal = self.strategy.analyze(symbol, df, htf_df=htf4_window)
        if signal is None:
            return

        # 多时间框架趋势偏差（软性）
        if htf_trend is not None:
            if signal.direction == "long" and htf_trend == "bearish":
                signal.strength *= 0.9  # 熊市做多：轻微惩罚
            elif signal.direction == "short" and htf_trend == "bullish":
                signal.strength *= 0.9  # 牛市做空：轻微惩罚
            elif signal.direction == "long" and htf_trend == "bullish":
                signal.strength = min(signal.strength * 1.1, 1.0)  # 牛市做多：加分
            elif signal.direction == "short" and htf_trend == "bearish":
                signal.strength = min(signal.strength * 1.1, 1.0)  # 熊市做空：加分

        row = df.iloc[-1]

        # 滑点模拟：开仓价格恶化
        slippage = self.config.get("fees", {}).get("slippage", 0.0005)
        if signal.direction == "long":
            entry_price = price * (1 + slippage)  # 做多买入价更高
        else:
            entry_price = price * (1 - slippage)  # 做空卖出价更低

        atr = float(row.get("atr", price * 0.01))
        atr_pct = atr / entry_price if entry_price > 0 else 0.01

        # ATR自适应止损：根据波动率动态调整
        sl_cfg = self.config.get("stop_loss", {})
        atr_sl_mult = sl_cfg.get("atr_multiplier", 0)  # 0=使用固定比例

        if atr_sl_mult > 0:
            # ATR模式：SL = entry ± atr_multiplier * ATR
            sl_pct = atr_sl_mult * atr_pct
            sl_pct = max(sl_pct, 0.01)  # 最小1%
            sl_pct = min(sl_pct, 0.05)  # 最大5%
            sl_price = entry_price * (1 - sl_pct) if signal.direction == "long" else entry_price * (1 + sl_pct)
            sl = StopLossResult(stop_price=round(sl_price, 8), stop_pct=sl_pct, method="ATR")
        else:
            # 固定比例模式
            fixed_sl_pct = sl_cfg.get("fixed_pct", 0.025)
            if signal.direction == "short" and sl_cfg.get("fixed_pct_short"):
                fixed_sl_pct = sl_cfg["fixed_pct_short"]
            sl_price = entry_price * (1 - fixed_sl_pct) if signal.direction == "long" else entry_price * (1 + fixed_sl_pct)
            sl = StopLossResult(stop_price=round(sl_price, 8), stop_pct=fixed_sl_pct, method="FIXED")

        # ATR自适应止盈
        tp_cfg = self.config.get("take_profit", {})
        atr_tp_mult = tp_cfg.get("atr_multiplier", 0)

        if atr_tp_mult > 0:
            # ATR模式：TP = entry ± atr_multiplier * ATR
            tp_pct = atr_tp_mult * atr_pct
            tp_pct = max(tp_pct, 0.015)  # 最小1.5%
            tp_pct = min(tp_pct, 0.08)  # 最大8%
            tp_price = entry_price * (1 + tp_pct) if signal.direction == "long" else entry_price * (1 - tp_pct)
            tp = TakeProfitResult(target_price=round(tp_price, 8), target_pct=tp_pct, method="ATR")
        else:
            # 固定比例模式
            fixed_tp_pct = tp_cfg.get("fixed_pct", 0.0375)
            if signal.direction == "short" and tp_cfg.get("fixed_pct_short"):
                fixed_tp_pct = tp_cfg["fixed_pct_short"]
            tp_price = entry_price * (1 + fixed_tp_pct) if signal.direction == "long" else entry_price * (1 - fixed_tp_pct)
            tp = TakeProfitResult(target_price=round(tp_price, 8), target_pct=fixed_tp_pct, method="FIXED")

        # 获取历史交易统计用于凯利公式
        stats = self.account.get_stats()
        win_rate = stats.get("win_rate")
        avg_win = stats.get("avg_win")
        avg_loss = stats.get("avg_loss")
        total_trades = stats.get("total_trades", 0)

        position = self.position_sizer.calculate_position(
            balance=self.account.balance,
            entry_price=entry_price,
            stop_distance_pct=sl.stop_pct,
            signal_strength=signal.strength,
            current_positions=self.account.open_positions,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_trades=total_trades,
        )

        if position.amount_usdt <= 0:
            return

        self.current_positions[symbol] = {
            "direction": signal.direction,
            "entry_price": entry_price,  # 使用滑点后的价格
            "size": position.amount_usdt,
            "leverage": position.leverage,
            "stop_loss": sl.stop_price,
            "take_profit": tp.target_price,
            "open_time": time,
            "open_bar": self.current_bar_index,
            "signal": signal,
        }

        self.trade_log.append({
            "action": "open",
            "symbol": symbol,
            "direction": signal.direction,
            "price": price,
            "size": position.amount_usdt,
            "leverage": position.leverage,
            "stop_loss": sl.stop_price,
            "take_profit": tp.target_price,
            "time": str(time),
            "reason": signal.reason,
        })

        logger.debug(
            f"开仓 {symbol} {signal.direction.upper()} @ {price:.4f} "
            f"金额={position.amount_usdt:.4f} 杠杆={position.leverage}x"
        )

    def _close_position(
        self,
        symbol: str,
        price: float,
        reason: str,
        time: datetime,
        partial_size: Optional[float] = None,
    ):
        if symbol not in self.current_positions:
            return

        pos = self.current_positions[symbol]

        # 滑点模拟：平仓价格恶化
        slippage = self.config.get("fees", {}).get("slippage", 0.0005)
        if pos["direction"] == "long":
            close_price = price * (1 - slippage)  # 做多平仓卖出价更低
        else:
            close_price = price * (1 + slippage)  # 做空平仓买入价更高

        if pos["direction"] == "long":
            pnl_pct = (close_price - pos["entry_price"]) / pos["entry_price"]
        else:
            pnl_pct = (pos["entry_price"] - close_price) / pos["entry_price"]

        pnl_pct *= pos["leverage"]

        close_size = partial_size if partial_size else pos["size"]
        pnl = close_size * pnl_pct

        # 手续费: 开仓 + 平仓，都按名义价值(保证金*杠杆)收
        fee_rate = self.config.get("fees", {}).get("taker", 0.0005)
        notional = close_size * pos["leverage"]
        open_fee = notional * fee_rate
        close_fee = notional * fee_rate
        pnl -= (open_fee + close_fee)

        self.account.balance += pnl
        self.account.total_pnl += pnl
        self.account.daily_pnl += pnl

        record = TradeRecord(
            symbol=symbol,
            direction=pos["direction"],
            entry_price=pos["entry_price"],
            exit_price=price,
            size=close_size,
            leverage=pos["leverage"],
            pnl=round(pnl, 6),
            pnl_pct=round(pnl_pct, 6),
            open_time=pos["open_time"],
            close_time=time,
            close_reason=reason,
        )
        self.account.trade_history.append(record)

        self.trade_log.append({
            "action": "close",
            "symbol": symbol,
            "direction": pos["direction"],
            "entry_price": pos["entry_price"],
            "exit_price": price,
            "pnl": round(pnl, 6),
            "pnl_pct": round(pnl_pct, 6),
            "reason": reason,
            "time": str(time),
        })

        if partial_size is None:
            del self.current_positions[symbol]
            self.last_close_bar[symbol] = self.current_bar_index

        emoji = "+" if pnl >= 0 else ""
        logger.debug(
            f"平仓 {symbol} {pos['direction'].upper()} "
            f"入场={pos['entry_price']:.4f} 出场={price:.4f} "
            f"盈亏={emoji}{pnl:.4f}U ({emoji}{pnl_pct:.2%}) | {reason}"
        )

    def _close_all_positions(self, symbol: str, row: pd.Series, time):
        if symbol in self.current_positions:
            price = float(row["close"])
            self._close_position(symbol, price, "回测结束", time)
