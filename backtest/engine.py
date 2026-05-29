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
        self.cooldown_bars = 4  # 平仓后冷却4根K线（1小时）
        self.current_bar_index = 0

    def run(
        self,
        symbol: str,
        df: pd.DataFrame,
        train_data: Optional[pd.DataFrame] = None,
        htf_data: Optional[pd.DataFrame] = None,
    ) -> dict:
        # 每个币种回测前重置每日盈亏计数器
        self.account.daily_pnl = 0.0

        if train_data is not None and not train_data.empty:
            self.strategy.train_ml({symbol: train_data}, force=False)

        df = self.indicators.calculate(df)
        if df.empty or len(df) < 60:
            return {"error": "数据不足"}

        # 计算高时间框架指标（用于趋势过滤）
        htf_indicators = None
        if htf_data is not None and not htf_data.empty:
            htf_df = self.indicators.calculate(htf_data)
            if not htf_df.empty and len(htf_df) > 20:
                htf_indicators = htf_df

        logger.info(f"回测 {symbol}: {len(df)} 根K线")

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

            # 重采样15m为1h，用1h EMA判断趋势
            htf_trend = self._get_htf_trend(window)

            self.current_bar_index = i
            self._check_positions(symbol, current_price, current, current_time)
            self._check_signals(symbol, window, current_price, current_time, htf_trend, i)

        self._close_all_positions(symbol, df.iloc[-1], df.index[-1])

        stats = self.account.get_stats()
        stats["symbol"] = symbol
        stats["trades"] = self.trade_log
        return stats

    def run_multi(
        self,
        data: dict[str, pd.DataFrame],
        train_data: Optional[dict[str, pd.DataFrame]] = None,
        htf_data: Optional[dict[str, pd.DataFrame]] = None,
    ) -> dict:
        all_stats = {}
        for symbol, df in data.items():
            train = train_data.get(symbol) if train_data else None
            htf = htf_data.get(symbol) if htf_data else None
            all_stats[symbol] = self.run(symbol, df, train, htf)

        combined_stats = self.account.get_stats()
        combined_stats["by_symbol"] = all_stats
        combined_stats["trades"] = self.trade_log
        return combined_stats

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

        # 检查止损：用K线的最低/最高价（而非收盘价）
        if pos["direction"] == "long":
            hit_sl = candle_low <= pos["stop_loss"]
        else:
            hit_sl = candle_high >= pos["stop_loss"]

        if hit_sl:
            # 以止损价成交（而非收盘价）
            self._close_position(symbol, pos["stop_loss"], "止损", time)
            return

        # 追踪止损
        new_stop = self.stop_loss_mgr.should_move_stop(
            pos["entry_price"], price, pos["stop_loss"], pos["direction"]
        )
        if new_stop is not None:
            pos["stop_loss"] = new_stop

        # 检查止盈：用K线的最高/最低价
        if pos["direction"] == "long":
            hit_tp = candle_high >= pos["take_profit"]
        else:
            hit_tp = candle_low <= pos["take_profit"]

        if hit_tp and not pos.get("partial_closed"):
            self._close_position(symbol, pos["take_profit"], "止盈", time)
            return

    def _check_signals(
        self, symbol: str, df: pd.DataFrame, price: float, time: datetime,
        htf_trend: str = None, bar_index: int = 0,
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

        signal = self.strategy.analyze(symbol, df)
        if signal is None:
            return

        # 多时间框架过滤：只做顺势交易
        if htf_trend is not None:
            if signal.direction == "short" and htf_trend == "bullish":
                return  # 大趋势向上，不做空
            if signal.direction == "long" and htf_trend == "bearish":
                return  # 大趋势向下，不做多

        row = df.iloc[-1]

        # 滑点模拟：开仓价格恶化
        slippage = self.config.get("general", {}).get("slippage", 0.001)
        if signal.direction == "long":
            entry_price = price * (1 + slippage)  # 做多买入价更高
        else:
            entry_price = price * (1 - slippage)  # 做空卖出价更低

        fixed_sl_pct = self.config.get("stop_loss", {}).get("fixed_pct")
        if fixed_sl_pct is not None and fixed_sl_pct > 0:
            sl_price = price * (1 - fixed_sl_pct) if signal.direction == "long" else price * (1 + fixed_sl_pct)
            sl = StopLossResult(stop_price=round(sl_price, 8), stop_pct=fixed_sl_pct, method="FIXED")
        else:
            atr = float(row.get("atr", price * 0.01))
            sl = self.stop_loss_mgr.calculate_atr_stop(price, atr, signal.direction)

        rr_ratio = self.config.get("take_profit", {}).get("risk_reward_ratio", 2.0)
        fixed_tp_pct = self.config.get("take_profit", {}).get("fixed_pct")
        if fixed_tp_pct is not None and fixed_tp_pct > 0:
            tp_price = price * (1 + fixed_tp_pct) if signal.direction == "long" else price * (1 - fixed_tp_pct)
            tp = TakeProfitResult(target_price=round(tp_price, 8), target_pct=fixed_tp_pct, method="FIXED")
        else:
            tp = self.take_profit_mgr.calculate_target(price, sl.stop_pct, signal.direction)

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
        slippage = self.config.get("general", {}).get("slippage", 0.001)
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
        fee_rate = self.config.get("general", {}).get("fee_rate", 0.0005)
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
