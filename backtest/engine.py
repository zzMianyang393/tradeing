"""回测引擎 - 本地模拟历史交易"""

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

    def run(
        self,
        symbol: str,
        df: pd.DataFrame,
        train_data: Optional[pd.DataFrame] = None,
        htf_data: Optional[pd.DataFrame] = None,
    ) -> dict:
        if train_data is not None and not train_data.empty:
            self.strategy.train_ml({symbol: train_data}, force=False)

        df = self.indicators.calculate(df)
        if df.empty or len(df) < 60:
            return {"error": "数据不足"}

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

            self._check_positions(symbol, current_price, current, current_time)
            self._check_signals(symbol, window, current_price, current_time)

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
            all_stats[symbol] = self.run(symbol, df, train)

        combined_stats = self.account.get_stats()
        combined_stats["by_symbol"] = all_stats
        combined_stats["trades"] = self.trade_log
        return combined_stats

    def _check_positions(
        self, symbol: str, price: float, row: pd.Series, time: datetime
    ):
        if symbol not in self.current_positions:
            return

        pos = self.current_positions[symbol]

        if self.stop_loss_mgr.check_stop_hit(price, pos["stop_loss"], pos["direction"]):
            self._close_position(symbol, price, "止损", time)
            return

        new_stop = self.stop_loss_mgr.should_move_stop(
            pos["entry_price"], price, pos["stop_loss"], pos["direction"]
        )
        if new_stop is not None:
            pos["stop_loss"] = new_stop

        should_tp, _ = self.take_profit_mgr.should_take_profit(
            price, pos["entry_price"], pos["direction"]
        )
        if should_tp and not pos.get("partial_closed"):
            self._close_position(symbol, price, "部分止盈", time, pos["size"] * 0.5)
            pos["partial_closed"] = True

        if self.take_profit_mgr.should_full_close(
            price, pos["take_profit"], pos["direction"]
        ):
            self._close_position(symbol, price, "止盈", time)
            return

    def _check_signals(
        self, symbol: str, df: pd.DataFrame, price: float, time: datetime,
    ):
        if symbol in self.current_positions:
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

        row = df.iloc[-1]

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

        position = self.position_sizer.calculate_position(
            balance=self.account.balance,
            entry_price=price,
            stop_distance_pct=sl.stop_pct,
            signal_strength=signal.strength,
            current_positions=self.account.open_positions,
        )

        if position.amount_usdt <= 0:
            return

        self.current_positions[symbol] = {
            "direction": signal.direction,
            "entry_price": price,
            "size": position.amount_usdt,
            "leverage": position.leverage,
            "stop_loss": sl.stop_price,
            "take_profit": tp.target_price,
            "open_time": time,
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

        if pos["direction"] == "long":
            pnl_pct = (price - pos["entry_price"]) / pos["entry_price"]
        else:
            pnl_pct = (pos["entry_price"] - price) / pos["entry_price"]

        pnl_pct *= pos["leverage"]

        close_size = partial_size if partial_size else pos["size"]
        pnl = close_size * pnl_pct

        fee = close_size * 0.0005
        pnl -= fee

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
