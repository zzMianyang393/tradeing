"""纸上交易引擎 - 本地模拟交易"""

from typing import Optional
from datetime import datetime

import pandas as pd
from loguru import logger

from execution.account import AccountManager
from strategy.hybrid import HybridStrategy, TradeSignal
from risk.position_sizer import PositionSizer
from risk.stop_loss import StopLossManager
from risk.take_profit import TakeProfitManager


class PaperTrader:
    def __init__(self, config: dict):
        self.config = config
        self.account = AccountManager(
            initial_capital=config.get("general", {}).get("initial_capital", 10.0)
        )
        self.strategy = HybridStrategy(config)
        self.position_sizer = PositionSizer(config)
        self.stop_loss_mgr = StopLossManager(config)
        self.take_profit_mgr = TakeProfitManager(config)

    def on_candle(self, symbol: str, df: pd.DataFrame):
        self._check_existing_positions(symbol, df)
        self._check_new_signals(symbol, df)

    def _check_existing_positions(self, symbol: str, df: pd.DataFrame):
        if symbol not in self.account.positions:
            return

        pos = self.account.positions[symbol]
        current_price = float(df["close"].iloc[-1])

        if self.stop_loss_mgr.check_stop_hit(
            current_price, pos.stop_loss, pos.direction
        ):
            self.account.close_position(symbol, current_price, "止损")
            return

        new_stop = self.stop_loss_mgr.should_move_stop(
            pos.entry_price, current_price, pos.stop_loss, pos.direction
        )
        if new_stop is not None:
            pos.stop_loss = new_stop
            logger.debug(f"{symbol}: 移动止损到 {new_stop:.8f}")

        should_tp, _ = self.take_profit_mgr.should_take_profit(
            current_price, pos.entry_price, pos.direction
        )
        if should_tp and not pos.partial_closed:
            partial_size = self.take_profit_mgr.get_partial_close_size(pos.size)
            self.account.close_position(symbol, current_price, "部分止盈")
            pos.partial_closed = True
            logger.info(f"{symbol}: 部分止盈 {partial_size:.4f}U")

        if self.take_profit_mgr.should_full_close(
            current_price, pos.take_profit, pos.direction
        ):
            self.account.close_position(symbol, current_price, "止盈")
            return

    def _check_new_signals(self, symbol: str, df: pd.DataFrame):
        if symbol in self.account.positions:
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

        latest = df.iloc[-1]
        entry_price = float(latest["close"])
        atr = float(latest.get("atr", entry_price * 0.01))

        sl_result = self.stop_loss_mgr.calculate_atr_stop(
            entry_price, atr, signal.direction
        )
        tp_result = self.take_profit_mgr.calculate_target(
            entry_price, sl_result.stop_pct, signal.direction
        )

        position = self.position_sizer.calculate_position(
            balance=self.account.balance,
            entry_price=entry_price,
            stop_distance_pct=sl_result.stop_pct,
            signal_strength=signal.strength,
            current_positions=self.account.open_positions,
        )

        if position.amount_usdt <= 0:
            return

        self.account.open_position(
            symbol=symbol,
            direction=signal.direction,
            entry_price=entry_price,
            size=position.amount_usdt,
            leverage=position.leverage,
            stop_loss=sl_result.stop_price,
            take_profit=tp_result.target_price,
        )

    def get_stats(self) -> dict:
        return self.account.get_stats()
