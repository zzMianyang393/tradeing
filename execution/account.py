"""账户管理模块"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

from loguru import logger



@dataclass
class Position:
    symbol: str
    direction: str  # "long" or "short"
    entry_price: float
    size: float  # USDT notional
    leverage: int
    stop_loss: float
    take_profit: float
    open_time: datetime
    partial_closed: bool = False

    @property
    def margin(self) -> float:
        return self.size / self.leverage


@dataclass
class TradeRecord:
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    size: float
    leverage: int
    pnl: float
    pnl_pct: float
    open_time: datetime
    close_time: datetime
    close_reason: str


class AccountManager:
    def __init__(self, initial_capital: float = 10.0, storage=None):
        self.initial_capital = initial_capital
        self.balance = initial_capital
        self.positions: dict[str, Position] = {}
        self.trade_history: list[TradeRecord] = []
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self._storage = storage
        if storage is not None:
            self._load_history()

    def _load_history(self):
        """从数据库加载历史交易，恢复 total_pnl"""
        try:
            trades = self._storage.load_trades()
            if trades:
                self.trade_history = trades
                self.total_pnl = sum(t.pnl for t in trades)
                logger.info(f"从数据库加载了 {len(trades)} 笔历史交易，累计盈亏={self.total_pnl:.4f}U")
        except Exception as e:
            logger.warning(f"加载历史交易失败: {e}")

    @property
    def open_positions(self) -> int:
        return len(self.positions)

    @property
    def unrealized_pnl(self) -> float:
        return sum(
            pos.size / pos.leverage * 0.01  # placeholder
            for pos in self.positions.values()
        )

    def open_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        size: float,
        leverage: int,
        stop_loss: float,
        take_profit: float,
    ) -> Optional[Position]:
        margin = size / leverage
        fee = size * 0.0005  # taker fee

        if margin + fee > self.balance:
            logger.warning(
                f"资金不足: 需要 {margin + fee:.4f} USDT, "
                f"可用 {self.balance:.4f} USDT"
            )
            return None

        self.balance -= (margin + fee)

        position = Position(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            size=size,
            leverage=leverage,
            stop_loss=stop_loss,
            take_profit=take_profit,
            open_time=datetime.utcnow(),
        )
        self.positions[symbol] = position

        logger.info(
            f"开仓: {symbol} {direction.upper()} "
            f"价格={entry_price}, 金额={size:.4f}U, "
            f"杠杆={leverage}x, 止损={stop_loss:.8f}, 止盈={take_profit:.8f}"
        )
        return position

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str = "manual",
    ) -> Optional[TradeRecord]:
        if symbol not in self.positions:
            return None

        pos = self.positions.pop(symbol)

        if pos.direction == "long":
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price

        pnl_pct *= pos.leverage
        pnl = pos.size * pnl_pct

        fee = pos.size * 0.0005
        pnl -= fee

        self.balance += pos.size / pos.leverage + pnl

        record = TradeRecord(
            symbol=symbol,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size=pos.size,
            leverage=pos.leverage,
            pnl=round(pnl, 6),
            pnl_pct=round(pnl_pct, 6),
            open_time=pos.open_time,
            close_time=datetime.utcnow(),
            close_reason=reason,
        )
        self.trade_history.append(record)
        self.daily_pnl += pnl
        self.total_pnl += pnl

        # 持久化到数据库
        if self._storage is not None:
            try:
                self._storage.save_trade(record)
            except Exception as e:
                logger.warning(f"保存交易记录失败: {e}")

        emoji = "+" if pnl >= 0 else ""
        logger.info(
            f"平仓: {symbol} {pos.direction.upper()} "
            f"入场={pos.entry_price}, 出场={exit_price}, "
            f"盈亏={emoji}{pnl:.4f}U ({emoji}{pnl_pct:.2%}) | {reason}"
        )
        return record

    def get_stats(self) -> dict:
        import math

        if not self.trade_history:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "avg_pnl": 0,
                "total_pnl": 0,
                "balance": self.balance,
            }

        wins = [t for t in self.trade_history if t.pnl > 0]
        losses = [t for t in self.trade_history if t.pnl <= 0]

        # Trade returns for Sharpe calculation (pnl_pct from each trade)
        trade_returns = [t.pnl_pct for t in self.trade_history]

        # Annualized Sharpe ratio: sqrt(252*96) * mean(returns) / std(returns)
        # 96 bars per day on 15m timeframe
        sharpe = 0.0
        if len(trade_returns) >= 2:
            mean_ret = sum(trade_returns) / len(trade_returns)
            var = sum((r - mean_ret) ** 2 for r in trade_returns) / (len(trade_returns) - 1)
            std_ret = math.sqrt(var) if var > 0 else 0
            if std_ret > 0:
                sharpe = math.sqrt(252 * 96) * mean_ret / std_ret

        # Max drawdown from balance curve
        max_dd = 0.0
        peak = self.initial_capital
        running_balance = self.initial_capital
        for t in self.trade_history:
            running_balance += t.pnl
            if running_balance > peak:
                peak = running_balance
            dd = (peak - running_balance) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        return {
            "total_trades": len(self.trade_history),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(self.trade_history),
            "avg_pnl": sum(t.pnl for t in self.trade_history) / len(self.trade_history),
            "total_pnl": self.total_pnl,
            "total_return_pct": self.total_pnl / self.initial_capital,
            "balance": self.balance,
            "max_win": max((t.pnl for t in self.trade_history), default=0),
            "max_loss": min((t.pnl for t in self.trade_history), default=0),
            "avg_win": sum(t.pnl for t in wins) / len(wins) if wins else 0,
            "avg_loss": sum(t.pnl for t in losses) / len(losses) if losses else 0,
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown": round(max_dd, 6),
        }

    def reset_daily_pnl(self):
        self.daily_pnl = 0.0
