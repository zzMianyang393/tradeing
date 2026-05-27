"""账户管理模块"""

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
    def __init__(self, initial_capital: float = 10.0):
        self.initial_capital = initial_capital
        self.balance = initial_capital
        self.positions: dict[str, Position] = {}
        self.trade_history: list[TradeRecord] = []
        self.daily_pnl = 0.0
        self.total_pnl = 0.0

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

        emoji = "+" if pnl >= 0 else ""
        logger.info(
            f"平仓: {symbol} {pos.direction.upper()} "
            f"入场={pos.entry_price}, 出场={exit_price}, "
            f"盈亏={emoji}{pnl:.4f}U ({emoji}{pnl_pct:.2%}) | {reason}"
        )
        return record

    def get_stats(self) -> dict:
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
        }

    def reset_daily_pnl(self):
        self.daily_pnl = 0.0
