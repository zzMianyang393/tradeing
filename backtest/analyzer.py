"""回测结果分析模块"""

from typing import Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from execution.account import TradeRecord


@dataclass
class AnalysisResult:
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    total_pnl: float = 0.0
    total_return_pct: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_holding_time: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    by_symbol: dict = None
    by_direction: dict = None
    equity_curve: list = None

    def __post_init__(self):
        if self.by_symbol is None:
            self.by_symbol = {}
        if self.by_direction is None:
            self.by_direction = {}
        if self.equity_curve is None:
            self.equity_curve = []


class BacktestAnalyzer:
    def __init__(self, initial_capital: float = 10.0):
        self.initial_capital = initial_capital

    def analyze(self, trades: list[TradeRecord]) -> AnalysisResult:
        if not trades:
            return AnalysisResult()

        result = AnalysisResult()
        result.total_trades = len(trades)
        result.wins = sum(1 for t in trades if t.pnl > 0)
        result.losses = sum(1 for t in trades if t.pnl <= 0)
        result.win_rate = result.wins / result.total_trades if result.total_trades > 0 else 0

        total_wins = sum(t.pnl for t in trades if t.pnl > 0)
        total_losses = abs(sum(t.pnl for t in trades if t.pnl <= 0))
        result.profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

        win_trades = [t for t in trades if t.pnl > 0]
        loss_trades = [t for t in trades if t.pnl <= 0]

        result.avg_win = np.mean([t.pnl for t in win_trades]) if win_trades else 0
        result.avg_loss = np.mean([t.pnl for t in loss_trades]) if loss_trades else 0
        result.total_pnl = sum(t.pnl for t in trades)
        result.total_return_pct = result.total_pnl / self.initial_capital
        result.avg_trade_pnl = result.total_pnl / result.total_trades
        result.best_trade = max(t.pnl for t in trades)
        result.worst_trade = min(t.pnl for t in trades)

        holding_times = []
        for t in trades:
            if hasattr(t.open_time, "timestamp") and hasattr(t.close_time, "timestamp"):
                delta = (t.close_time - t.open_time).total_seconds() / 3600
                holding_times.append(delta)
        result.avg_holding_time = np.mean(holding_times) if holding_times else 0

        equity = [self.initial_capital]
        for t in trades:
            equity.append(equity[-1] + t.pnl)
        result.equity_curve = equity

        peak = equity[0]
        max_dd = 0
        for val in equity:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            max_dd = max(max_dd, dd)
        result.max_drawdown = max_dd

        returns = np.array([t.pnl / self.initial_capital for t in trades])
        if len(returns) > 1 and np.std(returns) > 0:
            result.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            downside = returns[returns < 0]
            if len(downside) > 0 and np.std(downside) > 0:
                result.sortino_ratio = np.mean(returns) / np.std(downside) * np.sqrt(252)

        for t in trades:
            if t.symbol not in result.by_symbol:
                result.by_symbol[t.symbol] = {"trades": 0, "wins": 0, "pnl": 0}
            result.by_symbol[t.symbol]["trades"] += 1
            if t.pnl > 0:
                result.by_symbol[t.symbol]["wins"] += 1
            result.by_symbol[t.symbol]["pnl"] += t.pnl

        for direction in ["long", "short"]:
            dir_trades = [t for t in trades if t.direction == direction]
            if dir_trades:
                result.by_direction[direction] = {
                    "trades": len(dir_trades),
                    "wins": sum(1 for t in dir_trades if t.pnl > 0),
                    "pnl": sum(t.pnl for t in dir_trades),
                }

        return result

    def print_summary(self, result: AnalysisResult):
        logger.info("=" * 60)
        logger.info("回测结果摘要")
        logger.info("=" * 60)
        logger.info(f"总交易数: {result.total_trades}")
        logger.info(f"胜率: {result.win_rate:.2%} ({result.wins}胜 / {result.losses}负)")
        logger.info(f"盈亏比: {result.profit_factor:.2f}")
        logger.info(f"总盈亏: {result.total_pnl:+.4f} USDT ({result.total_return_pct:+.2%})")
        logger.info(f"平均盈亏: {result.avg_trade_pnl:+.6f} USDT")
        logger.info(f"平均盈利: {result.avg_win:+.4f} | 平均亏损: {result.avg_loss:+.4f}")
        logger.info(f"最大回撤: {result.max_drawdown:.2%}")
        logger.info(f"夏普比率: {result.sharpe_ratio:.2f}")
        logger.info(f"Sortino比率: {result.sortino_ratio:.2f}")
        logger.info(f"最佳交易: {result.best_trade:+.4f} | 最差: {result.worst_trade:+.4f}")
        logger.info(f"平均持仓: {result.avg_holding_time:.1f}小时")

        if result.by_symbol:
            logger.info("\n按币种统计:")
            for sym, stats in sorted(
                result.by_symbol.items(), key=lambda x: x[1]["pnl"], reverse=True
            ):
                wr = stats["wins"] / stats["trades"] if stats["trades"] > 0 else 0
                logger.info(
                    f"  {sym}: {stats['trades']}笔, "
                    f"胜率={wr:.0%}, 盈亏={stats['pnl']:+.4f}U"
                )

        if result.by_direction:
            logger.info("\n按方向统计:")
            for d, stats in result.by_direction.items():
                wr = stats["wins"] / stats["trades"] if stats["trades"] > 0 else 0
                logger.info(
                    f"  {d.upper()}: {stats['trades']}笔, "
                    f"胜率={wr:.0%}, 盈亏={stats['pnl']:+.4f}U"
                )

        logger.info("=" * 60)
