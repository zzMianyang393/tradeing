"""回测报告生成模块"""

import json
from pathlib import Path
from datetime import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger

from .analyzer import AnalysisResult


class BacktestReport:
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_chart(self, result: AnalysisResult, symbol: str = "ALL") -> str:
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=("权益曲线", "单笔盈亏", "累计盈亏"),
        )

        equity = result.equity_curve
        fig.add_trace(
            go.Scatter(
                y=equity,
                mode="lines",
                name="权益",
                line=dict(color="#2196F3", width=2),
            ),
            row=1, col=1,
        )

        if result.equity_curve:
            pnl_per_trade = [
                equity[i + 1] - equity[i]
                for i in range(len(equity) - 1)
            ]
            colors = ["#4CAF50" if p > 0 else "#F44336" for p in pnl_per_trade]
            fig.add_trace(
                go.Bar(
                    y=pnl_per_trade,
                    name="单笔盈亏",
                    marker_color=colors,
                ),
                row=2, col=1,
            )

            cumulative = [0]
            for p in pnl_per_trade:
                cumulative.append(cumulative[-1] + p)
            fig.add_trace(
                go.Scatter(
                    y=cumulative,
                    mode="lines",
                    name="累计盈亏",
                    line=dict(color="#FF9800", width=2),
                ),
                row=3, col=1,
            )

        fig.update_layout(
            height=800,
            title_text=f"回测报告 - {symbol}",
            showlegend=True,
            template="plotly_dark",
        )

        file_path = self.output_dir / f"report_{symbol}_{datetime.now():%Y%m%d_%H%M%S}.html"
        fig.write_html(str(file_path))
        logger.info(f"图表报告已保存: {file_path}")
        return str(file_path)

    def generate_json(self, result: AnalysisResult, symbol: str = "ALL") -> str:
        data = {
            "summary": {
                "total_trades": result.total_trades,
                "win_rate": result.win_rate,
                "profit_factor": result.profit_factor,
                "total_pnl": result.total_pnl,
                "total_return_pct": result.total_return_pct,
                "max_drawdown": result.max_drawdown,
                "sharpe_ratio": result.sharpe_ratio,
                "sortino_ratio": result.sortino_ratio,
                "avg_holding_time": result.avg_holding_time,
            },
            "by_symbol": result.by_symbol,
            "by_direction": result.by_direction,
            "timestamp": datetime.now().isoformat(),
        }

        file_path = self.output_dir / f"report_{symbol}_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"JSON报告已保存: {file_path}")
        return str(file_path)

    def print_target_check(self, result: AnalysisResult):
        logger.info("=" * 60)
        logger.info("目标达成检查")
        logger.info("=" * 60)

        win_rate_ok = 0.55 <= result.win_rate <= 0.65
        profit_ok = result.total_pnl > 0

        logger.info(
            f"胜率目标: 60% ± 5% → 实际: {result.win_rate:.1%} "
            f"{'✓ 达标' if win_rate_ok else '✗ 未达标'}"
        )
        logger.info(
            f"盈利目标: > 0 → 实际: {result.total_pnl:+.4f}U "
            f"{'✓ 达标' if profit_ok else '✗ 未达标'}"
        )
        logger.info(
            f"盈亏比: {result.profit_factor:.2f} "
            f"{'✓' if result.profit_factor >= 1.5 else '✗'}"
        )
        logger.info(
            f"最大回撤: {result.max_drawdown:.1%} "
            f"{'✓' if result.max_drawdown < 0.3 else '⚠ 偏高'}"
        )

        all_pass = win_rate_ok and profit_ok
        logger.info(f"\n整体评估: {'✓ 全部达标' if all_pass else '✗ 需要优化'}")
        logger.info("=" * 60)
