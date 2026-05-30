"""完整回测报告 - 日线均值回归策略"""

import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from strategy.hybrid import HybridStrategy
from strategy.indicators import TechnicalIndicators
from backtest.engine import BacktestEngine
from backtest.analyzer import BacktestAnalyzer, AnalysisResult
from backtest.report import BacktestReport


def load_config() -> dict:
    config_dir = Path("config")
    with open(config_dir / "settings.yaml", "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    with open(config_dir / "strategy.yaml", "r", encoding="utf-8") as f:
        strategy = yaml.safe_load(f)
    return {**settings, **strategy}


def format_time(ts):
    """格式化时间戳"""
    if isinstance(ts, (int, float)) and ts > 0:
        try:
            return datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M")
        except:
            return str(ts)
    return str(ts)


def run_backtest_report():
    config = load_config()

    symbols = {
        "BTC/USDT:USDT": "data/BTC_1d.csv",
        "ETH/USDT:USDT": "data/ETH_1d.csv",
    }

    print("=" * 80)
    print("回测报告 - 日线均值回归策略")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 运行回测
    all_trades = []
    all_stats = {}

    for symbol, csv_path in symbols.items():
        if not Path(csv_path).exists():
            print(f"\n跳过 {symbol}: {csv_path} 不存在")
            continue

        df = pd.read_csv(csv_path)
        split = int(len(df) * 0.7)
        train_df = df.iloc[:split]
        test_df = df.iloc[split:]

        print(f"\n{symbol}:")
        print(f"  总K线: {len(df)}, 训练集: {len(train_df)}, 测试集: {len(test_df)}")
        print(f"  测试期: {format_time(test_df['timestamp'].iloc[0])} ~ {format_time(test_df['timestamp'].iloc[-1])}")

        engine = BacktestEngine(config)
        stats = engine.run(symbol, test_df, train_df)
        all_stats[symbol] = stats

        trades = stats.get("trades", [])
        open_trades = [t for t in trades if t.get("action") == "open"]
        close_trades = [t for t in trades if t.get("action") == "close"]

        print(f"  开仓: {len(open_trades)}笔, 平仓: {len(close_trades)}笔")

        # 记录平仓交易
        for t in close_trades:
            t["symbol"] = symbol
            all_trades.append(t)

    # 分析结果
    analyzer = BacktestAnalyzer(initial_capital=config.get("general", {}).get("initial_capital", 10.0))

    # 转换为TradeRecord格式
    from execution.account import TradeRecord
    trade_records = []
    for t in all_trades:
        try:
            open_time = datetime.fromisoformat(t.get("time", "2000-01-01").replace("Z", "+00:00"))
        except:
            open_time = datetime.now()

        record = TradeRecord(
            symbol=t.get("symbol", ""),
            direction=t.get("direction", "long"),
            entry_price=t.get("entry_price", 0),
            exit_price=t.get("exit_price", 0),
            size=t.get("size", 0),
            leverage=t.get("leverage", 1),
            pnl=t.get("pnl", 0),
            pnl_pct=t.get("pnl_pct", 0),
            open_time=open_time,
            close_time=open_time,  # 简化
            close_reason=t.get("reason", ""),
        )
        trade_records.append(record)

    result = analyzer.analyze(trade_records)

    # 打印详细报告
    print("\n" + "=" * 80)
    print("详细回测结果")
    print("=" * 80)

    print(f"\n【总体统计】")
    print(f"  总交易数: {result.total_trades}")
    print(f"  盈利笔数: {result.wins}")
    print(f"  亏损笔数: {result.losses}")
    print(f"  胜率: {result.win_rate:.2%} ({'[OK]' if result.win_rate >= 0.55 else '[FAIL]'})")
    print(f"  盈亏比: {result.profit_factor:.2f} ({'[OK]' if result.profit_factor >= 1.5 else '[FAIL]'})")

    print(f"\n【盈亏统计】")
    print(f"  总盈亏: {result.total_pnl:+.4f} USDT ({result.total_return_pct:+.2%})")
    print(f"  平均盈亏: {result.avg_trade_pnl:+.6f} USDT")
    print(f"  平均盈利: {result.avg_win:+.4f} USDT")
    print(f"  平均亏损: {result.avg_loss:+.4f} USDT")
    print(f"  最佳交易: {result.best_trade:+.4f} USDT")
    print(f"  最差交易: {result.worst_trade:+.4f} USDT")

    print(f"\n【风险指标】")
    print(f"  最大回撤: {result.max_drawdown:.2%} ({'[OK]' if result.max_drawdown < 0.15 else '[HIGH]'})")
    print(f"  夏普比率: {result.sharpe_ratio:.2f} ({'[OK]' if result.sharpe_ratio > 2 else '[LOW]'})")
    print(f"  Sortino比率: {result.sortino_ratio:.2f}")

    # 按币种统计
    if result.by_symbol:
        print(f"\n【按币种统计】")
        for sym, stats in sorted(result.by_symbol.items(), key=lambda x: x[1]["pnl"], reverse=True):
            wr = stats["wins"] / stats["trades"] if stats["trades"] > 0 else 0
            print(f"  {sym}:")
            print(f"    交易数: {stats['trades']}")
            print(f"    胜率: {wr:.2%}")
            print(f"    盈亏: {stats['pnl']:+.4f} USDT")

    # 按方向统计
    if result.by_direction:
        print(f"\n【按方向统计】")
        for d, stats in result.by_direction.items():
            wr = stats["wins"] / stats["trades"] if stats["trades"] > 0 else 0
            print(f"  {d.upper()}:")
            print(f"    交易数: {stats['trades']}")
            print(f"    胜率: {wr:.2%}")
            print(f"    盈亏: {stats['pnl']:+.4f} USDT")

    # 交易明细
    print(f"\n【交易明细】")
    print(f"{'序号':>4} {'方向':>6} {'币种':>15} {'入场价':>12} {'出场价':>12} {'盈亏':>10} {'原因'}")
    print("-" * 80)

    for i, t in enumerate(all_trades, 1):
        direction = "多" if t.get("direction") == "long" else "空"
        symbol = t.get("symbol", "").split("/")[0]
        entry = t.get("entry_price", 0)
        exit_ = t.get("exit_price", 0)
        pnl = t.get("pnl", 0)
        reason = t.get("reason", "")

        pnl_str = f"{pnl:+.4f}"
        marker = "[W]" if pnl > 0 else "[L]"

        print(f"{i:>4} {direction:>6} {symbol:>15} {entry:>12.2f} {exit_:>12.2f} {pnl_str:>10} {reason} {marker}")

    # 目标达成检查
    print("\n" + "=" * 80)
    print("目标达成检查")
    print("=" * 80)

    win_rate_ok = result.win_rate >= 0.55
    profit_ok = result.total_pnl > 0

    print(f"  胜率目标 (>55%): {result.win_rate:.2%} {'[OK]' if win_rate_ok else '[FAIL]'}")
    print(f"  盈利目标 (>0U): {result.total_pnl:+.4f}U {'[OK]' if profit_ok else '[FAIL]'}")
    print(f"\n  整体评估: {'[OK] 全部达标！目标已实现！' if win_rate_ok and profit_ok else '[FAIL] 需要优化'}")
    print("=" * 80)

    # 生成图表报告
    try:
        report = BacktestReport()
        chart_path = report.generate_chart(result, "Daily_MR_BTC_ETH")
        print(f"\n图表报告已保存: {chart_path}")
    except Exception as e:
        print(f"\n图表生成失败: {e}")

    return result


if __name__ == "__main__":
    run_backtest_report()
