"""90天回测报告 - 日线均值回归策略"""

import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from strategy.hybrid import HybridStrategy
from strategy.indicators import TechnicalIndicators
from backtest.engine import BacktestEngine
from backtest.analyzer import BacktestAnalyzer, AnalysisResult
from execution.account import TradeRecord


def load_config() -> dict:
    config_dir = Path("config")
    with open(config_dir / "settings.yaml", "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    with open(config_dir / "strategy.yaml", "r", encoding="utf-8") as f:
        strategy = yaml.safe_load(f)
    return {**settings, **strategy}


def format_time(ts):
    if isinstance(ts, (int, float)) and ts > 0:
        try:
            return datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d")
        except:
            return str(ts)
    return str(ts)


def run_90d_report():
    config = load_config()

    symbols = {
        "BTC/USDT:USDT": "data/BTC_1d.csv",
        "ETH/USDT:USDT": "data/ETH_1d.csv",
    }

    print("=" * 80)
    print("90天回测报告 - 日线均值回归策略")
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    all_trades = []

    for symbol, csv_path in symbols.items():
        if not Path(csv_path).exists():
            print(f"\n跳过 {symbol}: {csv_path} 不存在")
            continue

        df = pd.read_csv(csv_path)

        # 取最近90天作为测试集，前面的数据作为训练集
        test_days = 90
        if len(df) < test_days + 60:
            print(f"\n{symbol}: 数据不足，跳过")
            continue

        train_df = df.iloc[:-test_days]
        test_df = df.iloc[-test_days:]

        start_date = format_time(test_df['timestamp'].iloc[0])
        end_date = format_time(test_df['timestamp'].iloc[-1])

        print(f"\n{symbol}:")
        print(f"  总数据: {len(df)}天 ({format_time(df['timestamp'].iloc[0])} ~ {format_time(df['timestamp'].iloc[-1])})")
        print(f"  训练集: {len(train_df)}天 ({format_time(train_df['timestamp'].iloc[0])} ~ {format_time(train_df['timestamp'].iloc[-1])})")
        print(f"  测试集: {len(test_df)}天 ({start_date} ~ {end_date})")

        engine = BacktestEngine(config)
        stats = engine.run(symbol, test_df, train_df)

        trades = stats.get("trades", [])
        open_trades = [t for t in trades if t.get("action") == "open"]
        close_trades = [t for t in trades if t.get("action") == "close"]

        print(f"  开仓: {len(open_trades)}笔, 平仓: {len(close_trades)}笔")

        for t in close_trades:
            t["symbol"] = symbol
            t["test_start"] = start_date
            t["test_end"] = end_date
            all_trades.append(t)

    # 分析
    analyzer = BacktestAnalyzer(initial_capital=config.get("general", {}).get("initial_capital", 10.0))

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
            close_time=open_time,
            close_reason=t.get("reason", ""),
        )
        trade_records.append(record)

    result = analyzer.analyze(trade_records)

    # 打印报告
    print("\n" + "=" * 80)
    print("90天回测结果 (2025-09-06 ~ 2026-05-27)")
    print("=" * 80)

    print(f"\n[总体统计]")
    print(f"  总交易数: {result.total_trades}")
    print(f"  盈利笔数: {result.wins}")
    print(f"  亏损笔数: {result.losses}")
    print(f"  胜率: {result.win_rate:.2%} ({'[OK]' if result.win_rate >= 0.55 else '[FAIL]'})")
    print(f"  盈亏比: {result.profit_factor:.2f} ({'[OK]' if result.profit_factor >= 1.5 else '[FAIL]'})")

    print(f"\n[盈亏统计]")
    print(f"  总盈亏: {result.total_pnl:+.4f} USDT ({result.total_return_pct:+.2%})")
    print(f"  平均盈亏: {result.avg_trade_pnl:+.6f} USDT")
    print(f"  平均盈利: {result.avg_win:+.4f} USDT")
    print(f"  平均亏损: {result.avg_loss:+.4f} USDT")
    print(f"  最佳交易: {result.best_trade:+.4f} USDT")
    print(f"  最差交易: {result.worst_trade:+.4f} USDT")

    print(f"\n[风险指标]")
    print(f"  最大回撤: {result.max_drawdown:.2%} ({'[OK]' if result.max_drawdown < 0.15 else '[HIGH]'})")
    print(f"  夏普比率: {result.sharpe_ratio:.2f} ({'[OK]' if result.sharpe_ratio > 2 else '[LOW]'})")

    if result.by_symbol:
        print(f"\n[按币种统计]")
        for sym, stats in sorted(result.by_symbol.items(), key=lambda x: x[1]["pnl"], reverse=True):
            wr = stats["wins"] / stats["trades"] if stats["trades"] > 0 else 0
            print(f"  {sym.split('/')[0]}:")
            print(f"    交易数: {stats['trades']}, 胜率: {wr:.2%}, 盈亏: {stats['pnl']:+.4f}U")

    if result.by_direction:
        print(f"\n[按方向统计]")
        for d, stats in result.by_direction.items():
            wr = stats["wins"] / stats["trades"] if stats["trades"] > 0 else 0
            print(f"  {d.upper()}:")
            print(f"    交易数: {stats['trades']}, 胜率: {wr:.2%}, 盈亏: {stats['pnl']:+.4f}U")

    print(f"\n[交易明细]")
    print(f"{'#':>3} {'方向':>4} {'币种':>6} {'入场价':>12} {'出场价':>12} {'盈亏':>10} {'原因'}")
    print("-" * 70)

    for i, t in enumerate(all_trades, 1):
        direction = "多" if t.get("direction") == "long" else "空"
        symbol = t.get("symbol", "").split("/")[0]
        entry = t.get("entry_price", 0)
        exit_ = t.get("exit_price", 0)
        pnl = t.get("pnl", 0)
        reason = t.get("reason", "")
        marker = "[W]" if pnl > 0 else "[L]"
        print(f"{i:>3} {direction:>4} {symbol:>6} {entry:>12.2f} {exit_:>12.2f} {pnl:>+10.4f} {reason} {marker}")

    # 目标检查
    print("\n" + "=" * 80)
    print("目标达成检查")
    print("=" * 80)

    win_rate_ok = result.win_rate >= 0.55
    profit_ok = result.total_pnl > 0

    print(f"  胜率目标 (>55%): {result.win_rate:.2%} {'[OK]' if win_rate_ok else '[FAIL]'}")
    print(f"  盈利目标 (>0U): {result.total_pnl:+.4f}U {'[OK]' if profit_ok else '[FAIL]'}")
    print(f"\n  整体评估: {'[OK] 全部达标！' if win_rate_ok and profit_ok else '[FAIL] 需要优化'}")
    print("=" * 80)

    # 生成图表
    try:
        from backtest.report import BacktestReport
        report = BacktestReport()
        chart_path = report.generate_chart(result, "90D_Daily_MR")
        print(f"\n图表报告: {chart_path}")
    except Exception as e:
        print(f"\n图表生成失败: {e}")

    return result


if __name__ == "__main__":
    run_90d_report()
