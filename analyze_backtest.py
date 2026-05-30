"""回测分析 - 正确统计开仓数据"""

import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from data.fetcher import DataFetcher
from data.storage import DataStorage
from data.hot_coins import HotCoinSelector
from strategy.hybrid import HybridStrategy
from backtest.engine import BacktestEngine
from backtest.analyzer import BacktestAnalyzer


def load_config() -> dict:
    config_dir = Path("config")
    with open(config_dir / "settings.yaml", "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    with open(config_dir / "strategy.yaml", "r", encoding="utf-8") as f:
        strategy = yaml.safe_load(f)
    return {**settings, **strategy}


def run_backtest(config: dict, days: int = 90):
    storage = DataStorage()
    fetcher = DataFetcher(storage)

    csv_files = {
        "BTC/USDT:USDT": "data/BTC_1d.csv",
        "ETH/USDT:USDT": "data/ETH_1d.csv",
    }

    all_data = {}
    train_data = {}

    for symbol, csv_path in csv_files.items():
        if Path(csv_path).exists():
            df = pd.read_csv(csv_path)
            print(f"{symbol}: 从CSV加载 {len(df)} 条K线")

            if len(df) > 100:
                split = int(len(df) * 0.7)
                train_data[symbol] = df.iloc[:split]
                all_data[symbol] = df.iloc[split:]
                print(f"  训练集: {split}条, 测试集: {len(df)-split}条")
        else:
            print(f"{symbol}: CSV不存在，跳过")

    if not all_data:
        print("没有可用数据")
        return

    engine = BacktestEngine(config)
    stats = engine.run_multi(all_data, train_data)

    print("\n" + "=" * 60)
    print("回测结果分析")
    print("=" * 60)

    trades = stats.get("trades", [])
    if not trades:
        print("没有交易记录")
        return

    # 分离 open 和 close 交易
    open_trades = [t for t in trades if t.get("action") == "open"]
    close_trades = [t for t in trades if t.get("action") == "close"]

    print(f"\n总K线数: {stats.get('total_bars', 0)}")
    print(f"总开仓次数: {len(open_trades)}")
    print(f"总平仓次数: {len(close_trades)}")

    # 按月份统计开仓
    monthly_stats = defaultdict(lambda: {"count": 0, "sizes": [], "symbols": defaultdict(int)})

    for t in open_trades:
        time_str = t.get("time", "")
        if time_str:
            try:
                dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
                month_key = dt.strftime("%Y-%m")
                monthly_stats[month_key]["count"] += 1
                monthly_stats[month_key]["sizes"].append(t.get("size", 0))
                monthly_stats[month_key]["symbols"][t.get("symbol", "")] += 1
            except:
                pass

    print("\n--- 按月统计（仅开仓） ---")
    print(f"{'月份':<10} {'开仓次数':>10} {'平均仓位':>12} {'最大仓位':>12} {'BTC':>6} {'ETH':>6}")
    print("-" * 65)

    for month in sorted(monthly_stats.keys()):
        stats_m = monthly_stats[month]
        count = stats_m["count"]
        sizes = stats_m["sizes"]
        avg_size = sum(sizes) / len(sizes) if sizes else 0
        max_size = max(sizes) if sizes else 0
        btc = stats_m["symbols"].get("BTC/USDT:USDT", 0)
        eth = stats_m["symbols"].get("ETH/USDT:USDT", 0)
        print(f"{month:<10} {count:>10} {avg_size:>12.4f} {max_size:>12.4f} {btc:>6} {eth:>6}")

    # 分析最大仓位附近的交易
    if open_trades:
        sizes = [t.get("size", 0) for t in open_trades]
        max_size = max(sizes)
        sorted_sizes = sorted(sizes, reverse=True)

        print(f"\n--- 仓位大小分析 ---")
        print(f"最大仓位: {max_size:.4f} USDT")
        print(f"最小仓位: {min(sizes):.4f} USDT")
        print(f"平均仓位: {sum(sizes)/len(sizes):.4f} USDT")
        print(f"\n最大仓位附近的10个交易:")
        for i, s in enumerate(sorted_sizes[:10]):
            print(f"  {i+1}. {s:.4f} USDT")

    # 分析交易结果
    if close_trades:
        wins = [t for t in close_trades if t.get("pnl", 0) > 0]
        losses = [t for t in close_trades if t.get("pnl", 0) <= 0]

        total_pnl = sum(t.get("pnl", 0) for t in close_trades)
        win_rate = len(wins) / len(close_trades) * 100 if close_trades else 0

        print(f"\n--- 交易结果 ---")
        print(f"胜率: {win_rate:.1f}% ({len(wins)}胜 / {len(losses)}负)")
        print(f"总盈亏: {total_pnl:.4f} USDT")
        print(f"平均每笔: {total_pnl/len(close_trades):.4f} USDT")

        if wins:
            avg_win = sum(t.get("pnl", 0) for t in wins) / len(wins)
            print(f"平均盈利: {avg_win:.4f} USDT")
        if losses:
            avg_loss = sum(t.get("pnl", 0) for t in losses) / len(losses)
            print(f"平均亏损: {avg_loss:.4f} USDT")


if __name__ == "__main__":
    config = load_config()
    run_backtest(config, days=90)
