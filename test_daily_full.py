"""日线完整测试 - 寻找55%+WR"""

import yaml, pandas as pd, numpy as np
from pathlib import Path

def load_config():
    config_dir = Path("config")
    with open(config_dir / "settings.yaml", "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    with open(config_dir / "strategy.yaml", "r", encoding="utf-8") as f:
        strategy = yaml.safe_load(f)
    return {**settings, **strategy}

def run_test(config_overrides, label, symbols):
    config = load_config()
    for key, val in config_overrides.items():
        if isinstance(val, dict) and key in config and isinstance(config[key], dict):
            config[key].update(val)
        else:
            config[key] = val

    from backtest.engine import BacktestEngine

    all_stats = {}
    for sym, csv_path in symbols.items():
        df = pd.read_csv(csv_path)
        split = int(len(df) * 0.7)
        train_df = df.iloc[:split]
        test_df = df.iloc[split:]

        engine = BacktestEngine(config)
        stats = engine.run(sym, test_df, train_df)
        all_stats[sym] = stats

    # 合并结果
    all_close = []
    all_open = []
    for sym, stats in all_stats.items():
        trades = stats.get("trades", [])
        all_close.extend([t for t in trades if t.get("action") == "close"])
        all_open.extend([t for t in trades if t.get("action") == "open"])

    if not all_close:
        print(f"  {label}: 0 trades")
        return None

    wins = [t for t in all_close if t.get("pnl", 0) > 0]
    losses = [t for t in all_close if t.get("pnl", 0) <= 0]
    total_pnl = sum(t.get("pnl", 0) for t in all_close)
    wr = len(wins) / len(all_close)

    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0

    marker = " ★★★" if wr >= 0.55 and total_pnl > 0 else (" ★★" if wr >= 0.5 and total_pnl > 0 else "")
    print(f"  {label}: {len(all_open)}opens/{len(all_close)}closes WR={wr:.1%} PnL={total_pnl:+.3f}U "
          f"AvgWin={avg_win:+.3f} AvgLoss={avg_loss:+.3f}{marker}")
    return {"label": label, "trades": len(all_close), "opens": len(all_open),
            "wr": wr, "pnl": total_pnl}

btc_symbols = {"BTC/USDT:USDT": "data/BTC_1d.csv"}
eth_symbols = {"ETH/USDT:USDT": "data/ETH_1d.csv"}
both_symbols = {"BTC/USDT:USDT": "data/BTC_1d.csv", "ETH/USDT:USDT": "data/ETH_1d.csv"}

print("=" * 80)
print("日线均值回归 - 全面测试")
print("=" * 80)

tests = [
    # BTC only, 6/7
    ({"rules": {"strategy_mode": "mean_reversion", "min_conditions": 6},
      "stop_loss": {"fixed_pct": 0.04, "max_pct": 0.04},
      "take_profit": {"fixed_pct": 0.06, "partial_close_trigger": 0.04, "partial_close_pct": 0.5}},
     "BTC_6of7_sl4_tp6_partial", btc_symbols),

    # BTC only, 5/7
    ({"rules": {"strategy_mode": "mean_reversion", "min_conditions": 5},
      "stop_loss": {"fixed_pct": 0.04, "max_pct": 0.04},
      "take_profit": {"fixed_pct": 0.06, "partial_close_trigger": 0.04, "partial_close_pct": 0.5}},
     "BTC_5of7_sl4_tp6_partial", btc_symbols),

    # BTC+ETH, 6/7
    ({"rules": {"strategy_mode": "mean_reversion", "min_conditions": 6},
      "stop_loss": {"fixed_pct": 0.04, "max_pct": 0.04},
      "take_profit": {"fixed_pct": 0.06, "partial_close_trigger": 0.04, "partial_close_pct": 0.5}},
     "BTC+ETH_6of7_sl4_tp6_partial", both_symbols),

    # BTC+ETH, 5/7
    ({"rules": {"strategy_mode": "mean_reversion", "min_conditions": 5},
      "stop_loss": {"fixed_pct": 0.04, "max_pct": 0.04},
      "take_profit": {"fixed_pct": 0.06, "partial_close_trigger": 0.04, "partial_close_pct": 0.5}},
     "BTC+ETH_5of7_sl4_tp6_partial", both_symbols),

    # BTC only, 5/7, SL 3% TP 4.5%
    ({"rules": {"strategy_mode": "mean_reversion", "min_conditions": 5},
      "stop_loss": {"fixed_pct": 0.03, "max_pct": 0.03},
      "take_profit": {"fixed_pct": 0.045, "partial_close_trigger": 0.03, "partial_close_pct": 0.5}},
     "BTC_5of7_sl3_tp4.5_partial", btc_symbols),

    # BTC only, 6/7, SL 3% TP 4.5%
    ({"rules": {"strategy_mode": "mean_reversion", "min_conditions": 6},
      "stop_loss": {"fixed_pct": 0.03, "max_pct": 0.03},
      "take_profit": {"fixed_pct": 0.045, "partial_close_trigger": 0.03, "partial_close_pct": 0.5}},
     "BTC_6of7_sl3_tp4.5_partial", btc_symbols),

    # BTC only, 5/7, SL 5% TP 7.5%
    ({"rules": {"strategy_mode": "mean_reversion", "min_conditions": 5},
      "stop_loss": {"fixed_pct": 0.05, "max_pct": 0.05},
      "take_profit": {"fixed_pct": 0.075, "partial_close_trigger": 0.05, "partial_close_pct": 0.5}},
     "BTC_5of7_sl5_tp7.5_partial", btc_symbols),
]

results = []
for overrides, label, syms in tests:
    print(f"\n测试: {label}")
    try:
        r = run_test(overrides, label, syms)
        if r:
            results.append(r)
    except Exception as e:
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("结果排名:")
print("=" * 80)
results.sort(key=lambda x: (x["wr"] >= 0.55 and x["pnl"] > 0, x["pnl"]), reverse=True)
for r in results:
    marker = " ★★★" if r["wr"] >= 0.55 and r["pnl"] > 0 else (" ★★" if r["wr"] >= 0.5 and r["pnl"] > 0 else "")
    print(f"  {r['label']:<40} {r['opens']:>3}opens {r['trades']:>3}closes "
          f"WR={r['wr']:>5.1%} PnL={r['pnl']:>+7.3f}U{marker}")
