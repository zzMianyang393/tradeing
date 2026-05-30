"""测试趋势跟踪策略 - 4h时间框架"""

import yaml, pandas as pd, numpy as np
from pathlib import Path

def load_config():
    config_dir = Path("config")
    with open(config_dir / "settings.yaml", "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    with open(config_dir / "strategy.yaml", "r", encoding="utf-8") as f:
        strategy = yaml.safe_load(f)
    return {**settings, **strategy}

def run_test(config_overrides, label, csv_path):
    config = load_config()
    for key, val in config_overrides.items():
        if isinstance(val, dict) and key in config and isinstance(config[key], dict):
            config[key].update(val)
        else:
            config[key] = val

    from backtest.engine import BacktestEngine

    df = pd.read_csv(csv_path)
    split = int(len(df) * 0.7)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    engine = BacktestEngine(config)
    stats = engine.run("BTC/USDT:USDT", test_df, train_df)

    trades = stats.get("trades", [])
    close_trades = [t for t in trades if t.get("action") == "close"]
    open_trades = [t for t in trades if t.get("action") == "open"]

    if not close_trades:
        print(f"  {label}: 0 trades")
        return None

    wins = [t for t in close_trades if t.get("pnl", 0) > 0]
    losses = [t for t in close_trades if t.get("pnl", 0) <= 0]
    total_pnl = sum(t.get("pnl", 0) for t in close_trades)
    wr = len(wins) / len(close_trades)

    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0

    marker = " ★★★" if wr >= 0.55 and total_pnl > 0 else (" ★★" if wr >= 0.5 and total_pnl > 0 else "")
    print(f"  {label}: {len(open_trades)}opens/{len(close_trades)}closes WR={wr:.1%} PnL={total_pnl:+.3f}U "
          f"AvgWin={avg_win:+.3f} AvgLoss={avg_loss:+.3f}{marker}")
    return {"label": label, "trades": len(close_trades), "opens": len(open_trades),
            "wr": wr, "pnl": total_pnl}

print("=" * 80)
print("趋势跟踪策略 - 4h时间框架 - BTC Only")
print("=" * 80)

tests = [
    # 趋势策略 + 4h + 6/7条件
    ({"rules": {"strategy_mode": "trend", "min_conditions": 6}}, "trend_4h_6of7"),

    # 趋势策略 + 4h + 5/7条件
    ({"rules": {"strategy_mode": "trend", "min_conditions": 5}}, "trend_4h_5of7"),

    # 趋势策略 + 4h + 5/7 + 更宽SL/TP
    ({"rules": {"strategy_mode": "trend", "min_conditions": 5},
      "stop_loss": {"fixed_pct": 0.03, "max_pct": 0.03},
      "take_profit": {"fixed_pct": 0.06}}, "trend_4h_5of7_sl3_tp6"),

    # 趋势策略 + 4h + 5/7 + SL 3% TP 4.5%
    ({"rules": {"strategy_mode": "trend", "min_conditions": 5},
      "stop_loss": {"fixed_pct": 0.03, "max_pct": 0.03},
      "take_profit": {"fixed_pct": 0.045}}, "trend_4h_5of7_sl3_tp4.5"),

    # 趋势策略 + 4h + 6/7 + SL 3% TP 6%
    ({"rules": {"strategy_mode": "trend", "min_conditions": 6},
      "stop_loss": {"fixed_pct": 0.03, "max_pct": 0.03},
      "take_profit": {"fixed_pct": 0.06}}, "trend_4h_6of7_sl3_tp6"),

    # 趋势策略 + 4h + 5/7 + 部分止盈
    ({"rules": {"strategy_mode": "trend", "min_conditions": 5},
      "stop_loss": {"fixed_pct": 0.03, "max_pct": 0.03},
      "take_profit": {"fixed_pct": 0.06, "partial_close_trigger": 0.04, "partial_close_pct": 0.5}},
     "trend_4h_5of7_sl3_tp6_partial4"),

    # 趋势策略 + 4h + 6/7 + 部分止盈
    ({"rules": {"strategy_mode": "trend", "min_conditions": 6},
      "stop_loss": {"fixed_pct": 0.03, "max_pct": 0.03},
      "take_profit": {"fixed_pct": 0.06, "partial_close_trigger": 0.04, "partial_close_pct": 0.5}},
     "trend_4h_6of7_sl3_tp6_partial4"),

    # 趋势策略 + 4h + 5/7 + SL 2.5% TP 5%
    ({"rules": {"strategy_mode": "trend", "min_conditions": 5},
      "stop_loss": {"fixed_pct": 0.025, "max_pct": 0.025},
      "take_profit": {"fixed_pct": 0.05}}, "trend_4h_5of7_sl2.5_tp5"),
]

results = []
for overrides, label in tests:
    print(f"\n测试: {label}")
    try:
        r = run_test(overrides, label, "data/BTC_4h.csv")
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
