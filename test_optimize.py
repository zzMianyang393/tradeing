"""快速参数优化 - BTC only, 重点突破55%WR"""

import yaml, pandas as pd, numpy as np
from pathlib import Path

def load_config():
    config_dir = Path("config")
    with open(config_dir / "settings.yaml", "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    with open(config_dir / "strategy.yaml", "r", encoding="utf-8") as f:
        strategy = yaml.safe_load(f)
    return {**settings, **strategy}

def run_test(config_overrides, label):
    config = load_config()
    # Deep merge: only override specified nested keys
    for key, val in config_overrides.items():
        if isinstance(val, dict) and key in config and isinstance(config[key], dict):
            config[key].update(val)
        else:
            config[key] = val

    from strategy.hybrid import HybridStrategy
    from strategy.indicators import TechnicalIndicators
    from backtest.engine import BacktestEngine

    csv_path = "data/BTC_15m.csv"
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
        return

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
            "wr": wr, "pnl": total_pnl, "avg_win": avg_win, "avg_loss": avg_loss}

tests = [
    # 基线: 6/7条件
    ({"rules": {"min_conditions": 6}}, "baseline_6of7"),

    # 放宽长仓到5/7
    ({"rules": {"min_conditions": 5, "long_conditions_min": 5}}, "long_5of7"),

    # 放宽长仓到5/7 + 更宽SL
    ({"rules": {"min_conditions": 5, "long_conditions_min": 5},
      "stop_loss": {"fixed_pct": 0.03, "max_pct": 0.03},
      "take_profit": {"fixed_pct": 0.045, "risk_reward_ratio": 1.5}}, "long5_wider_sl3_tp4.5"),

    # 放宽长仓到5/7 + 更宽SL + 部分止盈
    ({"rules": {"min_conditions": 5, "long_conditions_min": 5},
      "stop_loss": {"fixed_pct": 0.028, "max_pct": 0.028},
      "take_profit": {"fixed_pct": 0.042, "risk_reward_ratio": 1.5,
                      "partial_close_trigger": 0.028, "partial_close_pct": 0.5}}, "long5_sl2.8_tp4.2_partial2.8"),

    # 放宽长仓到5/7 + 降低cooldown到5
    ({"rules": {"min_conditions": 5, "long_conditions_min": 5}}, "long5_cooldown5"),

    # 更严格: 7/7条件
    ({"rules": {"min_conditions": 7, "long_conditions_min": 7}}, "strict_7of7"),

    # 6/7条件 + 更紧SL
    ({"rules": {"min_conditions": 6},
      "stop_loss": {"fixed_pct": 0.02, "max_pct": 0.02},
      "take_profit": {"fixed_pct": 0.04, "risk_reward_ratio": 2.0}}, "sl2_tp4_rr2"),

    # 6/7条件 + 更紧SL + 部分止盈更激进
    ({"rules": {"min_conditions": 6},
      "stop_loss": {"fixed_pct": 0.02, "max_pct": 0.02},
      "take_profit": {"fixed_pct": 0.035, "risk_reward_ratio": 1.75,
                      "partial_close_trigger": 0.025, "partial_close_pct": 0.6}}, "sl2_tp3.5_partial2.5_60pct"),

    # 6/7 + 部分止盈更激进(70%在2.5%时止盈)
    ({"rules": {"min_conditions": 6},
      "take_profit": {"fixed_pct": 0.0375, "risk_reward_ratio": 1.5,
                      "partial_close_trigger": 0.025, "partial_close_pct": 0.7}}, "partial2.5_70pct"),

    # 6/7 + 部分止盈50%在2%时 (更保守)
    ({"rules": {"min_conditions": 6},
      "take_profit": {"fixed_pct": 0.0375, "risk_reward_ratio": 1.5,
                      "partial_close_trigger": 0.02, "partial_close_pct": 0.5}}, "partial2_50pct"),

    # 5/7 + SL 2.2% + TP 3.3% (1.5:1)
    ({"rules": {"min_conditions": 5, "long_conditions_min": 5},
      "stop_loss": {"fixed_pct": 0.022, "max_pct": 0.022},
      "take_profit": {"fixed_pct": 0.033, "risk_reward_ratio": 1.5}}, "long5_sl2.2_tp3.3"),

    # 5/7 + SL 2% + TP 3% + 部分止盈2%
    ({"rules": {"min_conditions": 5, "long_conditions_min": 5},
      "stop_loss": {"fixed_pct": 0.02, "max_pct": 0.02},
      "take_profit": {"fixed_pct": 0.03, "risk_reward_ratio": 1.5,
                      "partial_close_trigger": 0.02, "partial_close_pct": 0.5}}, "long5_sl2_tp3_partial2"),
]

print("=" * 80)
print("BTC Only 参数优化")
print("=" * 80)

results = []
for overrides, label in tests:
    print(f"\n测试: {label}")
    try:
        r = run_test(overrides, label)
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
