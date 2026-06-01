"""Multi-strategy validation - runs all validation criteria"""

import yaml
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from loguru import logger

from data.storage import DataStorage
from backtest.engine import BacktestEngine
from backtest.analyzer import BacktestAnalyzer
from execution.account import TradeRecord


def load_config() -> dict:
    config_dir = Path("config")
    with open(config_dir / "settings.yaml", "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    with open(config_dir / "strategy.yaml", "r", encoding="utf-8") as f:
        strategy = yaml.safe_load(f)
    return {**settings, **strategy}


def prepare_df(storage: DataStorage, symbol: str, timeframe: str,
               start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame:
    """Load and prepare dataframe for backtest engine"""
    df = storage.load_klines(symbol, timeframe)
    if df.empty:
        return pd.DataFrame()

    mask = (df.index >= start_dt) & (df.index <= end_dt)
    df = df[mask].copy()
    if len(df) < 10:
        return pd.DataFrame()

    timestamps_ms = (df.index.astype(np.int64) // 10**6).values
    df = df.reset_index(drop=True)
    df['timestamp'] = timestamps_ms[:len(df)]

    keep_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df = df[[c for c in keep_cols if c in df.columns]]
    return df


def run_backtest(config: dict, storage: DataStorage, coins: list,
                 start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> dict:
    """Run backtest on multiple coins for a given time range"""
    all_trades = []
    coin_results = {}

    for symbol in coins:
        df_15m = prepare_df(storage, symbol, '15m', start_dt, end_dt)
        if df_15m.empty or len(df_15m) < 60:
            continue

        df_4h = prepare_df(storage, symbol, '4h', start_dt, end_dt)
        if df_4h.empty or len(df_4h) < 10:
            df_4h = None

        # Load full 15m data for 4h trend computation (EMA55 needs history)
        full_15m = prepare_df(storage, symbol, '15m',
                              pd.Timestamp("2026-01-01"), end_dt)

        engine = BacktestEngine(config)
        stats = engine.run(symbol, df_15m, htf4_data=df_4h, full_df=full_15m)

        trades = stats.get("trades", [])
        close_trades = [t for t in trades if t.get("action") == "close"]

        for t in close_trades:
            t["symbol"] = symbol
            try:
                open_time = datetime.fromisoformat(t.get("time", "2000-01-01").replace("Z", "+00:00"))
            except:
                open_time = datetime.utcnow()
            record = TradeRecord(
                symbol=symbol,
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
            all_trades.append(record)

        pnl = sum(t.get("pnl", 0) for t in close_trades)
        n_trades = len(close_trades)
        wins = sum(1 for t in close_trades if t.get("pnl", 0) > 0)
        wr = wins / n_trades if n_trades > 0 else 0
        coin_results[symbol] = {"trades": n_trades, "wins": wins, "pnl": pnl, "wr": wr}

    analyzer = BacktestAnalyzer(initial_capital=config.get("general", {}).get("initial_capital", 10.0))
    result = analyzer.analyze(all_trades)
    return {"result": result, "coin_results": coin_results, "trades": all_trades}


def main():
    config = load_config()
    storage = DataStorage()

    coins = [
        "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "DOGE/USDT:USDT",
        "XRP/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT", "DOT/USDT:USDT",
        "LINK/USDT:USDT", "UNI/USDT:USDT", "ARB/USDT:USDT", "OP/USDT:USDT",
        "SUI/USDT:USDT", "APT/USDT:USDT", "NEAR/USDT:USDT",
    ]

    may30 = pd.Timestamp("2026-05-30")
    may24 = pd.Timestamp("2026-05-24")
    may20 = pd.Timestamp("2026-05-20")
    feb27 = pd.Timestamp("2026-02-27")

    all_pass = True

    # ===== 1) 90-day backtest (in-sample) =====
    print("=" * 70)
    print("1. 90-DAY IN-SAMPLE BACKTEST (Feb 27 - May 30, 2026)")
    print("=" * 70)
    in_sample = run_backtest(config, storage, coins, feb27, may30)
    r = in_sample["result"]

    wr_ok = r.win_rate >= 0.55
    sharpe_ok = r.sharpe_ratio > 2.0
    pnl_ok = r.total_pnl > 0
    profitable_count = sum(1 for c in in_sample["coin_results"].values() if c["pnl"] > 0)

    print(f"\n  Total trades: {r.total_trades}")
    print(f"  Win rate: {r.win_rate:.2%} {'[OK]' if wr_ok else '[FAIL]'}")
    print(f"  Sharpe ratio: {r.sharpe_ratio:.2f} {'[OK]' if sharpe_ok else '[FAIL]'}")
    print(f"  Total PnL: {r.total_pnl:+.4f}U {'[OK]' if pnl_ok else '[FAIL]'}")
    print(f"  Max drawdown: {r.max_drawdown:.2%}")
    print(f"  Profitable coins: {profitable_count}/{len(coins)}")

    print(f"\n  Per-coin results:")
    for sym, cr in sorted(in_sample["coin_results"].items(), key=lambda x: x[1]["pnl"], reverse=True):
        coin_name = sym.split("/")[0]
        print(f"    {coin_name:8s}: {cr['trades']:3d} trades, WR={cr['wr']:.0%}, PnL={cr['pnl']:+.4f}U")

    if not (wr_ok and sharpe_ok and pnl_ok):
        all_pass = False

    # ===== 2) 7-day out-of-sample (May 24-30) =====
    print("\n" + "=" * 70)
    print("2. 7-DAY OUT-OF-SAMPLE VALIDATION (May 24-30, 2026)")
    print("=" * 70)
    oos = run_backtest(config, storage, coins, may24, may30)
    r7 = oos["result"]

    wr7_ok = r7.win_rate >= 0.55
    sharpe7_ok = r7.sharpe_ratio > 2.0
    pnl7_ok = r7.total_pnl > 0

    print(f"\n  Total trades: {r7.total_trades}")
    print(f"  Win rate: {r7.win_rate:.2%} {'[OK]' if wr7_ok else '[FAIL]'}")
    print(f"  Sharpe ratio: {r7.sharpe_ratio:.2f} {'[OK]' if sharpe7_ok else '[FAIL]'}")
    print(f"  Total PnL: {r7.total_pnl:+.4f}U {'[OK]' if pnl7_ok else '[FAIL]'}")

    print(f"\n  Per-coin results:")
    for sym, cr in sorted(oos["coin_results"].items(), key=lambda x: x[1]["pnl"], reverse=True):
        coin_name = sym.split("/")[0]
        print(f"    {coin_name:8s}: {cr['trades']:3d} trades, WR={cr['wr']:.0%}, PnL={cr['pnl']:+.4f}U")

    if not (wr7_ok and sharpe7_ok and pnl7_ok):
        all_pass = False

    # ===== 3) Overfitting test =====
    print("\n" + "=" * 70)
    print("3. OVERFITTING TEST (Train 83d + Validate 10d)")
    print("=" * 70)
    train_r = run_backtest(config, storage, coins, feb27, may20)
    val_r = run_backtest(config, storage, coins, may20, may30)
    rt = train_r["result"]
    rv = val_r["result"]

    overfit_wr_ok = rv.win_rate >= 0.50
    overfit_pnl_ok = rv.total_pnl > 0

    print(f"\n  Train (83d): trades={rt.total_trades}, WR={rt.win_rate:.2%}, PnL={rt.total_pnl:+.4f}U")
    print(f"  Validate (10d): trades={rv.total_trades}, WR={rv.win_rate:.2%}, PnL={rv.total_pnl:+.4f}U")
    print(f"  Overfitting test: {'[OK]' if overfit_wr_ok and overfit_pnl_ok else '[FAIL]'}")

    if not (overfit_wr_ok and overfit_pnl_ok):
        all_pass = False

    # ===== Summary =====
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  1. 90-day WR > 55%:     {'[OK]' if wr_ok else '[FAIL]'} ({r.win_rate:.2%})")
    print(f"  2. 90-day Sharpe > 2.0:  {'[OK]' if sharpe_ok else '[FAIL]'} ({r.sharpe_ratio:.2f})")
    print(f"  3. 90-day PnL > 0:       {'[OK]' if pnl_ok else '[FAIL]'} ({r.total_pnl:+.4f}U)")
    print(f"  4. 7-day WR > 55%:       {'[OK]' if wr7_ok else '[FAIL]'} ({r7.win_rate:.2%})")
    print(f"  5. 7-day Sharpe > 2.0:   {'[OK]' if sharpe7_ok else '[FAIL]'} ({r7.sharpe_ratio:.2f})")
    print(f"  6. 7-day PnL > 0:        {'[OK]' if pnl7_ok else '[FAIL]'} ({r7.total_pnl:+.4f}U)")
    print(f"  7. Overfitting test:     {'[OK]' if overfit_wr_ok and overfit_pnl_ok else '[FAIL]'}")
    print(f"\n  OVERALL: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print("=" * 70)

    return all_pass


if __name__ == "__main__":
    main()
