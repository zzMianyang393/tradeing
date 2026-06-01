"""Baseline test - run current trend_4h_filter on all 10 coins"""

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

    # Filter to time range
    mask = (df.index >= start_dt) & (df.index <= end_dt)
    df = df[mask].copy()
    if len(df) < 10:
        return pd.DataFrame()

    # Extract timestamps before resetting index
    timestamps_ms = (df.index.astype(np.int64) // 10**6).values

    # Reset index
    df = df.reset_index(drop=True)
    df['timestamp'] = timestamps_ms[:len(df)]

    # Keep only needed columns
    keep_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df = df[[c for c in keep_cols if c in df.columns]]

    return df


def run_backtest(config: dict, storage: DataStorage, coins: list,
                 start_dt: pd.Timestamp, end_dt: pd.Timestamp, label: str = "") -> dict:
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

        engine = BacktestEngine(config)
        stats = engine.run(symbol, df_15m, htf4_data=df_4h)

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
        logger.info(f"  {symbol}: {n_trades} trades, WR={wr:.0%}, PnL={pnl:+.4f}U")

    analyzer = BacktestAnalyzer(initial_capital=config.get("general", {}).get("initial_capital", 10.0))
    result = analyzer.analyze(all_trades)
    return {"result": result, "coin_results": coin_results, "trades": all_trades}


def main():
    config = load_config()
    storage = DataStorage()

    coins = [
        "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "DOGE/USDT:USDT",
        "XRP/USDT:USDT", "NEAR/USDT:USDT", "HYPE/USDT:USDT", "PEPE/USDT:USDT",
        "SUI/USDT:USDT", "FIL/USDT:USDT",
    ]

    may30 = pd.Timestamp("2026-05-30")
    may24 = pd.Timestamp("2026-05-24")
    may20 = pd.Timestamp("2026-05-20")
    feb27 = pd.Timestamp("2026-02-27")

    # 1) 90-day backtest
    print("=" * 70)
    print("90-DAY IN-SAMPLE BACKTEST (Feb 27 - May 30, 2026)")
    print("=" * 70)
    in_sample = run_backtest(config, storage, coins, feb27, may30, "90d")
    r = in_sample["result"]
    print(f"\n  Total trades: {r.total_trades}")
    print(f"  Win rate: {r.win_rate:.2%} {'[OK]' if r.win_rate >= 0.55 else '[FAIL]'}")
    print(f"  Sharpe ratio: {r.sharpe_ratio:.2f} {'[OK]' if r.sharpe_ratio > 2 else '[FAIL]'}")
    print(f"  Total PnL: {r.total_pnl:+.4f}U {'[OK]' if r.total_pnl > 0 else '[FAIL]'}")
    print(f"  Max drawdown: {r.max_drawdown:.2%}")
    print(f"  Profitable coins: {sum(1 for c in in_sample['coin_results'].values() if c['pnl'] > 0)}/{len(coins)}")

    # 2) 7-day out-of-sample
    print("\n" + "=" * 70)
    print("7-DAY OUT-OF-SAMPLE VALIDATION (May 24-30, 2026)")
    print("=" * 70)
    oos = run_backtest(config, storage, coins, may24, may30, "7d")
    r7 = oos["result"]
    print(f"\n  Total trades: {r7.total_trades}")
    print(f"  Win rate: {r7.win_rate:.2%} {'[OK]' if r7.win_rate >= 0.55 else '[FAIL]'}")
    print(f"  Sharpe ratio: {r7.sharpe_ratio:.2f} {'[OK]' if r7.sharpe_ratio > 2 else '[FAIL]'}")
    print(f"  Total PnL: {r7.total_pnl:+.4f}U {'[OK]' if r7.total_pnl > 0 else '[FAIL]'}")

    # 3) Overfitting test
    print("\n" + "=" * 70)
    print("OVERFITTING TEST (Train 83d + Validate 10d)")
    print("=" * 70)
    train_r = run_backtest(config, storage, coins, feb27, may20, "train")
    val_r = run_backtest(config, storage, coins, may20, may30, "validate")
    rt = train_r["result"]
    rv = val_r["result"]
    print(f"\n  Train (83d): trades={rt.total_trades}, WR={rt.win_rate:.2%}, PnL={rt.total_pnl:+.4f}U")
    print(f"  Validate (10d): trades={rv.total_trades}, WR={rv.win_rate:.2%}, PnL={rv.total_pnl:+.4f}U")
    overfit_ok = rv.win_rate >= 0.50 and rv.total_pnl > 0
    print(f"  Overfitting test: {'[OK]' if overfit_ok else '[FAIL]'}")


if __name__ == "__main__":
    main()
