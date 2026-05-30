"""最终验证脚本 - 使用已验证的4h趋势过滤策略"""

import sys
sys.path.insert(0, '.')

import sqlite3
import yaml
import pandas as pd
import numpy as np
from datetime import datetime

def load_data(symbol):
    conn = sqlite3.connect('data/trading.db')
    df = pd.read_sql_query(
        'SELECT * FROM klines WHERE symbol=? AND timeframe="15m" ORDER BY timestamp ASC',
        conn, params=[symbol]
    )
    conn.close()
    df['timestamp'] = df['timestamp'].astype(int)
    return df

def compute_4h_trend(df):
    """从15m数据计算4h趋势"""
    df.index = pd.to_datetime(df['timestamp'], unit='ms')
    df_4h = df[['open', 'high', 'low', 'close', 'volume']].resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna()
    df_4h['ema55'] = df_4h['close'].ewm(span=55, adjust=False).mean()

    trend_map = {}
    for idx, row in df_4h.iterrows():
        ts = int(idx.timestamp() * 1000)
        if row['close'] > row['ema55'] * 1.01:
            trend_map[ts] = 'bullish'
        elif row['close'] < row['ema55'] * 0.99:
            trend_map[ts] = 'bearish'
        else:
            trend_map[ts] = 'neutral'

    return trend_map

def get_trend(trend_map, ts_ms):
    for k in reversed(sorted(trend_map.keys())):
        if k <= ts_ms:
            return trend_map[k]
    return 'neutral'

def run_backtest(df, sl=0.015, tp=0.015, rsi_long=40, rsi_short=60, vol_thresh=1.0,
                 position_size=2.0, leverage=5, fee_rate=0.001, cooldown_bars=6):
    """运行回测"""
    from strategy.indicators import TechnicalIndicators

    ind = TechnicalIndicators({})
    df = ind.calculate(df)

    trend_map = compute_4h_trend(df)
    df = df.reset_index(drop=True)

    in_pos = False
    pos = {}
    cooldown = 0
    trades = []
    returns = []
    balance = 10.0

    for i in range(60, len(df)):
        row = df.iloc[i]
        price = row['close']
        ts = row['timestamp']
        htf = get_trend(trend_map, ts)

        if cooldown > 0:
            cooldown -= 1
            continue

        # Check existing position
        if in_pos:
            d = pos['dir']
            if d == 'long':
                if row['low'] <= pos['sl']:
                    pnl = position_size * leverage * ((pos['sl'] - pos['entry']) / pos['entry']) - position_size * leverage * fee_rate
                    ret = ((pos['sl'] - pos['entry']) / pos['entry']) * leverage - fee_rate
                    trades.append(pnl)
                    returns.append(ret)
                    balance += pnl
                    in_pos = False
                    cooldown = cooldown_bars
                elif row['high'] >= pos['tp']:
                    pnl = position_size * leverage * ((pos['tp'] - pos['entry']) / pos['entry']) - position_size * leverage * fee_rate
                    ret = ((pos['tp'] - pos['entry']) / pos['entry']) * leverage - fee_rate
                    trades.append(pnl)
                    returns.append(ret)
                    balance += pnl
                    in_pos = False
                    cooldown = cooldown_bars
            else:  # short
                if row['high'] >= pos['sl']:
                    pnl = position_size * leverage * ((pos['entry'] - pos['sl']) / pos['entry']) - position_size * leverage * fee_rate
                    ret = ((pos['entry'] - pos['sl']) / pos['entry']) * leverage - fee_rate
                    trades.append(pnl)
                    returns.append(ret)
                    balance += pnl
                    in_pos = False
                    cooldown = cooldown_bars
                elif row['low'] <= pos['tp']:
                    pnl = position_size * leverage * ((pos['entry'] - pos['tp']) / pos['entry']) - position_size * leverage * fee_rate
                    ret = ((pos['entry'] - pos['tp']) / pos['entry']) * leverage - fee_rate
                    trades.append(pnl)
                    returns.append(ret)
                    balance += pnl
                    in_pos = False
                    cooldown = cooldown_bars

        # Check new signal
        if not in_pos:
            rsi = row.get('rsi', 50)
            vol_r = row.get('volume_ratio', 1)

            if htf == 'bullish' and rsi < rsi_long and vol_r > vol_thresh:
                in_pos = True
                pos = {
                    'dir': 'long',
                    'entry': price,
                    'sl': price * (1 - sl),
                    'tp': price * (1 + tp),
                }
            elif htf == 'bearish' and rsi > rsi_short and vol_r > vol_thresh:
                in_pos = True
                pos = {
                    'dir': 'short',
                    'entry': price,
                    'sl': price * (1 + sl),
                    'tp': price * (1 - tp),
                }

    # Calculate stats
    wins = sum(1 for t in trades if t > 0)
    wr = wins / len(trades) if trades else 0
    total_pnl = sum(trades)
    sharpe = 0
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = np.sqrt(252 * 96) * np.mean(returns) / np.std(returns)

    # Max drawdown
    if trades:
        cumsum = np.cumsum(trades)
        peak = np.maximum.accumulate(cumsum)
        dd = peak - cumsum
        max_dd = np.max(dd) / 10.0 if np.max(dd) > 0 else 0  # relative to initial capital
    else:
        max_dd = 0

    return {
        'trades': len(trades),
        'win_rate': wr,
        'total_pnl': total_pnl,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'balance': balance,
        'wins': wins,
        'losses': len(trades) - wins,
    }


def main():
    coins = [
        'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'DOGE/USDT:USDT',
        'XRP/USDT:USDT', 'ADA/USDT:USDT', 'AVAX/USDT:USDT', 'DOT/USDT:USDT',
        'LINK/USDT:USDT', 'UNI/USDT:USDT', 'ARB/USDT:USDT', 'OP/USDT:USDT',
        'SUI/USDT:USDT', 'APT/USDT:USDT', 'NEAR/USDT:USDT',
    ]

    split_ts = int(datetime(2026, 5, 24).timestamp() * 1000)
    oos_end = int(datetime(2026, 5, 31).timestamp() * 1000)

    # Strategy parameters (validated)
    SL = 0.015
    TP = 0.015

    print("=" * 80)
    print(f"Strategy: 4h Trend Filter (SL={SL:.1%}, TP={TP:.1%})")
    print(f"In-sample: before 2026-05-24")
    print(f"Out-of-sample: 2026-05-24 → 2026-05-30")
    print("=" * 80)

    # ========== IN-SAMPLE ==========
    print("\n📊 IN-SAMPLE BACKTEST")
    print("-" * 60)
    is_results = {}
    is_total_t = 0
    is_total_w = 0
    is_total_pnl = 0
    is_profitable = 0
    is_all_returns = []

    for coin in coins:
        df = load_data(coin)
        train = df[df['timestamp'] < split_ts].copy()
        result = run_backtest(train, sl=SL, tp=TP)

        is_results[coin] = result
        is_total_t += result['trades']
        is_total_w += result['wins']
        is_total_pnl += result['total_pnl']
        if result['total_pnl'] > 0:
            is_profitable += 1

        name = coin.split('/')[0]
        status = '✅' if result['total_pnl'] > 0 else '⚠️'
        print(f"  {status} {name:6s}: trades={result['trades']:4d}, WR={result['win_rate']:5.1%}, "
              f"PnL={result['total_pnl']:+8.4f}, Sharpe={result['sharpe']:7.2f}")

    is_wr = is_total_w / is_total_t if is_total_t else 0

    # ========== OUT-OF-SAMPLE ==========
    print("\n📊 OUT-OF-SAMPLE BACKTEST (7 days: May 24-30)")
    print("-" * 60)
    oos_results = {}
    oos_total_t = 0
    oos_total_w = 0
    oos_total_pnl = 0
    oos_profitable = 0

    for coin in coins:
        df = load_data(coin)
        test = df[(df['timestamp'] >= split_ts) & (df['timestamp'] < oos_end)].copy()
        result = run_backtest(test, sl=SL, tp=TP)

        oos_results[coin] = result
        oos_total_t += result['trades']
        oos_total_w += result['wins']
        oos_total_pnl += result['total_pnl']
        if result['total_pnl'] > 0:
            oos_profitable += 1

        name = coin.split('/')[0]
        status = '✅' if result['total_pnl'] > 0 else '⚠️'
        print(f"  {status} {name:6s}: trades={result['trades']:4d}, WR={result['win_rate']:5.1%}, "
              f"PnL={result['total_pnl']:+8.4f}, Sharpe={result['sharpe']:7.2f}")

    oos_wr = oos_total_w / oos_total_t if oos_total_t else 0

    # ========== SUMMARY ==========
    print("\n" + "=" * 80)
    print("📋 SUMMARY")
    print("=" * 80)
    print(f"\nIn-Sample:")
    print(f"  Total trades: {is_total_t}")
    print(f"  Win rate: {is_wr:.1%}")
    print(f"  Total PnL: {is_total_pnl:+.4f} U")
    print(f"  Profitable coins: {is_profitable}/15")

    print(f"\nOut-of-Sample (7 days):")
    print(f"  Total trades: {oos_total_t}")
    print(f"  Win rate: {oos_wr:.1%}")
    print(f"  Total PnL: {oos_total_pnl:+.4f} U")
    print(f"  Profitable coins: {oos_profitable}/15")

    # ========== VALIDATION ==========
    print(f"\n{'=' * 80}")
    print("🎯 VALIDATION RESULTS")
    print(f"{'=' * 80}")

    checks = [
        ("Coins >= 10 (OOS)", oos_profitable >= 10),
        ("IS Win Rate > 55%", is_wr > 0.55),
        ("IS PnL > 0", is_total_pnl > 0),
        ("OOS Win Rate > 55%", oos_wr > 0.55),
        ("OOS PnL > 0", oos_total_pnl > 0),
        ("OOS Trades > 20", oos_total_t > 20),
    ]

    all_pass = True
    for check, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_pass = False
        print(f"  {status} | {check}")

    print(f"\n{'=' * 80}")
    if all_pass:
        print("🎉 ALL CHECKS PASSED - STRATEGY APPROVED FOR DEPLOYMENT")
    else:
        print("⚠️ SOME CHECKS FAILED - FURTHER OPTIMIZATION NEEDED")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
