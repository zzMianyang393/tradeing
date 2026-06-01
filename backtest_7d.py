"""
7天回测 - 最新数据 (May 24-31, 2026)
"""

import sys
sys.path.insert(0, '.')
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

SL = 0.015
TP = 0.015
FEE_RATE = 0.0005
SLIPPAGE = 0.0005

COINS = [
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'DOGE/USDT:USDT',
    'XRP/USDT:USDT', 'ADA/USDT:USDT', 'AVAX/USDT:USDT', 'DOT/USDT:USDT',
    'LINK/USDT:USDT', 'UNI/USDT:USDT', 'ARB/USDT:USDT', 'OP/USDT:USDT',
    'SUI/USDT:USDT', 'APT/USDT:USDT', 'NEAR/USDT:USDT',
]

def load_data():
    conn = sqlite3.connect('data/trading.db')
    df = pd.read_sql_query(
        'SELECT * FROM klines WHERE timeframe="15m" ORDER BY timestamp ASC',
        conn
    )
    conn.close()
    return df

def resample_4h(df_15m):
    tmp = df_15m[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    tmp.index = pd.to_datetime(tmp['timestamp'], unit='ms')
    agg = tmp.resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna()
    agg = agg.reset_index()
    agg['timestamp'] = (agg['timestamp'].astype('int64') // 1_000_000).astype(int)
    return agg

def compute_4h_trend(df_4h):
    close = df_4h['close']
    ema55 = close.ewm(span=55, adjust=False).mean()
    trend_map = {}
    for i in range(len(df_4h)):
        ts = df_4h['timestamp'].iloc[i]
        price = close.iloc[i]
        ema = ema55.iloc[i]
        if price > ema * 1.01:
            trend_map[ts] = 'bullish'
        elif price < ema * 0.99:
            trend_map[ts] = 'bearish'
        else:
            trend_map[ts] = 'neutral'
    return trend_map

def compute_indicators(df):
    close = df['close']
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    return df

def get_4h_trend_at(trend_map, ts):
    keys = sorted(trend_map.keys())
    for k in reversed(keys):
        if k <= ts:
            return trend_map[k]
    return 'neutral'

def run_backtest_detailed(df_coin, trend_map):
    """返回详细交易记录"""
    df = df_coin.copy()
    df = compute_indicators(df)
    df = df.dropna(subset=['rsi', 'volume_ratio'])
    df = df.reset_index(drop=True)

    capital = 10.0
    position = None
    trades = []
    equity_curve = [10.0]
    peak = 10.0
    max_dd = 0

    for i in range(1, len(df)):
        row = df.iloc[i]
        ts = row['timestamp']
        price = float(row['close'])
        rsi = float(row['rsi'])
        vol_ratio = float(row['volume_ratio'])
        trend = get_4h_trend_at(trend_map, ts)
        dt = datetime.utcfromtimestamp(ts / 1000).strftime('%m-%d %H:%M')

        if position is not None:
            entry_price = position['entry_price']
            direction = position['direction']
            pnl_pct = (price - entry_price) / entry_price if direction == 'long' else (entry_price - price) / entry_price

            if pnl_pct <= -SL:
                pnl = position['size'] * (pnl_pct - FEE_RATE - SLIPPAGE)
                capital += position['size'] + pnl
                trades.append({
                    'direction': direction, 'pnl': pnl, 'exit': 'SL',
                    'entry_time': position['entry_time'], 'exit_time': dt,
                    'entry_price': entry_price, 'exit_price': price,
                })
                position = None
            elif pnl_pct >= TP:
                pnl = position['size'] * (pnl_pct - FEE_RATE - SLIPPAGE)
                capital += position['size'] + pnl
                trades.append({
                    'direction': direction, 'pnl': pnl, 'exit': 'TP',
                    'entry_time': position['entry_time'], 'exit_time': dt,
                    'entry_price': entry_price, 'exit_price': price,
                })
                position = None

        if position is None and capital > 0.1:
            signal = None
            if trend in ('bullish', 'neutral') and rsi < 40 and vol_ratio > 1.0:
                signal = 'long'
            elif trend in ('bearish', 'neutral') and rsi > 60 and vol_ratio > 1.0:
                signal = 'short'

            if signal:
                size = min(capital * 0.95, capital)
                if size > 0.1:
                    position = {'direction': signal, 'entry_price': price, 'size': size, 'entry_time': dt}
                    capital -= size

        current_equity = capital
        if position is not None:
            if position['direction'] == 'long':
                unrealized = position['size'] * ((price - position['entry_price']) / position['entry_price'])
            else:
                unrealized = position['size'] * ((position['entry_price'] - price) / position['entry_price'])
            current_equity += position['size'] + unrealized

        equity_curve.append(current_equity)
        if current_equity > peak:
            peak = current_equity
        dd = (peak - current_equity) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    if not trades:
        return {
            'trades': [], 'win_rate': 0, 'pnl': 0, 'sharpe': 0,
            'max_dd': 0, 'equity_curve': equity_curve, 'total': 0, 'wins': 0
        }

    wins = sum(1 for t in trades if t['pnl'] > 0)
    total_pnl = sum(t['pnl'] for t in trades)
    win_rate = wins / len(trades)

    returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
    sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(365 * 96)) if np.std(returns) > 0 else 0

    return {
        'trades': trades, 'win_rate': win_rate, 'pnl': total_pnl,
        'sharpe': sharpe, 'max_dd': max_dd, 'equity_curve': equity_curve,
        'total': len(trades), 'wins': wins
    }

def main():
    print("=" * 70)
    print("  7天回测详情 (May 24-31, 2026)")
    print("  策略: 4h EMA55趋势 + RSI + 成交量过滤")
    print("  参数: SL=1.5%, TP=1.5%, RSI做多<40, RSI做空>60")
    print("=" * 70)

    df_all = load_data()
    df_all = df_all[df_all['symbol'].isin(COINS)]

    start = pd.Timestamp('2026-05-24')
    end = pd.Timestamp('2026-05-31')
    start_ts = int(start.timestamp() * 1000)
    end_ts = int(end.timestamp() * 1000)

    # 4h趋势需要更早数据
    early_start = start - pd.Timedelta(days=15)
    early_start_ts = int(early_start.timestamp() * 1000)

    all_trades = []
    results = []

    for symbol in COINS:
        coin_df = df_all[df_all['symbol'] == symbol].copy()
        if coin_df.empty:
            continue

        full_df = coin_df[coin_df['timestamp'] >= early_start_ts].copy()
        full_df = full_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        full_df = full_df.sort_values('timestamp').reset_index(drop=True)

        if len(full_df) < 100:
            continue

        df_4h = resample_4h(full_df)
        if len(df_4h) < 55:
            continue

        trend_map = compute_4h_trend(df_4h)
        test_df = full_df[(full_df['timestamp'] >= start_ts) & (full_df['timestamp'] <= end_ts)].copy()

        if len(test_df) < 10:
            continue

        result = run_backtest_detailed(test_df, trend_map)
        name = symbol.split('/')[0]
        result['symbol'] = name
        results.append(result)
        all_trades.extend(result['trades'])

    # 打印每个币种
    print(f"\n{'币种':6s} {'交易':>4s} {'胜':>3s} {'负':>3s} {'胜率':>6s} {'盈亏(U)':>8s} {'Sharpe':>7s} {'MaxDD':>6s}")
    print("-" * 55)

    for r in sorted(results, key=lambda x: x['symbol']):
        losses = r['total'] - r['wins']
        wr = f"{r['win_rate']:.1%}" if r['total'] > 0 else "-"
        print(f"{r['symbol']:6s} {r['total']:4d} {r['wins']:3d} {losses:3d} {wr:>6s} {r['pnl']:>+8.4f} {r['sharpe']:>7.2f} {r['max_dd']:>6.1%}")

    # 汇总
    total_trades = sum(r['total'] for r in results)
    total_wins = sum(r['wins'] for r in results)
    total_pnl = sum(r['pnl'] for r in results)
    wr = total_wins / total_trades if total_trades > 0 else 0
    profitable = sum(1 for r in results if r['pnl'] > 0)

    print("-" * 55)
    print(f"{'总计':6s} {total_trades:4d} {total_wins:3d} {total_trades-total_wins:3d} {wr:>6.1%} {total_pnl:>+8.4f}")

    # 验证
    print(f"\n{'='*55}")
    print(f"  验证结果:")
    checks = [
        ("交易数 > 0", total_trades > 0),
        ("胜率 > 55%", wr > 0.55),
        ("盈亏 > 0", total_pnl > 0),
        (f"盈利币种 >= {len(results)*0.7:.0f}", profitable >= len(results) * 0.7),
    ]
    for name, ok in checks:
        print(f"    {'✅' if ok else '❌'} {name}")

    # 最近10笔交易
    print(f"\n{'='*55}")
    print(f"  最近10笔交易:")
    print(f"  {'时间':12s} {'方向':5s} {'入场':>10s} {'出场':>10s} {'结果':4s} {'盈亏(U)':>8s}")
    print(f"  {'-'*55}")
    for t in all_trades[-10:]:
        result = "止盈" if t['exit'] == 'TP' else "止损"
        print(f"  {t['entry_time']:12s} {'做多' if t['direction']=='long' else '做空':5s} {t['entry_price']:>10.4f} {t['exit_price']:>10.4f} {result:4s} {t['pnl']:>+8.4f}")

if __name__ == '__main__':
    main()
