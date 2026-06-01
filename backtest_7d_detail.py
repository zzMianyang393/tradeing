"""
7天回测 - 完整交易明细
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
    df = df_coin.copy()
    df = compute_indicators(df)
    df = df.dropna(subset=['rsi', 'volume_ratio'])
    df = df.reset_index(drop=True)

    capital = 10.0
    position = None
    trades = []

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
                    'symbol': '', 'direction': direction, 'pnl': pnl, 'pnl_pct': pnl_pct - FEE_RATE - SLIPPAGE,
                    'exit': 'SL', 'entry_time': position['entry_time'], 'exit_time': dt,
                    'entry_price': entry_price, 'exit_price': price,
                    'size': position['size'],
                })
                position = None
            elif pnl_pct >= TP:
                pnl = position['size'] * (pnl_pct - FEE_RATE - SLIPPAGE)
                capital += position['size'] + pnl
                trades.append({
                    'symbol': '', 'direction': direction, 'pnl': pnl, 'pnl_pct': pnl_pct - FEE_RATE - SLIPPAGE,
                    'exit': 'TP', 'entry_time': position['entry_time'], 'exit_time': dt,
                    'entry_price': entry_price, 'exit_price': price,
                    'size': position['size'],
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

    return trades

def main():
    print("=" * 100)
    print("  7天回测完整交易明细 (May 24-31, 2026)")
    print("  策略: 4h EMA55趋势 + RSI + 成交量过滤 | SL=TP=1.5%")
    print("=" * 100)

    df_all = load_data()
    df_all = df_all[df_all['symbol'].isin(COINS)]

    start = pd.Timestamp('2026-05-24')
    end = pd.Timestamp('2026-05-31')
    start_ts = int(start.timestamp() * 1000)
    end_ts = int(end.timestamp() * 1000)
    early_start = start - pd.Timedelta(days=15)
    early_start_ts = int(early_start.timestamp() * 1000)

    all_trades = []

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

        trades = run_backtest_detailed(test_df, trend_map)
        name = symbol.split('/')[0]
        for t in trades:
            t['symbol'] = name
        all_trades.extend(trades)

    # 按时间排序
    all_trades.sort(key=lambda x: x['entry_time'])

    # 打印全部交易
    print(f"\n{'#':>4s} {'币种':5s} {'时间':12s} {'方向':4s} {'入场价':>10s} {'出场价':>10s} {'结果':4s} {'盈亏%':>7s} {'盈亏U':>8s} {'累计U':>8s}")
    print("-" * 85)

    cumulative = 0
    wins = 0
    losses = 0
    win_pnl = 0
    loss_pnl = 0

    for i, t in enumerate(all_trades, 1):
        cumulative += t['pnl']
        direction = '做多' if t['direction'] == 'long' else '做空'
        result = '止盈' if t['exit'] == 'TP' else '止损'
        pnl_sign = '+' if t['pnl'] > 0 else ''

        if t['pnl'] > 0:
            wins += 1
            win_pnl += t['pnl']
        else:
            losses += 1
            loss_pnl += t['pnl']

        print(f"{i:4d} {t['symbol']:5s} {t['entry_time']:12s} {direction:4s} {t['entry_price']:>10.4f} {t['exit_price']:>10.4f} {result:4s} {pnl_sign}{t['pnl_pct']*100:>6.2f}% {pnl_sign}{t['pnl']:>7.4f} {'+' if cumulative>0 else ''}{cumulative:>7.4f}")

    # 统计
    print("-" * 85)
    print(f"\n{'='*60}")
    print(f"  交易统计")
    print(f"{'='*60}")
    print(f"  总交易:   {len(all_trades)} 笔")
    print(f"  盈利:     {wins} 笔 (止盈)")
    print(f"  亏损:     {losses} 笔 (止损)")
    print(f"  胜率:     {wins/len(all_trades):.1%}")
    print(f"  盈利总额: +{win_pnl:.4f}U")
    print(f"  亏损总额: {loss_pnl:.4f}U")
    print(f"  净盈亏:   {cumulative:+.4f}U")
    print(f"  盈亏比:   {abs(win_pnl/loss_pnl):.2f}" if loss_pnl != 0 else "  盈亏比:   N/A")
    print(f"  平均盈利: +{win_pnl/wins:.4f}U" if wins > 0 else "  平均盈利: N/A")
    print(f"  平均亏损: {loss_pnl/losses:.4f}U" if losses > 0 else "  平均亏损: N/A")

    # 按币种统计
    print(f"\n{'='*60}")
    print(f"  按币种统计")
    print(f"{'='*60}")
    from collections import defaultdict
    coin_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0})
    for t in all_trades:
        if t['pnl'] > 0:
            coin_stats[t['symbol']]['wins'] += 1
        else:
            coin_stats[t['symbol']]['losses'] += 1
        coin_stats[t['symbol']]['pnl'] += t['pnl']

    print(f"  {'币种':5s} {'盈':>3s} {'亏':>3s} {'胜率':>6s} {'净盈亏':>8s}")
    print(f"  {'-'*35}")
    for coin in sorted(coin_stats.keys()):
        s = coin_stats[coin]
        total = s['wins'] + s['losses']
        wr = s['wins'] / total if total > 0 else 0
        sign = '+' if s['pnl'] > 0 else ''
        print(f"  {coin:5s} {s['wins']:3d} {s['losses']:3d} {wr:>6.1%} {sign}{s['pnl']:>7.4f}U")

if __name__ == '__main__':
    main()
