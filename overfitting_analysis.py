"""
过拟合分析 - 训练/验证分离
训练集: Feb 27 - May 24 (参数调优期间)
验证集: May 24 - May 31 (完全未见过的数据)
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

def run_backtest(df_coin, trend_map):
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

        if position is not None:
            entry_price = position['entry_price']
            direction = position['direction']
            pnl_pct = (price - entry_price) / entry_price if direction == 'long' else (entry_price - price) / entry_price

            if pnl_pct <= -SL:
                pnl = position['size'] * (pnl_pct - FEE_RATE - SLIPPAGE)
                capital += position['size'] + pnl
                trades.append({'direction': direction, 'pnl': pnl})
                position = None
            elif pnl_pct >= TP:
                pnl = position['size'] * (pnl_pct - FEE_RATE - SLIPPAGE)
                capital += position['size'] + pnl
                trades.append({'direction': direction, 'pnl': pnl})
                position = None

        if position is None and capital > 0.1:
            if trend in ('bullish', 'neutral') and rsi < 40 and vol_ratio > 1.0:
                size = min(capital * 0.95, capital)
                if size > 0.1:
                    position = {'direction': 'long', 'entry_price': price, 'size': size}
                    capital -= size
            elif trend in ('bearish', 'neutral') and rsi > 60 and vol_ratio > 1.0:
                size = min(capital * 0.95, capital)
                if size > 0.1:
                    position = {'direction': 'short', 'entry_price': price, 'size': size}
                    capital -= size

    if not trades:
        return {'trades': 0, 'win_rate': 0, 'pnl': 0, 'wins': 0}

    wins = sum(1 for t in trades if t['pnl'] > 0)
    return {
        'trades': len(trades),
        'win_rate': wins / len(trades),
        'pnl': sum(t['pnl'] for t in trades),
        'wins': wins,
    }

def analyze_period(df_all, start, end, label, train_end=None):
    start_ts = int(start.timestamp() * 1000)
    end_ts = int(end.timestamp() * 1000)
    # 4h趋势需要更早的数据
    early_start = start - pd.Timedelta(days=15)
    early_start_ts = int(early_start.timestamp() * 1000)

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

        result = run_backtest(test_df, trend_map)
        result['symbol'] = symbol.split('/')[0]
        results.append(result)

    if not results:
        return None

    total_trades = sum(r['trades'] for r in results)
    total_wins = sum(r['wins'] for r in results)
    total_pnl = sum(r['pnl'] for r in results)
    wr = total_wins / total_trades if total_trades > 0 else 0
    profitable = sum(1 for r in results if r['pnl'] > 0)

    return {
        'label': label,
        'trades': total_trades,
        'win_rate': wr,
        'pnl': total_pnl,
        'profitable_coins': profitable,
        'total_coins': len(results),
    }

def main():
    print("=" * 70)
    print("  过拟合分析 - 训练/验证分离")
    print("=" * 70)

    df_all = load_data()
    df_all = df_all[df_all['symbol'].isin(COINS)]

    # 时间分割点
    train_end = pd.Timestamp('2026-05-24')  # 参数调优截止日
    data_end = pd.Timestamp('2026-05-31')

    # 分析各个窗口
    periods = [
        # 训练期内的不同窗口
        (pd.Timestamp('2026-02-27'), pd.Timestamp('2026-03-31'), "训练期-30天 (Feb 27-Mar 31)"),
        (pd.Timestamp('2026-02-27'), pd.Timestamp('2026-04-30'), "训练期-60天 (Feb 27-Apr 30)"),
        (pd.Timestamp('2026-02-27'), train_end, "训练期-87天 (Feb 27-May 24) ← 参数调优期"),
        # 验证期 (完全未见过)
        (train_end, data_end, "验证期-7天 (May 24-31) ← 完全未见过"),
        # 滑动窗口测试 (从训练期中选几个不重叠的子窗口)
        (pd.Timestamp('2026-03-15'), pd.Timestamp('2026-03-31'), "子窗口-15天 (Mar 15-31)"),
        (pd.Timestamp('2026-04-08'), pd.Timestamp('2026-04-24'), "子窗口-15天 (Apr 8-24)"),
        (pd.Timestamp('2026-05-08'), pd.Timestamp('2026-05-24'), "子窗口-15天 (May 8-24)"),
    ]

    print(f"\n参数调优截止日: {train_end.strftime('%Y-%m-%d')}")
    print(f"测试截止日:     {data_end.strftime('%Y-%m-%d')}")
    print(f"\n{'时间段':35s} {'交易数':>6s} {'胜率':>6s} {'盈亏(U)':>8s} {'盈利币种':>8s}")
    print("-" * 70)

    results = []
    for start, end, label in periods:
        r = analyze_period(df_all, start, end, label)
        if r:
            results.append(r)
            wr = f"{r['win_rate']:.1%}" if r['trades'] > 0 else "N/A"
            coins = f"{r['profitable_coins']}/{r['total_coins']}"
            print(f"{r['label']:35s} {r['trades']:6d} {wr:>6s} {r['pnl']:>+8.4f} {coins:>8s}")

    # 过拟合检测
    print("\n" + "=" * 70)
    print("  过拟合检测")
    print("=" * 70)

    train_results = [r for r in results if '训练期-87天' in r['label']]
    val_results = [r for r in results if '验证期-7天' in r['label']]
    sub_windows = [r for r in results if '子窗口' in r['label']]

    if train_results and val_results:
        train = train_results[0]
        val = val_results[0]

        wr_drop = train['win_rate'] - val['win_rate']
        pnl_ratio = val['pnl'] / abs(train['pnl']) if train['pnl'] != 0 else 0

        print(f"\n  训练期 (87天):")
        print(f"    胜率: {train['win_rate']:.1%}")
        print(f"    盈亏: {train['pnl']:+.4f}U")

        print(f"\n  验证期 (7天, 完全未见过):")
        print(f"    胜率: {val['win_rate']:.1%}")
        print(f"    盈亏: {val['pnl']:+.4f}U")

        print(f"\n  性能衰减:")
        print(f"    胜率下降: {wr_drop:+.1%} ({'严重' if wr_drop > 0.1 else '正常' if wr_drop > 0.05 else '轻微'})")
        print(f"    盈亏比:  {pnl_ratio:.2f} ({'好' if pnl_ratio > 0.3 else '差'})")

        # 子窗口一致性
        if sub_windows:
            wrs = [r['win_rate'] for r in sub_windows]
            wr_std = np.std(wrs)
            print(f"\n  子窗口一致性 (15天窗口×3):")
            print(f"    胜率范围: {min(wrs):.1%} ~ {max(wrs):.1%}")
            print(f"    胜率标准差: {wr_std:.3f} ({'稳定' if wr_std < 0.05 else '波动较大'})")

        # 过拟合判断
        print(f"\n  结论:")
        overfitting = False

        if wr_drop > 0.10:
            print(f"    ❌ 验证期胜率下降超过10%，可能存在过拟合")
            overfitting = True
        else:
            print(f"    ✅ 验证期胜率下降{wr_drop:+.1%}，在可接受范围内")

        if val['pnl'] < 0:
            print(f"    ❌ 验证期亏损，策略在新数据上失效")
            overfitting = True
        else:
            print(f"    ✅ 验证期盈利{val['pnl']:+.4f}U，策略在新数据上仍有效")

        if sub_windows and np.std(wrs) > 0.08:
            print(f"    ⚠️  子窗口胜率波动较大，策略稳定性存疑")
        elif sub_windows:
            print(f"    ✅ 子窗口胜率稳定，策略鲁棒性好")

        if not overfitting:
            print(f"\n  ✅ 未发现明显过拟合")
        else:
            print(f"\n  ⚠️  存在过拟合风险，建议:")
            print(f"      1. 简化策略参数")
            print(f"      2. 增加训练数据量")
            print(f"      3. 使用更严格的入场条件")

if __name__ == '__main__':
    main()
