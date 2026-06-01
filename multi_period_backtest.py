"""
多周期回测脚本 - 4h趋势过滤策略
30天 / 60天 / 90天 三个时间窗口
数据截至 2026-05-31
"""

import sys
sys.path.insert(0, '.')

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

SL = 0.015  # 1.5%
TP = 0.015  # 1.5%
FEE_RATE = 0.0005  # 0.05%
SLIPPAGE = 0.0005  # 0.05%

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
    """15m → 4h 重采样"""
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
    """计算4h EMA55趋势"""
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
    """计算技术指标"""
    close = df['close']
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    # Volume ratio
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    return df

def get_4h_trend_at(trend_map, ts):
    """获取给定时间戳对应的4h趋势"""
    keys = sorted(trend_map.keys())
    for k in reversed(keys):
        if k <= ts:
            return trend_map[k]
    return 'neutral'

def run_backtest(df_coin, trend_map, initial_capital=10.0):
    """对单个币种执行回测"""
    df = df_coin.copy()
    df = compute_indicators(df)
    df = df.dropna(subset=['rsi', 'volume_ratio'])
    df = df.reset_index(drop=True)

    capital = initial_capital
    position = None
    trades = []
    equity_curve = [initial_capital]
    peak = initial_capital
    max_dd = 0

    for i in range(1, len(df)):
        row = df.iloc[i]
        ts = row['timestamp']
        price = float(row['close'])
        rsi = float(row['rsi'])
        vol_ratio = float(row['volume_ratio'])
        trend = get_4h_trend_at(trend_map, ts)

        # 持仓管理
        if position is not None:
            entry_price = position['entry_price']
            direction = position['direction']

            if direction == 'long':
                pnl_pct = (price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - price) / entry_price

            # 止损
            if pnl_pct <= -SL:
                pnl = position['size'] * (pnl_pct - FEE_RATE - SLIPPAGE)
                capital += position['size'] + pnl
                trades.append({'direction': direction, 'pnl': pnl, 'exit': 'SL'})
                position = None
            # 止盈
            elif pnl_pct >= TP:
                pnl = position['size'] * (pnl_pct - FEE_RATE - SLIPPAGE)
                capital += position['size'] + pnl
                trades.append({'direction': direction, 'pnl': pnl, 'exit': 'TP'})
                position = None

        # 开仓
        if position is None and capital > 0.1:
            # 4h趋势做多
            if trend in ('bullish', 'neutral') and rsi < 40 and vol_ratio > 1.0:
                size = min(capital * 0.95, capital)
                if size > 0.1:
                    position = {'direction': 'long', 'entry_price': price, 'size': size}
                    capital -= size
            # 4h趋势做空
            elif trend in ('bearish', 'neutral') and rsi > 60 and vol_ratio > 1.0:
                size = min(capital * 0.95, capital)
                if size > 0.1:
                    position = {'direction': 'short', 'entry_price': price, 'size': size}
                    capital -= size

        # 记录权益
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

    # 计算统计
    if not trades:
        return {
            'trades': 0, 'win_rate': 0, 'pnl': 0, 'sharpe': 0,
            'max_dd': 0, 'profitable': False, 'equity_curve': equity_curve
        }

    wins = sum(1 for t in trades if t['pnl'] > 0)
    total_pnl = sum(t['pnl'] for t in trades)
    win_rate = wins / len(trades)

    # Sharpe (年化)
    returns = np.diff(equity_curve) / equity_curve[:-1]
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(365 * 96)  # 15m bars
    else:
        sharpe = 0

    return {
        'trades': len(trades),
        'win_rate': win_rate,
        'pnl': total_pnl,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'profitable': total_pnl > 0,
        'wins': wins,
        'losses': len(trades) - wins,
    }

def run_period(df_all, start_date, end_date, period_name):
    """运行指定时间段的回测"""
    print(f"\n{'='*70}")
    print(f"  {period_name}: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
    print(f"{'='*70}")

    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)

    # 需要更早的数据来计算4h趋势（EMA55需要55根4h蜡烛 = 220根15m蜡烛）
    early_start = start_date - pd.Timedelta(days=15)
    early_start_ts = int(early_start.timestamp() * 1000)

    results = []
    for symbol in COINS:
        coin_df = df_all[df_all['symbol'] == symbol].copy()
        if coin_df.empty or len(coin_df) < 100:
            continue

        # 获取完整数据用于4h趋势计算
        full_df = coin_df[coin_df['timestamp'] >= early_start_ts].copy()
        full_df = full_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        full_df = full_df.sort_values('timestamp').reset_index(drop=True)

        if len(full_df) < 100:
            continue

        # 重采样4h
        df_4h = resample_4h(full_df)
        if len(df_4h) < 55:
            continue

        # 计算4h趋势
        trend_map = compute_4h_trend(df_4h)

        # 过滤到测试时间段
        test_df = full_df[(full_df['timestamp'] >= start_ts) & (full_df['timestamp'] <= end_ts)].copy()
        if len(test_df) < 60:
            continue

        result = run_backtest(test_df, trend_map)
        result['symbol'] = symbol.split('/')[0]
        results.append(result)

    # 汇总
    if not results:
        print("  没有有效结果")
        return None

    total_trades = sum(r['trades'] for r in results)
    total_wins = sum(r.get('wins', 0) for r in results)
    total_pnl = sum(r['pnl'] for r in results)
    win_rate = total_wins / total_trades if total_trades > 0 else 0

    returns = []
    for r in results:
        if r['trades'] > 0:
            avg_ret = r['pnl'] / r['trades'] / 10.0  # 基于10U初始资金
            returns.extend([avg_ret] * r['trades'])

    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(365 * 96)
    else:
        sharpe = 0

    max_dd = max(r['max_dd'] for r in results)
    profitable_coins = sum(1 for r in results if r['profitable'])

    print(f"\n  {'币种':6s} {'交易数':>6s} {'胜率':>6s} {'盈亏(U)':>8s} {'Sharpe':>7s} {'MaxDD':>6s}")
    print(f"  {'-'*50}")
    for r in sorted(results, key=lambda x: x['symbol']):
        wr_str = f"{r['win_rate']:.1%}" if r['trades'] > 0 else "N/A"
        pnl_str = f"{r['pnl']:+.4f}"
        sharpe_str = f"{r['sharpe']:.2f}" if r['trades'] > 0 else "N/A"
        dd_str = f"{r['max_dd']:.1%}"
        print(f"  {r['symbol']:6s} {r['trades']:6d} {wr_str:>6s} {pnl_str:>8s} {sharpe_str:>7s} {dd_str:>6s}")

    print(f"\n  {'总计':6s} {total_trades:6d} {win_rate:>6.1%} {total_pnl:>+8.4f} {sharpe:>7.2f} {max_dd:>6.1%}")
    print(f"  盈利币种: {profitable_coins}/{len(results)}")

    # 验证
    checks = [
        ("交易数 > 0", total_trades > 0),
        ("胜率 > 55%", win_rate > 0.55),
        ("Sharpe > 2.0", sharpe > 2.0),
        ("盈亏 > 0", total_pnl > 0),
        (f"盈利币种 >= {len(results)*0.7:.0f}", profitable_coins >= len(results) * 0.7),
    ]

    all_passed = all(c[1] for c in checks)
    print(f"\n  验证结果:")
    for name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"    {status} {name}")
    print(f"\n  {'✅ 全部通过' if all_passed else '❌ 未完全通过'}")

    return {
        'period': period_name,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'pnl': total_pnl,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'profitable_coins': profitable_coins,
        'total_coins': len(results),
        'all_passed': all_passed,
    }

def main():
    print("=" * 70)
    print("  多周期回测 - 4h趋势过滤策略 (SL=1.5%, TP=1.5%)")
    print("  数据截至: 2026-05-31")
    print("  币种数量: 15")
    print("=" * 70)

    # 加载数据
    print("\n加载数据...")
    df_all = load_data()
    df_all = df_all[df_all['symbol'].isin(COINS)]
    print(f"  已加载 {len(df_all)} 条记录")

    # 三个时间段
    end_date = pd.Timestamp('2026-05-31')

    periods = [
        (end_date - pd.Timedelta(days=30), end_date, "30天回测 (May 1-31)"),
        (end_date - pd.Timedelta(days=60), end_date, "60天回测 (Apr 1 - May 31)"),
        (end_date - pd.Timedelta(days=90), end_date, "90天回测 (Mar 2 - May 31)"),
    ]

    summary = []
    for start, end, name in periods:
        result = run_period(df_all, start, end, name)
        if result:
            summary.append(result)

    # 总结
    print("\n" + "=" * 70)
    print("  总结")
    print("=" * 70)
    print(f"\n  {'时间段':20s} {'交易数':>6s} {'胜率':>6s} {'盈亏':>8s} {'Sharpe':>7s} {'通过':>4s}")
    print(f"  {'-'*55}")
    for s in summary:
        status = "✅" if s['all_passed'] else "❌"
        print(f"  {s['period']:20s} {s['total_trades']:6d} {s['win_rate']:>6.1%} {s['pnl']:>+8.4f} {s['sharpe']:>7.2f} {status:>4s}")

if __name__ == '__main__':
    main()
