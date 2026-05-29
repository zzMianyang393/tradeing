#!/usr/bin/env python3
"""cond7基础上的深度优化 - 调盈亏比"""

import yaml, sqlite3, pandas as pd, numpy as np, json
from strategy.hybrid import HybridStrategy
from strategy.indicators import TechnicalIndicators

def load_all_data():
    conn = sqlite3.connect('data/trading.db')
    symbols = [r[0] for r in conn.execute(
        "SELECT DISTINCT symbol FROM klines WHERE timeframe='15m'"
    ).fetchall()]
    data = {}
    for sym in symbols:
        df = pd.read_sql(
            "SELECT timestamp, open, high, low, close, volume FROM klines WHERE symbol=? AND timeframe='15m' ORDER BY timestamp",
            conn, params=(sym,)
        )
        data[sym] = df
    conn.close()
    return data

def backtest(data, sl_pct, tp_pct, min_cond, direction_mode='both'):
    with open('config/settings.yaml') as f:
        settings = yaml.safe_load(f)
    with open('config/strategy.yaml') as f:
        strat = yaml.safe_load(f)
    config = {**settings, **strat}
    config['adx']['threshold'] = 15

    strategy = HybridStrategy(config)
    indicators = TechnicalIndicators(config)

    all_trades = []
    balance = 10.0

    for symbol, raw_df in data.items():
        df = raw_df.copy().reset_index(drop=True)
        if len(df) < 100:
            continue
        split = int(len(df) * 0.7)
        train_df = df.iloc[:split]
        test_df = df.iloc[split:]
        if len(test_df) < 60:
            continue

        strategy.train_ml({symbol: train_df}, force=False)
        test_df = indicators.calculate(test_df)
        if test_df.empty or len(test_df) < 60:
            continue

        positions = {}
        for i in range(60, len(test_df)):
            row = test_df.iloc[i]
            price = float(row['close'])
            prev = test_df.iloc[i-1] if i > 0 else row

            # 检查持仓
            if symbol in positions:
                pos = positions[symbol]
                hit_sl = hit_tp = False
                if pos['dir'] == 'short':
                    if price >= pos['entry'] * (1 + sl_pct): hit_sl = True
                    elif price <= pos['entry'] * (1 - tp_pct): hit_tp = True
                else:
                    if price <= pos['entry'] * (1 - sl_pct): hit_sl = True
                    elif price >= pos['entry'] * (1 + tp_pct): hit_tp = True

                if hit_sl or hit_tp:
                    pnl_pct = ((pos['entry'] - price) / pos['entry']) if pos['dir'] == 'short' else ((price - pos['entry']) / pos['entry'])
                    pnl_pct *= pos['lev']
                    fee = pos['amt'] * pos['lev'] * 0.0005 * 2
                    pnl = pos['amt'] * pnl_pct - fee
                    balance += pos['amt'] + pnl
                    all_trades.append({'pnl': pnl, 'dir': pos['dir'], 'reason': '止盈' if hit_tp else '止损'})
                    del positions[symbol]
                continue

            # 趋势过滤
            ema_slow = row.get('ema_slow', 0)
            trend = 'neutral'
            if ema_slow > 0:
                if price > ema_slow * 1.005: trend = 'bullish'
                elif price < ema_slow * 0.995: trend = 'bearish'

            # 尝试开仓
            opened = False
            
            if trend != 'bearish' and direction_mode != 'short_only':
                sig = strategy.signal_gen._check_long_conditions(row, prev, test_df.iloc[:i+1])
                score = sum(sig.values())
                if score >= min_cond:
                    lev = 50 if score >= 6 else 27
                    amt = min(balance * 0.25, 5.0)
                    amt = max(amt, 1.0)
                    if amt <= balance and balance >= 1:
                        positions[symbol] = {'dir': 'long', 'entry': price, 'amt': amt, 'lev': lev}
                        balance -= amt
                        opened = True

            if not opened and trend != 'bullish' and direction_mode != 'long_only':
                sig = strategy.signal_gen._check_short_conditions(row, prev, test_df.iloc[:i+1])
                score = sum(sig.values())
                if score >= min_cond:
                    lev = 50 if score >= 6 else 27
                    amt = min(balance * 0.25, 5.0)
                    amt = max(amt, 1.0)
                    if amt <= balance and balance >= 1:
                        positions[symbol] = {'dir': 'short', 'entry': price, 'amt': amt, 'lev': lev}
                        balance -= amt

        # 平剩余
        for sym, pos in list(positions.items()):
            lp = float(test_df.iloc[-1]['close'])
            pnl_pct = ((pos['entry'] - lp) / pos['entry']) if pos['dir'] == 'short' else ((lp - pos['entry']) / pos['entry'])
            pnl_pct *= pos['lev']
            fee = pos['amt'] * pos['lev'] * 0.0005 * 2
            pnl = pos['amt'] * pnl_pct - fee
            balance += pos['amt'] + pnl
            all_trades.append({'pnl': pnl, 'dir': pos['dir'], 'reason': '收盘'})

    if not all_trades:
        return None
    
    wins = [t for t in all_trades if t['pnl'] > 0]
    losses = [t for t in all_trades if t['pnl'] <= 0]
    total_pnl = sum(t['pnl'] for t in all_trades)
    avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0

    return {
        'trades': len(all_trades),
        'win_rate': len(wins) / len(all_trades),
        'total_pnl': total_pnl,
        'balance': balance,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': abs(sum(t['pnl'] for t in wins) / sum(t['pnl'] for t in losses)) if losses and sum(t['pnl'] for t in losses) != 0 else 0,
        'long_trades': len([t for t in all_trades if t['dir'] == 'long']),
        'short_trades': len([t for t in all_trades if t['dir'] == 'short']),
    }


print("加载数据...")
all_data = load_all_data()

# cond7 + 不同盈亏比组合
tests = [
    # (name, sl, tp, min_cond, direction)
    ("cond7_baseline",     0.008, 0.005, 7, 'both'),
    ("cond7_tp08",         0.008, 0.008, 7, 'both'),
    ("cond7_tp10",         0.008, 0.010, 7, 'both'),
    ("cond7_tp12",         0.008, 0.012, 7, 'both'),
    ("cond7_tp15",         0.008, 0.015, 7, 'both'),
    ("cond7_tp20",         0.008, 0.020, 7, 'both'),
    ("cond7_long_only",    0.008, 0.005, 7, 'long_only'),
    ("cond7_long_tp10",    0.008, 0.010, 7, 'long_only'),
    ("cond7_long_tp15",    0.008, 0.015, 7, 'long_only'),
    ("cond7_long_tp20",    0.008, 0.020, 7, 'long_only'),
    ("cond7_sl06_tp10",    0.006, 0.010, 7, 'both'),
    ("cond7_sl06_tp12",    0.006, 0.012, 7, 'both'),
    ("cond7_sl06_tp15",    0.006, 0.015, 7, 'both'),
    ("cond7_sl10_tp15",    0.010, 0.015, 7, 'both'),
    ("cond7_sl10_tp20",    0.010, 0.020, 7, 'both'),
    ("cond7_sl10_tp25",    0.010, 0.025, 7, 'both'),
    ("cond6_baseline",     0.008, 0.005, 6, 'both'),
    ("cond6_tp10",         0.008, 0.010, 6, 'both'),
    ("cond6_tp12",         0.008, 0.012, 6, 'both'),
    ("cond6_tp15",         0.008, 0.015, 6, 'both'),
    ("cond6_long_only",    0.008, 0.005, 6, 'long_only'),
    ("cond6_long_tp10",    0.008, 0.010, 6, 'long_only'),
    ("cond6_long_tp15",    0.008, 0.015, 6, 'long_only'),
    ("cond6_long_tp20",    0.008, 0.020, 6, 'long_only'),
    ("cond6_long_tp25",    0.008, 0.025, 6, 'long_only'),
]

results = []
for name, sl, tp, mc, dm in tests:
    print(f"  {name}: SL={sl:.1%} TP={tp:.1%} 条件>={mc} {dm}", end=' ... ')
    r = backtest(all_data, sl, tp, mc, dm)
    if r:
        r['name'] = name
        results.append(r)
        print(f"{r['trades']}笔 胜率={r['win_rate']:.1%} 盈亏={r['total_pnl']:+.2f}U PF={r['profit_factor']:.2f} L={r['long_trades']} S={r['short_trades']}")
    else:
        print("无交易")

# 排序输出
print(f"\n{'='*80}")
print(f"{'名称':<25} {'笔数':>4} {'胜率':>6} {'盈亏':>8} {'余额':>7} {'PF':>5} {'平均赢':>7} {'平均亏':>7}")
print(f"{'='*80}")
results.sort(key=lambda x: x.get('total_pnl', -999), reverse=True)
for r in results:
    marker = " ★" if r['win_rate'] >= 0.5 and r['total_pnl'] > 0 else ""
    print(f"  {r['name']:<23} {r['trades']:>4} {r['win_rate']:>5.1%} {r['total_pnl']:>+7.2f}U {r['balance']:>6.2f}U {r['profit_factor']:>5.2f} {r['avg_win']:>+7.4f} {r['avg_loss']:>+7.4f}{marker}")

# 保存
with open('/tmp/sweep2_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
