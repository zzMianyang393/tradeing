"""波动率分组策略 - 根据ATR自动选择SL/TP"""

import yaml, pandas as pd, numpy as np
from pathlib import Path

def load_config():
    config_dir = Path('config')
    with open(config_dir / 'settings.yaml', 'r', encoding='utf-8') as f:
        settings = yaml.safe_load(f)
    with open(config_dir / 'strategy.yaml', 'r', encoding='utf-8') as f:
        strategy = yaml.safe_load(f)
    return {**settings, **strategy}

from backtest.engine import BacktestEngine

all_coins = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE', 'ADA', 'AVAX',
             'LINK', 'DOT', 'UNI', 'LTC', 'ATOM', 'FIL', 'APT', 'ARB']

def get_atr_group(atr_pct):
    """根据ATR%分组"""
    if atr_pct < 0.20:
        return 'low'   # 低波动: BTC
    elif atr_pct < 0.35:
        return 'medium'  # 中波动: ETH, SOL, BNB
    else:
        return 'high'   # 高波动: DOT, ATOM, XRP

# 不同波动率组的最优参数
group_params = {
    'low': {  # BTC等低波动币
        'stop_loss': {'fixed_pct': 0.025, 'trailing_activate': 0, 'trailing_callback': 0},
        'take_profit': {'fixed_pct': 0.0375, 'partial_close_trigger': 0.03, 'partial_close_pct': 0.5},
    },
    'medium': {  # ETH, SOL等中波动币
        'stop_loss': {'atr_multiplier': 6, 'trailing_activate': 0, 'trailing_callback': 0},
        'take_profit': {'atr_multiplier': 9, 'partial_close_trigger': 0, 'partial_close_pct': 0.5},
    },
    'high': {  # DOT, ATOM等高波动币
        'stop_loss': {'fixed_pct': 0.025, 'trailing_activate': 0, 'trailing_callback': 0},
        'take_profit': {'fixed_pct': 0.0375, 'partial_close_trigger': 0.03, 'partial_close_pct': 0.5},
    },
}

print('波动率分组策略测试:')
header = f"{'币种':>6} {'ATR%':>6} {'分组':>8} {'交易':>4} {'胜率':>6} {'盈亏':>9} {'适合':>4}"
print(header)
print('-' * 60)

results = []
for sym in all_coins:
    csv = f'data/{sym}_15m.csv'
    if not Path(csv).exists():
        continue
    df = pd.read_csv(csv)
    test_bars = min(8640, len(df) - 200)
    if test_bars < 100:
        continue

    # 计算ATR%
    test_df = df.iloc[-test_bars:]
    high = test_df['high']
    low = test_df['low']
    close = test_df['close']
    tr = pd.concat([high-low, abs(high-close.shift()), abs(low-close.shift())], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    atr_pct = atr / close.iloc[-1] * 100

    # 根据ATR分组选择参数
    group = get_atr_group(atr_pct)
    params = group_params[group]

    config = load_config()
    config['general']['timeframe'] = '15m'
    config['rules']['strategy_mode'] = 'mean_reversion'
    config.update(params)

    train_df = df.iloc[:-test_bars]

    engine = BacktestEngine(config)
    stats = engine.run(f'{sym}/USDT:USDT', test_df, train_df)

    trades = stats.get('trades', [])
    closes = [t for t in trades if t.get('action') == 'close']
    opens = [t for t in trades if t.get('action') == 'open']

    if not closes:
        line = f"{sym:>6} {atr_pct:>5.2f}% {group:>8} {0:>4} {'N/A':>6} {0:>9.4f}"
        print(line)
        continue

    wins = [t for t in closes if t.get('pnl', 0) > 0]
    total_pnl = sum(t.get('pnl', 0) for t in closes)
    wr = len(wins) / len(closes)

    suitable = '[OK]' if wr >= 0.5 and total_pnl > 0 else ''
    line = f"{sym:>6} {atr_pct:>5.2f}% {group:>8} {len(opens):>4} {wr:>5.0%} {total_pnl:>+9.4f} {suitable}"
    print(line)
    results.append({'symbol': sym, 'group': group, 'pnl': total_pnl, 'wr': wr, 'trades': len(opens), 'atr': atr_pct})

# 汇总
print('-' * 60)
suitable = [r for r in results if r['wr'] >= 0.5 and r['pnl'] > 0]
total_pnl = sum(r['pnl'] for r in results)
suitable_pnl = sum(r['pnl'] for r in suitable)

prof_symbols = [r["symbol"] for r in suitable]
print(f'适合的币种: {prof_symbols} ({len(suitable)}个)')
print(f'全部合计: {total_pnl:+.4f}U')
print(f'适合币种合计: {suitable_pnl:+.4f}U')

# 按分组统计
for group in ['low', 'medium', 'high']:
    group_results = [r for r in results if r['group'] == group]
    if group_results:
        group_pnl = sum(r['pnl'] for r in group_results)
        group_profitable = [r for r in group_results if r['pnl'] > 0]
        print(f'{group}分组: {len(group_results)}个币, 盈利{len(group_profitable)}个, 合计{group_pnl:+.4f}U')
