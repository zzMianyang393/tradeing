"""15m信号 + 1h确认 组合策略"""

import yaml, pandas as pd, numpy as np
from pathlib import Path

def load_config():
    config_dir = Path('config')
    with open(config_dir / 'settings.yaml', 'r', encoding='utf-8') as f:
        settings = yaml.safe_load(f)
    with open(config_dir / 'strategy.yaml', 'r', encoding='utf-8') as f:
        strategy = yaml.safe_load(f)
    return {**settings, **strategy}

from strategy.indicators import TechnicalIndicators
from backtest.engine import BacktestEngine
from datetime import datetime

all_coins = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE', 'ADA', 'AVAX',
             'LINK', 'DOT', 'UNI', 'LTC', 'ATOM', 'FIL', 'APT', 'ARB']

def resample_1h(df_15m):
    df = df_15m.copy()
    if "timestamp" in df.columns:
        df.index = pd.to_datetime(df["timestamp"], unit="ms")
    else:
        df.index = pd.to_datetime(df.iloc[:, 0])
    df_1h = df[["open", "high", "low", "close", "volume"]].resample("1h").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna().reset_index()
    df_1h.rename(columns={df_1h.columns[0]: "timestamp"}, inplace=True)
    if df_1h["timestamp"].dtype != "int64":
        df_1h["timestamp"] = df_1h["timestamp"].astype(np.int64) // 10**6
    return df_1h

def get_1h_trend(df_1h, current_ts):
    """获取当前时间的1h趋势"""
    if df_1h is None or len(df_1h) < 60:
        return "neutral"

    mask = df_1h["timestamp"] <= current_ts
    available = df_1h[mask]
    if len(available) < 30:
        return "neutral"

    close = available["close"]
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema55 = close.ewm(span=55, adjust=False).mean()

    ef, em, es = ema9.iloc[-1], ema21.iloc[-1], ema55.iloc[-1]
    price = close.iloc[-1]

    if ef > em > es and price > es:
        return "bullish"
    elif ef < em < es and price < es:
        return "bearish"
    return "neutral"

print('15m信号 + 1h确认 策略测试:')
header = f"{'币种':>6} {'交易':>4} {'胜率':>6} {'盈亏':>9} {'达标':>4}"
print(header)
print('-' * 40)

results = []
for sym in all_coins:
    csv = f'data/{sym}_15m.csv'
    if not Path(csv).exists():
        continue

    df_15m = pd.read_csv(csv)
    df_1h = resample_1h(df_15m)

    test_bars = min(8640, len(df_15m) - 200)
    train_df = df_15m.iloc[:-test_bars]
    test_df = df_15m.iloc[-test_bars:]

    # 计算1h指标
    indicators = TechnicalIndicators({})
    df_1h_ind = indicators.calculate(df_1h)

    config = load_config()
    config['general']['timeframe'] = '15m'
    config['rules']['strategy_mode'] = 'mean_reversion'
    config['rules']['min_conditions'] = 6
    config['stop_loss'] = {'fixed_pct': 0.025, 'trailing_activate': 0, 'trailing_callback': 0}
    config['take_profit'] = {'fixed_pct': 0.0375, 'partial_close_trigger': 0.03, 'partial_close_pct': 0.5}

    engine = BacktestEngine(config)
    stats = engine.run(f'{sym}/USDT:USDT', test_df, train_df, htf_data=df_1h_ind)

    trades = stats.get('trades', [])
    closes = [t for t in trades if t.get('action') == 'close']
    opens = [t for t in trades if t.get('action') == 'open']

    if closes:
        wins = [t for t in closes if t.get('pnl', 0) > 0]
        total_pnl = sum(t.get('pnl', 0) for t in closes)
        wr = len(wins) / len(closes)

        suitable = '[OK]' if wr >= 0.55 and total_pnl > 0 else ''
        line = f"{sym:>6} {len(opens):>4} {wr:>5.0%} {total_pnl:>+9.4f} {suitable}"
        print(line)
        results.append({'symbol': sym, 'pnl': total_pnl, 'wr': wr})

# 汇总
print('-' * 40)
suitable = [r for r in results if r['wr'] >= 0.55 and r['pnl'] > 0]
total_pnl = sum(r['pnl'] for r in results)
suitable_pnl = sum(r['pnl'] for r in suitable)
prof_symbols = [r['symbol'] for r in suitable]
print(f'达标币种: {prof_symbols} ({len(suitable)}个)')
print(f'全部合计: {total_pnl:+.4f}U')
print(f'达标合计: {suitable_pnl:+.4f}U')
