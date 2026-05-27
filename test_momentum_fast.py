"""快速测试动量回踩策略"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml

with open('config/strategy.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

csv_file = 'data/BTC_15m.csv'
df = pd.read_csv(csv_file)

# Pre-calculate indicators
ema_fast = 9
ema_slow = 55

df['ema_fast'] = df['close'].ewm(span=ema_fast, adjust=False).mean()
df['ema_slow'] = df['close'].ewm(span=ema_slow, adjust=False).mean()

# ATR
high_low = df['high'] - df['low']
high_close = (df['high'] - df['close'].shift()).abs()
low_close = (df['low'] - df['close'].shift()).abs()
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df['atr'] = tr.rolling(14).mean()

# Volume SMA
df['volume_sma'] = df['volume'].rolling(20).mean()

# Strategy params
min_candle_pct = 0.005
volume_spike = 1.5
pullback_pct = 0.3
sl_atr_mult = 1.5
rr_ratio = 2.0

# Run backtest
balance = 10.0
initial_balance = balance
trades = []
wins = 0
losses = 0
total_win_pnl = 0
total_loss_pnl = 0
max_balance = balance
max_drawdown = 0

for i in range(60, len(df) - 48):
    # Get data
    candle = df.iloc[i]
    prev = df.iloc[i-1]

    price = float(candle['close'])
    atr = float(candle.get('atr', price * 0.01))

    if atr <= 0:
        continue

    # Condition 1: Previous candle is strong bullish
    prev_body = float(prev['close']) - float(prev['open'])
    prev_range = float(prev['high']) - float(prev['low'])
    prev_body_pct = prev_body / float(prev['open']) if float(prev['open']) > 0 else 0

    if prev_body <= 0 or prev_body_pct < min_candle_pct or prev_range <= 0:
        continue

    # Condition 2: Previous candle has volume spike
    prev_volume = float(prev['volume'])
    volume_sma = float(prev.get('volume_sma', prev_volume))
    if volume_sma > 0 and prev_volume < volume_sma * volume_spike:
        continue

    # Condition 3: Current candle is pullback
    curr_close = float(candle['close'])
    curr_low = float(candle['low'])
    prev_high = float(prev['high'])

    if curr_close >= prev_high:
        continue

    # Pullback level check
    prev_body_top = float(prev['close'])
    prev_body_bottom = float(prev['open'])
    pullback_range = (prev_body_top - prev_body_bottom) * pullback_pct

    if curr_close > prev_body_top - pullback_range:
        continue
    if curr_close < prev_body_bottom + pullback_range:
        continue

    # Condition 4: EMA trend up
    ema_fast_val = float(candle.get('ema_fast', 0))
    ema_slow_val = float(candle.get('ema_slow', 0))
    if ema_fast_val <= ema_slow_val:
        continue

    # Condition 5: Price near EMA fast
    ema_distance = abs(curr_close - ema_fast_val) / ema_fast_val if ema_fast_val > 0 else 1
    if ema_distance > 0.01:
        continue

    # Calculate SL/TP
    sl = curr_low - atr * 0.5
    stop_distance = curr_close - sl

    if stop_distance < atr * sl_atr_mult:
        sl = curr_close - atr * sl_atr_mult
        stop_distance = curr_close - sl

    tp = curr_close + stop_distance * rr_ratio
    stop_distance_pct = stop_distance / curr_close

    # Position sizing (simplified: 2% risk per trade)
    risk_amount = balance * 0.02
    if stop_distance_pct <= 0:
        continue
    notional = risk_amount / stop_distance_pct
    amount_usdt = notional / 5  # 5x leverage
    amount_usdt = min(amount_usdt, balance * 0.3)

    if amount_usdt < 1:
        continue

    # Simulate trade
    entry = curr_close
    outcome = None

    for j in range(i+1, min(i+48, len(df))):
        high = float(df.iloc[j]['high'])
        low = float(df.iloc[j]['low'])

        if low <= sl:
            pnl_pct = (sl - entry) / entry
            pnl = amount_usdt * pnl_pct * 5
            fee = amount_usdt * 0.0005
            pnl -= fee
            outcome = pnl
            break
        if high >= tp:
            pnl_pct = (tp - entry) / entry
            pnl = amount_usdt * pnl_pct * 5
            fee = amount_usdt * 0.0005
            pnl -= fee
            outcome = pnl
            break

    if outcome is None:
        exit_price = float(df.iloc[min(i+48, len(df)-1)]['close'])
        pnl_pct = (exit_price - entry) / entry
        pnl = amount_usdt * pnl_pct * 5
        fee = amount_usdt * 0.0005
        pnl -= fee
        outcome = pnl

    balance += outcome
    max_balance = max(max_balance, balance)
    drawdown = (max_balance - balance) / max_balance
    max_drawdown = max(max_drawdown, drawdown)

    if outcome > 0:
        wins += 1
        total_win_pnl += outcome
    else:
        losses += 1
        total_loss_pnl += outcome

    trades.append({
        'pnl': outcome,
        'reason': '止损' if outcome < 0 else '止盈',
    })

# Print results
total = wins + losses
if total == 0:
    print("没有产生交易")
else:
    win_rate = wins * 100 / total
    avg_win = total_win_pnl / wins if wins > 0 else 0
    avg_loss = total_loss_pnl / losses if losses > 0 else 0
    rr_ratio = abs(avg_loss / avg_win) if avg_win > 0 else 0
    total_pnl = balance - initial_balance

    # Breakeven win rate
    if avg_win > 0 and avg_loss < 0:
        breakeven_wr = abs(avg_loss) / (avg_win + abs(avg_loss)) * 100
    else:
        breakeven_wr = 50

    print("=" * 60)
    print("动量回踩策略回测结果")
    print("=" * 60)
    print(f"总交易数: {total}")
    print(f"胜率: {win_rate:.2f}% ({wins}胜 / {losses}负)")
    print(f"盈亏比: {rr_ratio:.2f}")
    print(f"")
    print(f"平均盈利: +{avg_win:.4f}U")
    print(f"平均亏损: {avg_loss:.4f}U")
    print(f"")
    print(f"总盈亏: {total_pnl:.4f}U ({total_pnl/initial_balance*100:.2f}%)")
    print(f"")
    print(f"盈亏平衡胜率: {breakeven_wr:.2f}%")
    print(f"实际 vs 平衡: {win_rate:.2f}% vs {breakeven_wr:.2f}%")
    print(f"")
    print(f"最大回撤: {max_drawdown*100:.2f}%")
    print(f"")
    print("=" * 60)
    print("目标检查")
    print("=" * 60)

    wr_ok = win_rate >= 50
    pnl_ok = total_pnl > 0
    rr_ok = rr_ratio >= 1.0

    print(f"胜率目标: ≥ 50% → 实际: {win_rate:.1f}% {'✓' if wr_ok else '✗'}")
    print(f"盈利目标: > 0U → 实际: {total_pnl:.4f}U {'✓' if pnl_ok else '✗'}")
    print(f"盈亏比目标: ≥ 1.0 → 实际: {rr_ratio:.2f} {'✓' if rr_ok else '✗'}")
    print()

    if wr_ok and pnl_ok and rr_ok:
        print("策略达标！")
    else:
        print("需要优化")
