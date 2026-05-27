"""趋势跟踪 + 紧止损策略"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml

with open('config/strategy.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

csv_file = 'data/BTC_15m.csv'
df = pd.read_csv(csv_file)

# Pre-calculate indicators
df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
df['ema55'] = df['close'].ewm(span=55, adjust=False).mean()

# RSI
rsi_period = 14
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

# ATR
high_low = df['high'] - df['low']
high_close = (df['high'] - df['close'].shift()).abs()
low_close = (df['low'] - df['close'].shift()).abs()
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df['atr'] = tr.rolling(14).mean()

# Volume SMA
df['volume_sma'] = df['volume'].rolling(20).mean()

# Strategy params
sl_pct = 0.005   # 0.5% stop loss
tp_pct = 0.01    # 1% take profit (2:1 R:R)

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
    candle = df.iloc[i]
    prev = df.iloc[i-1]

    price = float(candle['close'])
    ema9 = float(candle['ema9'])
    ema21 = float(candle['ema21'])
    ema55 = float(candle['ema55'])
    rsi = float(candle.get('rsi', 50))
    volume = float(candle['volume'])
    volume_sma = float(candle.get('volume_sma', volume))

    # Condition 1: Strong uptrend (EMA alignment)
    if not (ema9 > ema21 > ema55):
        continue

    # Condition 2: Price pullback to EMA9 area (within 0.5%)
    ema9_distance = (price - ema9) / ema9
    if ema9_distance > 0.005 or ema9_distance < -0.005:
        continue

    # Condition 3: RSI not overbought
    if rsi > 65:
        continue

    # Condition 4: Volume confirmation
    if volume < volume_sma * 0.8:
        continue

    # Condition 5: Previous candle was bullish
    prev_close = float(prev['close'])
    prev_open = float(prev['open'])
    if prev_close <= prev_open:
        continue

    # Calculate SL/TP
    entry = price
    sl = entry * (1 - sl_pct)
    tp = entry * (1 + tp_pct)

    # Position sizing (2% risk)
    risk_amount = balance * 0.02
    stop_distance_pct = sl_pct
    notional = risk_amount / stop_distance_pct
    amount_usdt = notional / 5  # 5x leverage
    amount_usdt = min(amount_usdt, balance * 0.3)

    if amount_usdt < 1:
        continue

    # Simulate trade
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

    trades.append(outcome)

# Print results
total = wins + losses
if total == 0:
    print("No trades generated")
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
    print("Trend Following + Tight SL Strategy")
    print("=" * 60)
    print(f"Total Trades: {total}")
    print(f"Win Rate: {win_rate:.2f}% ({wins}W / {losses}L)")
    print(f"Risk/Reward: {rr_ratio:.2f}")
    print(f"")
    print(f"Avg Win: +{avg_win:.4f}U")
    print(f"Avg Loss: {avg_loss:.4f}U")
    print(f"")
    print(f"Total PnL: {total_pnl:.4f}U ({total_pnl/initial_balance*100:.2f}%)")
    print(f"")
    print(f"Breakeven WR: {breakeven_wr:.2f}%")
    print(f"Actual vs Breakeven: {win_rate:.2f}% vs {breakeven_wr:.2f}%")
    print(f"")
    print(f"Max Drawdown: {max_drawdown*100:.2f}%")
    print(f"")
    print("=" * 60)
    print("Target Check")
    print("=" * 60)

    wr_ok = win_rate >= 45
    pnl_ok = total_pnl > 0
    rr_ok = rr_ratio >= 1.0

    print(f"Win Rate Target: >= 45% -> Actual: {win_rate:.1f}% {'PASS' if wr_ok else 'FAIL'}")
    print(f"Profit Target: > 0U -> Actual: {total_pnl:.4f}U {'PASS' if pnl_ok else 'FAIL'}")
    print(f"R:R Target: >= 1.0 -> Actual: {rr_ratio:.2f} {'PASS' if rr_ok else 'FAIL'}")
    print()

    if wr_ok and pnl_ok and rr_ok:
        print("Strategy PASS!")
    else:
        print("Needs optimization")
