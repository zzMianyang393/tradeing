"""快速测试超跌反弹策略"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml

with open('config/strategy.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

csv_file = 'data/BTC_15m.csv'
df = pd.read_csv(csv_file)

# Pre-calculate indicators
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

# EMA
df['ema_fast'] = df['close'].ewm(span=9, adjust=False).mean()

# Strategy params
rsi_oversold = 25
min_drop_pct = 0.03
lookback = 3
sl_atr_mult = 1.5
tp_rr = 1.5  # R:R ratio

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

for i in range(lookback + 15, len(df) - 48):
    latest = df.iloc[i]
    price = float(latest['close'])
    atr = float(latest.get('atr', price * 0.01))

    if atr <= 0:
        continue

    rsi = float(latest.get('rsi', 50))

    # Condition 1: RSI extreme oversold
    if rsi > rsi_oversold:
        continue

    # Condition 2: Recent drop > 3%
    lookback_candles = df.iloc[-(lookback+1):-1] if i >= lookback else df.iloc[:i]
    if len(lookback_candles) < lookback:
        continue

    start_price = float(lookback_candles.iloc[0]['open'])
    end_price = float(latest['close'])
    drop_pct = (start_price - end_price) / start_price

    if drop_pct < min_drop_pct:
        continue

    # Condition 3: Current candle is stabilizing (bullish or doji)
    curr_open = float(latest['open'])
    curr_close = float(latest['close'])
    curr_body = abs(curr_close - curr_open)
    curr_range = float(latest['high']) - float(latest['low'])

    is_bullish = curr_close > curr_open
    is_doji = curr_body < curr_range * 0.3 if curr_range > 0 else False

    if not (is_bullish or is_doji):
        continue

    # Condition 4: Lower shadow (buying support)
    curr_low = float(latest['low'])
    lower_shadow = min(curr_open, curr_close) - curr_low
    if curr_range > 0 and lower_shadow < curr_range * 0.2:
        continue

    # Condition 5: Price below EMA (oversold)
    ema_fast = float(latest.get('ema_fast', price))
    if price > ema_fast:
        continue

    # Calculate SL/TP
    sl = curr_low - atr * sl_atr_mult
    stop_distance = price - sl
    tp = price + stop_distance * tp_rr
    stop_distance_pct = stop_distance / price

    # Position sizing (2% risk)
    risk_amount = balance * 0.02
    if stop_distance_pct <= 0:
        continue
    notional = risk_amount / stop_distance_pct
    amount_usdt = notional / 5  # 5x leverage
    amount_usdt = min(amount_usdt, balance * 0.3)

    if amount_usdt < 1:
        continue

    # Simulate trade
    entry = price
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
    print("Mean Reversion Strategy Backtest")
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

    wr_ok = win_rate >= 50
    pnl_ok = total_pnl > 0
    rr_ok = rr_ratio >= 1.0

    print(f"Win Rate Target: >= 50% -> Actual: {win_rate:.1f}% {'PASS' if wr_ok else 'FAIL'}")
    print(f"Profit Target: > 0U -> Actual: {total_pnl:.4f}U {'PASS' if pnl_ok else 'FAIL'}")
    print(f"R:R Target: >= 1.0 -> Actual: {rr_ratio:.2f} {'PASS' if rr_ok else 'FAIL'}")
    print()

    if wr_ok and pnl_ok and rr_ok:
        print("Strategy PASS!")
    else:
        print("Needs optimization")
