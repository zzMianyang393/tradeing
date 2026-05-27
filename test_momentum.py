"""测试动量回踩策略"""

import pandas as pd
from pathlib import Path
import yaml
from strategy.momentum import MomentumStrategy
from risk.position_sizer import PositionSizer

with open('config/strategy.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Override strategy params
config['rules'] = {'min_conditions': 3, 'direction_mode': 'long_only'}

csv_file = 'data/BTC_15m.csv'
df = pd.read_csv(csv_file)

strategy = MomentumStrategy(config)
pos_sizer = PositionSizer(config)

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

print("=" * 60)
print("动量回踩策略回测")
print("=" * 60)

for i in range(60, len(df) - 48):
    window = df.iloc[:i+1]
    signal = strategy.analyze(window)

    if signal is None:
        continue

    price = signal.entry_price
    atr = float(window.iloc[-1].get('atr', price * 0.01))

    # Position sizing
    stop_distance_pct = (price - signal.stop_loss) / price
    position = pos_sizer.calculate_position(
        balance=balance,
        entry_price=price,
        stop_distance_pct=stop_distance_pct,
        signal_strength=signal.strength,
        current_positions=0,
    )

    if position.amount_usdt <= 0:
        continue

    # Simulate trade
    entry = price
    sl = signal.stop_loss
    tp = signal.take_profit

    outcome = None
    exit_reason = None

    for j in range(i+1, min(i+48, len(df))):
        high = float(df.iloc[j]['high'])
        low = float(df.iloc[j]['low'])

        # Check SL first (conservative)
        if low <= sl:
            pnl_pct = (sl - entry) / entry
            pnl = position.amount_usdt * pnl_pct * position.leverage
            fee = position.amount_usdt * 0.0005
            pnl -= fee
            outcome = pnl
            exit_reason = '止损'
            break

        # Check TP
        if high >= tp:
            pnl_pct = (tp - entry) / entry
            pnl = position.amount_usdt * pnl_pct * position.leverage
            fee = position.amount_usdt * 0.0005
            pnl -= fee
            outcome = pnl
            exit_reason = '止盈'
            break

    # Timeout - exit at market
    if outcome is None:
        exit_price = float(df.iloc[min(i+48, len(df)-1)]['close'])
        pnl_pct = (exit_price - entry) / entry
        pnl = position.amount_usdt * pnl_pct * position.leverage
        fee = position.amount_usdt * 0.0005
        pnl -= fee
        outcome = pnl
        exit_reason = '超时'

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
        'idx': i,
        'entry': entry,
        'sl': sl,
        'tp': tp,
        'pnl': outcome,
        'reason': exit_reason,
        'balance': balance,
        'leverage': position.leverage,
    })

    # Print progress every 100 trades
    if len(trades) % 100 == 0:
        win_rate = wins * 100 / len(trades) if trades else 0
        print(f"  已完成 {len(trades)} 笔交易, 胜率={win_rate:.1f}%, 余额={balance:.4f}U")

# Print results
print()
print("=" * 60)
print("回测结果")
print("=" * 60)

total = wins + losses
if total == 0:
    print("没有产生交易")
else:
    win_rate = wins * 100 / total
    avg_win = total_win_pnl / wins if wins > 0 else 0
    avg_loss = total_loss_pnl / losses if losses > 0 else 0
    rr_ratio = abs(avg_loss / avg_win) if avg_win > 0 else 0
    total_pnl = balance - initial_balance
    avg_pnl = total_pnl / total

    # Breakeven win rate
    if avg_win > 0 and avg_loss < 0:
        breakeven_wr = abs(avg_loss) / (avg_win + abs(avg_loss)) * 100
    else:
        breakeven_wr = 50

    print(f"总交易数: {total}")
    print(f"胜率: {win_rate:.2f}% ({wins}胜 / {losses}负)")
    print(f"盈亏比: {rr_ratio:.2f}")
    print(f"")
    print(f"平均盈利: +{avg_win:.4f}U")
    print(f"平均亏损: {avg_loss:.4f}U")
    print(f"")
    print(f"总盈亏: {total_pnl:.4f}U ({total_pnl/initial_balance*100:.2f}%)")
    print(f"平均盈亏: {avg_pnl:.4f}U/笔")
    print(f"")
    print(f"盈亏平衡胜率: {breakeven_wr:.2f}%")
    print(f"实际 vs 平衡: {win_rate:.2f}% vs {breakeven_wr:.2f}%")
    print(f"")
    print(f"最大回撤: {max_drawdown*100:.2f}%")
    print(f"")
    print(f"夏普比率: {avg_pnl / (total_pnl/total) if total_pnl != 0 else 0:.2f}")

    # Exit reason breakdown
    print()
    print("退出原因统计:")
    reasons = {}
    for t in trades:
        r = t['reason']
        if r not in reasons:
            reasons[r] = {'count': 0, 'pnl': 0, 'wins': 0}
        reasons[r]['count'] += 1
        reasons[r]['pnl'] += t['pnl']
        if t['pnl'] > 0:
            reasons[r]['wins'] += 1

    for reason, stats in sorted(reasons.items(), key=lambda x: x[1]['pnl'], reverse=True):
        wr = stats['wins'] * 100 / stats['count'] if stats['count'] > 0 else 0
        print(f"  {reason}: {stats['count']}笔, 胜率={wr:.1f}%, PnL={stats['pnl']:.4f}U")

    # Target check
    print()
    print("=" * 60)
    print("目标检查")
    print("=" * 60)

    wr_target = 55
    wr_ok = win_rate >= wr_target - 5
    pnl_ok = total_pnl > 0
    rr_ok = rr_ratio >= 1.0

    print(f"胜率目标: {wr_target}% ± 5% → 实际: {win_rate:.1f}% {'✓' if wr_ok else '✗'}")
    print(f"盈利目标: > 0U → 实际: {total_pnl:.4f}U {'✓' if pnl_ok else '✗'}")
    print(f"盈亏比目标: ≥ 1.0 → 实际: {rr_ratio:.2f} {'✓' if rr_ok else '✗'}")
    print()

    if wr_ok and pnl_ok and rr_ok:
        print("策略达标！")
    else:
        print("需要优化")
