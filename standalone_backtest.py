"""独立回测脚本 - 4h趋势过滤策略验证
直接从SQLite读取15m数据，自己计算4h趋势，独立执行回测。
不依赖BacktestEngine和SignalGenerator。
"""

import sys
sys.path.insert(0, '.')

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime


# ==================== 数据加载 ====================

def load_data(symbol):
    """从SQLite直接加载15m K线"""
    conn = sqlite3.connect('data/trading.db')
    df = pd.read_sql_query(
        'SELECT * FROM klines WHERE symbol=? AND timeframe="15m" ORDER BY timestamp ASC',
        conn, params=[symbol]
    )
    conn.close()
    df['timestamp'] = df['timestamp'].astype(int)
    return df


def compute_4h_trend(df):
    """从15m数据计算4h趋势 (EMA55)
    返回 {timestamp_ms: 'bullish'/'bearish'/'neutral'} 映射
    """
    df.index = pd.to_datetime(df['timestamp'], unit='ms')
    df_4h = df[['open', 'high', 'low', 'close', 'volume']].resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna()
    df_4h['ema55'] = df_4h['close'].ewm(span=55, adjust=False).mean()

    trend_map = {}
    for idx, row in df_4h.iterrows():
        ts = int(idx.timestamp() * 1000)
        if row['close'] > row['ema55'] * 1.01:
            trend_map[ts] = 'bullish'
        elif row['close'] < row['ema55'] * 0.99:
            trend_map[ts] = 'bearish'
        else:
            trend_map[ts] = 'neutral'
    return trend_map


def get_trend(trend_map, ts_ms):
    """查找给定时间戳对应的4h趋势"""
    for k in reversed(sorted(trend_map.keys())):
        if k <= ts_ms:
            return trend_map[k]
    return 'neutral'


# ==================== 技术指标 ====================

def calculate_indicators(df):
    """计算RSI和volume_ratio"""
    close = df['close']
    volume = df['volume']

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Volume ratio
    df['volume_sma'] = volume.rolling(20).mean()
    df['volume_ratio'] = volume / df['volume_sma'].replace(0, np.nan)

    return df


# ==================== 回测引擎 ====================

def run_backtest(df, sl=0.015, tp=0.015, rsi_long=40, rsi_short=60,
                 vol_thresh=1.0, position_size=2.0, leverage=5,
                 fee_rate=0.001, cooldown_bars=6, min_margin=1.0):
    """独立回测 - 4h趋势过滤 + RSI + 成交量

    策略逻辑:
      - 4h趋势bullish + 15m RSI<40 + volume>1.0 → 做多
      - 4h趋势bearish + 15m RSI>60 + volume>1.0 → 做空
      - SL=TP=1.5%, 杠杆5x, 手续费0.1% (双边)
    """
    skipped_min_margin = 0
    df = calculate_indicators(df.copy())
    trend_map = compute_4h_trend(df)
    df = df.reset_index(drop=True)

    in_pos = False
    pos = {}
    cooldown = 0
    trades = []
    returns = []
    balance = 10.0

    for i in range(60, len(df)):
        row = df.iloc[i]
        price = row['close']
        ts = row['timestamp']
        htf = get_trend(trend_map, ts)

        if cooldown > 0:
            cooldown -= 1
            # 冷却期内仍然检查持仓止损止盈（修复：避免持仓无法及时平仓）
            if in_pos:
                d = pos['dir']
                if d == 'long':
                    if row['low'] <= pos['sl']:
                        ret = ((pos['sl'] - pos['entry']) / pos['entry']) * leverage - fee_rate
                        pnl = position_size * ret
                        trades.append(pnl)
                        returns.append(ret)
                        balance += pnl
                        in_pos = False
                    elif row['high'] >= pos['tp']:
                        ret = ((pos['tp'] - pos['entry']) / pos['entry']) * leverage - fee_rate
                        pnl = position_size * ret
                        trades.append(pnl)
                        returns.append(ret)
                        balance += pnl
                        in_pos = False
                else:
                    if row['high'] >= pos['sl']:
                        ret = ((pos['entry'] - pos['sl']) / pos['entry']) * leverage - fee_rate
                        pnl = position_size * ret
                        trades.append(pnl)
                        returns.append(ret)
                        balance += pnl
                        in_pos = False
                    elif row['low'] <= pos['tp']:
                        ret = ((pos['entry'] - pos['tp']) / pos['entry']) * leverage - fee_rate
                        pnl = position_size * ret
                        trades.append(pnl)
                        returns.append(ret)
                        balance += pnl
                        in_pos = False
            continue

        # --- 检查现有持仓 ---
        if in_pos:
            d = pos['dir']
            if d == 'long':
                if row['low'] <= pos['sl']:
                    # 止损
                    ret = ((pos['sl'] - pos['entry']) / pos['entry']) * leverage - fee_rate
                    pnl = position_size * ret
                    trades.append(pnl)
                    returns.append(ret)
                    balance += pnl
                    in_pos = False
                    cooldown = cooldown_bars
                elif row['high'] >= pos['tp']:
                    # 止盈
                    ret = ((pos['tp'] - pos['entry']) / pos['entry']) * leverage - fee_rate
                    pnl = position_size * ret
                    trades.append(pnl)
                    returns.append(ret)
                    balance += pnl
                    in_pos = False
                    cooldown = cooldown_bars
            else:  # short
                if row['high'] >= pos['sl']:
                    # 止损
                    ret = ((pos['entry'] - pos['sl']) / pos['entry']) * leverage - fee_rate
                    pnl = position_size * ret
                    trades.append(pnl)
                    returns.append(ret)
                    balance += pnl
                    in_pos = False
                    cooldown = cooldown_bars
                elif row['low'] <= pos['tp']:
                    # 止盈
                    ret = ((pos['entry'] - pos['tp']) / pos['entry']) * leverage - fee_rate
                    pnl = position_size * ret
                    trades.append(pnl)
                    returns.append(ret)
                    balance += pnl
                    in_pos = False
                    cooldown = cooldown_bars

        # --- 检查新信号 ---
        if not in_pos and balance > position_size:
            rsi = row.get('rsi', 50)
            vol_r = row.get('volume_ratio', 1)

            if pd.isna(rsi) or pd.isna(vol_r):
                continue

            # 最小保证金检查（与OKX模拟盘对齐）
            if position_size < min_margin:
                skipped_min_margin += 1
                continue

            if htf == 'bullish' and rsi < rsi_long and vol_r > vol_thresh:
                in_pos = True
                pos = {
                    'dir': 'long',
                    'entry': price,
                    'sl': price * (1 - sl),
                    'tp': price * (1 + tp),
                }
            elif htf == 'bearish' and rsi > rsi_short and vol_r > vol_thresh:
                in_pos = True
                pos = {
                    'dir': 'short',
                    'entry': price,
                    'sl': price * (1 + sl),
                    'tp': price * (1 - tp),
                }

    # 计算统计
    wins = sum(1 for t in trades if t > 0)
    wr = wins / len(trades) if trades else 0
    total_pnl = sum(trades)
    sharpe = 0
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = np.sqrt(252 * 96) * np.mean(returns) / np.std(returns)

    if trades:
        cumsum = np.cumsum(trades)
        peak = np.maximum.accumulate(cumsum)
        dd = peak - cumsum
        max_dd = np.max(dd) / 10.0 if np.max(dd) > 0 else 0
    else:
        max_dd = 0

    return {
        'trades': len(trades),
        'win_rate': wr,
        'total_pnl': total_pnl,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'balance': balance,
        'wins': wins,
        'losses': len(trades) - wins,
        'skipped_min_margin': skipped_min_margin,
    }


# ==================== 主函数 ====================

def main():
    import sys
    oos_days = int(sys.argv[1]) if len(sys.argv) > 1 else 7

    coins = [
        'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'DOGE/USDT:USDT',
        'XRP/USDT:USDT', 'ADA/USDT:USDT', 'AVAX/USDT:USDT', 'DOT/USDT:USDT',
        'LINK/USDT:USDT', 'UNI/USDT:USDT', 'ARB/USDT:USDT', 'OP/USDT:USDT',
        'SUI/USDT:USDT', 'APT/USDT:USDT', 'NEAR/USDT:USDT',
    ]

    # 样本外结束时间 = 当前最新数据往前推oos_days天
    from datetime import timezone
    now_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    oos_start = now_ts - oos_days * 24 * 3600 * 1000
    split_ts = oos_start  # 样本内/样本外分界线

    SL = 0.015
    TP = 0.015

    oos_start_dt = datetime.utcfromtimestamp(oos_start / 1000).strftime('%Y-%m-%d')
    oos_end_dt = datetime.utcfromtimestamp(now_ts / 1000).strftime('%Y-%m-%d')

    print("=" * 80)
    print(f"Strategy: 4h Trend Filter (SL={SL:.1%}, TP={TP:.1%})")
    print(f"Out-of-sample: {oos_start_dt} -> {oos_end_dt} ({oos_days} days)")
    print("=" * 80)

    # ========== IN-SAMPLE ==========
    print("\n📊 IN-SAMPLE BACKTEST")
    print("-" * 60)
    is_total_t = 0
    is_total_w = 0
    is_total_pnl = 0
    is_profitable = 0
    is_skipped = 0

    for coin in coins:
        df = load_data(coin)
        if df.empty:
            name = coin.split('/')[0]
            print(f"  ⚠️ {name:6s}: NO DATA")
            continue
        train = df[df['timestamp'] < split_ts].copy()
        result = run_backtest(train, sl=SL, tp=TP)

        is_total_t += result['trades']
        is_total_w += result['wins']
        is_total_pnl += result['total_pnl']
        is_skipped += result['skipped_min_margin']
        if result['total_pnl'] > 0:
            is_profitable += 1

        name = coin.split('/')[0]
        status = '✅' if result['total_pnl'] > 0 else '⚠️'
        skip_info = f", skipped={result['skipped_min_margin']}" if result['skipped_min_margin'] else ""
        print(f"  {status} {name:6s}: trades={result['trades']:4d}, WR={result['win_rate']:5.1%}, "
              f"PnL={result['total_pnl']:+8.4f}, Sharpe={result['sharpe']:7.2f}{skip_info}")

    is_wr = is_total_w / is_total_t if is_total_t else 0

    # ========== OUT-OF-SAMPLE ==========
    print(f"\n📊 OUT-OF-SAMPLE BACKTEST ({oos_days} days: {oos_start_dt} -> {oos_end_dt})")
    print("-" * 60)
    oos_total_t = 0
    oos_total_w = 0
    oos_total_pnl = 0
    oos_profitable = 0
    oos_skipped = 0

    for coin in coins:
        df = load_data(coin)
        if df.empty:
            name = coin.split('/')[0]
            print(f"  ⚠️ {name:6s}: NO DATA")
            continue
        test = df[(df['timestamp'] >= split_ts) & (df['timestamp'] < now_ts)].copy()
        result = run_backtest(test, sl=SL, tp=TP)

        oos_total_t += result['trades']
        oos_total_w += result['wins']
        oos_total_pnl += result['total_pnl']
        oos_skipped += result['skipped_min_margin']
        if result['total_pnl'] > 0:
            oos_profitable += 1

        name = coin.split('/')[0]
        status = '✅' if result['total_pnl'] > 0 else '⚠️'
        skip_info = f", skipped={result['skipped_min_margin']}" if result['skipped_min_margin'] else ""
        print(f"  {status} {name:6s}: trades={result['trades']:4d}, WR={result['win_rate']:5.1%}, "
              f"PnL={result['total_pnl']:+8.4f}, Sharpe={result['sharpe']:7.2f}{skip_info}")

    oos_wr = oos_total_w / oos_total_t if oos_total_t else 0

    # ========== SUMMARY ==========
    print("\n" + "=" * 80)
    print("📋 SUMMARY")
    print("=" * 80)
    print(f"\nIn-Sample:")
    print(f"  Total trades: {is_total_t}")
    print(f"  Win rate: {is_wr:.1%}")
    print(f"  Total PnL: {is_total_pnl:+.4f} U")
    print(f"  Profitable coins: {is_profitable}/{len(coins)}")
    print(f"  Skipped (min margin): {is_skipped}")

    print(f"\nOut-of-Sample ({oos_days} days):")
    print(f"  Total trades: {oos_total_t}")
    print(f"  Win rate: {oos_wr:.1%}")
    print(f"  Total PnL: {oos_total_pnl:+.4f} U")
    print(f"  Profitable coins: {oos_profitable}/{len(coins)}")
    print(f"  Skipped (min margin): {oos_skipped}")

    # ========== VALIDATION ==========
    print(f"\n{'=' * 80}")
    print("🎯 VALIDATION RESULTS")
    print(f"{'=' * 80}")

    checks = [
        ("Coins >= 10 (OOS)", oos_profitable >= 10),
        ("IS Win Rate > 55%", is_wr > 0.55),
        ("IS PnL > 0", is_total_pnl > 0),
        ("OOS Win Rate > 55%", oos_wr > 0.55),
        ("OOS PnL > 0", oos_total_pnl > 0),
        ("OOS Trades > 20", oos_total_t > 20),
    ]

    all_pass = True
    for check, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_pass = False
        print(f"  {status} | {check}")

    print(f"\n{'=' * 80}")
    if all_pass:
        print("🎉 ALL CHECKS PASSED - STRATEGY APPROVED FOR DEPLOYMENT")
    else:
        print("⚠️ SOME CHECKS FAILED - FURTHER OPTIMIZATION NEEDED")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
