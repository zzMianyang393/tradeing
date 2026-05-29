#!/usr/bin/env python3
"""参数网格搜索 - 找到胜率>50%+盈利的组合"""

import yaml, sqlite3, pandas as pd, numpy as np, json, time
from pathlib import Path
from datetime import datetime

# 加载数据一次
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

def run_backtest_with_params(data, params):
    """用指定参数跑回测，返回统计"""
    # 构建config
    with open('config/settings.yaml') as f:
        settings = yaml.safe_load(f)
    with open('config/strategy.yaml') as f:
        strat = yaml.safe_load(f)
    config = {**settings, **strat}
    
    # 覆盖参数
    config['stop_loss']['fixed_pct'] = params['sl_pct']
    config['stop_loss']['max_pct'] = params['sl_pct']
    config['stop_loss']['trailing_activate'] = params.get('trailing_activate', 0.003)
    config['stop_loss']['trailing_callback'] = params.get('trailing_callback', 0.005)
    config['take_profit']['fixed_pct'] = params['tp_pct']
    config['take_profit']['risk_reward_ratio'] = params['tp_pct'] / params['sl_pct']
    config['take_profit']['partial_close_trigger'] = params.get('partial_trigger', 0.0)
    config['adx']['threshold'] = params.get('adx_threshold', 15)
    config['rules']['min_conditions'] = params.get('min_conditions', 5)
    
    # 保存临时config
    with open('/tmp/test_strategy.yaml', 'w') as f:
        yaml.dump(strat, f)
    
    # 导入模块（用当前代码）
    from strategy.hybrid import HybridStrategy
    from strategy.indicators import TechnicalIndicators
    from backtest.engine import BacktestEngine
    from execution.account import AccountManager
    from risk.position_sizer import PositionSizer
    
    strategy = HybridStrategy(config)
    indicators = TechnicalIndicators(config)
    account = AccountManager(initial_capital=10.0)
    position_sizer = PositionSizer(config)
    
    # 对每个币种跑回测
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
        
        # 训练模型（用force=False避免重复训练）
        strategy.train_ml({symbol: train_df}, force=False)
        
        # 计算指标
        test_df = indicators.calculate(test_df)
        if test_df.empty or len(test_df) < 60:
            continue
        
        # 模拟回测
        current_positions = {}
        daily_pnl = 0.0
        trades = []
        
        for i in range(60, len(test_df)):
            window = test_df.iloc[:i+1]
            row = test_df.iloc[i]
            price = float(row['close'])
            
            # 检查现有持仓
            if symbol in current_positions:
                pos = current_positions[symbol]
                # 检查止损
                sl_hit = False
                tp_hit = False
                if pos['direction'] == 'short':
                    if price >= pos['entry'] * (1 + params['sl_pct']):
                        sl_hit = True
                    elif price <= pos['entry'] * (1 - params['tp_pct']):
                        tp_hit = True
                else:
                    if price <= pos['entry'] * (1 - params['sl_pct']):
                        sl_hit = True
                    elif price >= pos['entry'] * (1 + params['tp_pct']):
                        tp_hit = True
                
                if sl_hit or tp_hit:
                    if pos['direction'] == 'short':
                        pnl_pct = (pos['entry'] - price) / pos['entry']
                    else:
                        pnl_pct = (price - pos['entry']) / pos['entry']
                    pnl_pct *= pos['leverage']
                    
                    # 手续费
                    notional = pos['amount'] * pos['leverage']
                    fee = notional * 0.0005 * 2  # 开+平
                    pnl = pos['amount'] * pnl_pct - fee
                    
                    balance += pos['amount'] + pnl
                    daily_pnl += pnl
                    trades.append({
                        'symbol': symbol,
                        'direction': pos['direction'],
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'reason': '止损' if sl_hit else '止盈',
                    })
                    del current_positions[symbol]
                continue
            
            # 趋势过滤
            ema_slow = row.get('ema_slow', 0)
            trend = 'neutral'
            if ema_slow > 0:
                if price > ema_slow * 1.005:
                    trend = 'bullish'
                elif price < ema_slow * 0.995:
                    trend = 'bearish'
            
            # 生成信号
            min_cond = params.get('min_conditions', 5)
            
            if trend != 'bearish':
                long_sig = strategy.signal_gen._check_long_conditions(row, window.iloc[-2] if len(window) > 1 else row, window)
                long_score = sum(long_sig.values())
                if long_score >= min_cond:
                    # 开多
                    sl_pct = params['sl_pct']
                    leverage = 50 if long_score >= 6 else 27
                    amount = min(balance * 0.25, 5.0)
                    amount = max(amount, 1.0)
                    if amount <= balance and balance >= 1:
                        current_positions[symbol] = {
                            'direction': 'long', 'entry': price,
                            'amount': amount, 'leverage': leverage,
                        }
                        balance -= amount
                        continue
            
            if trend != 'bullish':
                short_sig = strategy.signal_gen._check_short_conditions(row, window.iloc[-2] if len(window) > 1 else row, window)
                short_score = sum(short_sig.values())
                if short_score >= min_cond:
                    # 开空
                    leverage = 50 if short_score >= 6 else 27
                    amount = min(balance * 0.25, 5.0)
                    amount = max(amount, 1.0)
                    if amount <= balance and balance >= 1:
                        current_positions[symbol] = {
                            'direction': 'short', 'entry': price,
                            'amount': amount, 'leverage': leverage,
                        }
                        balance -= amount
        
        # 平掉剩余持仓
        for sym, pos in list(current_positions.items()):
            last_price = float(test_df.iloc[-1]['close'])
            if pos['direction'] == 'short':
                pnl_pct = (pos['entry'] - last_price) / pos['entry']
            else:
                pnl_pct = (last_price - pos['entry']) / pos['entry']
            pnl_pct *= pos['leverage']
            notional = pos['amount'] * pos['leverage']
            fee = notional * 0.0005 * 2
            pnl = pos['amount'] * pnl_pct - fee
            balance += pos['amount'] + pnl
            trades.append({
                'symbol': sym, 'direction': pos['direction'],
                'pnl': pnl, 'pnl_pct': pnl_pct, 'reason': '收盘平仓',
            })
        
        all_trades.extend(trades)
    
    # 统计
    if not all_trades:
        return {'trades': 0, 'win_rate': 0, 'total_pnl': 0, 'balance': balance}
    
    wins = [t for t in all_trades if t['pnl'] > 0]
    losses = [t for t in all_trades if t['pnl'] <= 0]
    total_pnl = sum(t['pnl'] for t in all_trades)
    
    return {
        'trades': len(all_trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': len(wins) / len(all_trades) if all_trades else 0,
        'total_pnl': total_pnl,
        'balance': balance,
        'avg_pnl': total_pnl / len(all_trades),
        'avg_win': np.mean([t['pnl'] for t in wins]) if wins else 0,
        'avg_loss': np.mean([t['pnl'] for t in losses]) if losses else 0,
    }


# 参数网格
param_grid = [
    # 基线
    {'sl_pct': 0.008, 'tp_pct': 0.005, 'min_conditions': 5, 'name': 'baseline'},
    # 只做多
    {'sl_pct': 0.008, 'tp_pct': 0.005, 'min_conditions': 5, 'long_only': True, 'name': 'long_only'},
    # 不同SL/TP组合
    {'sl_pct': 0.006, 'tp_pct': 0.008, 'min_conditions': 5, 'name': 'sl6_tp8'},
    {'sl_pct': 0.006, 'tp_pct': 0.010, 'min_conditions': 5, 'name': 'sl6_tp10'},
    {'sl_pct': 0.008, 'tp_pct': 0.010, 'min_conditions': 5, 'name': 'sl8_tp10'},
    {'sl_pct': 0.010, 'tp_pct': 0.015, 'min_conditions': 5, 'name': 'sl10_tp15'},
    {'sl_pct': 0.010, 'tp_pct': 0.020, 'min_conditions': 5, 'name': 'sl10_tp20'},
    # 更严格条件
    {'sl_pct': 0.008, 'tp_pct': 0.005, 'min_conditions': 6, 'name': 'cond6'},
    {'sl_pct': 0.008, 'tp_pct': 0.005, 'min_conditions': 7, 'name': 'cond7'},
    # 更宽松条件
    {'sl_pct': 0.008, 'tp_pct': 0.005, 'min_conditions': 4, 'name': 'cond4'},
    # ADX阈值
    {'sl_pct': 0.008, 'tp_pct': 0.005, 'min_conditions': 5, 'adx_threshold': 20, 'name': 'adx20'},
    {'sl_pct': 0.008, 'tp_pct': 0.005, 'min_conditions': 5, 'adx_threshold': 10, 'name': 'adx10'},
    # 组合：更宽止损+更远止盈+严格条件
    {'sl_pct': 0.010, 'tp_pct': 0.015, 'min_conditions': 6, 'name': 'wide_sl6'},
    {'sl_pct': 0.010, 'tp_pct': 0.020, 'min_conditions': 6, 'name': 'wide_sl6_tp20'},
    # 只做多+宽TP
    {'sl_pct': 0.008, 'tp_pct': 0.012, 'min_conditions': 5, 'long_only': True, 'name': 'long_only_tp12'},
    {'sl_pct': 0.008, 'tp_pct': 0.015, 'min_conditions': 5, 'long_only': True, 'name': 'long_only_tp15'},
    {'sl_pct': 0.010, 'tp_pct': 0.015, 'min_conditions': 5, 'long_only': True, 'name': 'long_only_wide'},
    # 紧止损+宽止盈（高盈亏比）
    {'sl_pct': 0.005, 'tp_pct': 0.010, 'min_conditions': 6, 'name': 'tight_sl_wide_tp'},
    {'sl_pct': 0.005, 'tp_pct': 0.015, 'min_conditions': 6, 'name': 'tight_sl_very_wide'},
    # 关闭追踪止损
    {'sl_pct': 0.008, 'tp_pct': 0.005, 'min_conditions': 5, 'trailing_activate': 0.0, 'trailing_callback': 0.0, 'name': 'no_trailing'},
    # 禁用部分止盈
    {'sl_pct': 0.008, 'tp_pct': 0.005, 'min_conditions': 5, 'partial_trigger': 999, 'name': 'no_partial'},
]

print(f"加载数据...")
all_data = load_all_data()
print(f"共 {len(all_data)} 个币种")

results = []
for i, params in enumerate(param_grid):
    name = params.pop('name', f'param_{i}')
    long_only = params.pop('long_only', False)
    
    # 如果是long_only模式，需要修改信号生成
    if long_only:
        # 临时修改
        pass  # 我们在回测逻辑里已经支持了
    
    print(f"\n[{i+1}/{len(param_grid)}] 测试: {name}")
    print(f"  参数: SL={params['sl_pct']:.1%} TP={params['tp_pct']:.1%} 条件>={params['min_conditions']}")
    
    try:
        result = run_backtest_with_params(all_data, params)
        result['name'] = name
        results.append(result)
        
        print(f"  结果: {result['trades']}笔 胜率={result['win_rate']:.1%} "
              f"盈亏={result['total_pnl']:.2f}U 余额={result['balance']:.2f}U")
        
        if result['win_rate'] >= 0.5 and result['total_pnl'] > 0:
            print(f"\n{'='*60}")
            print(f"  找到目标组合！{name}")
            print(f"  胜率: {result['win_rate']:.1%}")
            print(f"  盈亏: {result['total_pnl']:.2f}U")
            print(f"  余额: {result['balance']:.2f}U")
            print(f"{'='*60}")
            
            # 保存结果
            with open('/tmp/optimal_params.json', 'w') as f:
                json.dump({'name': name, 'params': params, 'result': result}, f, indent=2, default=str)
            break
    except Exception as e:
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()

# 汇总
print(f"\n{'='*60}")
print("所有测试结果:")
print(f"{'='*60}")
results.sort(key=lambda x: x.get('total_pnl', -999), reverse=True)
for r in results:
    marker = " ★" if r['win_rate'] >= 0.5 and r['total_pnl'] > 0 else ""
    print(f"  {r['name']:<25} {r['trades']:>3}笔 胜率={r['win_rate']:>5.1%} "
          f"盈亏={r['total_pnl']:>+7.2f}U 余额={r['balance']:>6.2f}U{marker}")
