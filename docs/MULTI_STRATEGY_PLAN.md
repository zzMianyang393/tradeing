# 多时间框架多策略交易系统 - 完整实施方案

## 一、项目目标

| 目标 | 指标 | 当前状态 | 差距 |
|------|------|----------|------|
| 币种数量 | 10-15个主流币 | 5个盈利 | 需+5-10个 |
| 胜率 | >55% | 53.2%(过拟合) | 需+2% |
| 盈利 | +5-10U (90天) | +17.82U(过拟合) | 已超额 |
| 过拟合 | 7天样本外验证通过 | 未通过 | 需验证 |
| 最大回撤 | <10% | ~6% | 已达标 |

## 二、问题诊断

### 2.1 当前限制

| 问题 | 原因 | 影响 |
|------|------|------|
| 15m所有策略WR都低 | 时间框架噪音太大 | 无法达到55% |
| 均值回归WR天花板~50% | 市场随机性决定 | 结构性限制 |
| 趋势跟踪WR~35% | 假突破太多 | 亏损 |
| 动量突破WR~33% | 信号不稳定 | 亏损 |
| 过拟合严重 | 参数在测试集上优化 | 不泛化 |

### 2.2 测试结果对比

| 策略 | 达标币种 | 总盈亏 | 整体WR | 结论 |
|------|----------|--------|--------|------|
| 均值回归(无优化) | 0个 | -11.67U | 35.6% | ❌ |
| 趋势跟踪 | 0个 | -13.98U | 33% | ❌ |
| 多策略组合 | 0个 | -13.98U | 33% | ❌ |
| 投资组合 | 0个 | -11.67U | 35.6% | ❌ |
| 动量突破 | 0个 | -8.93U | 33% | ❌ |
| 均值回归(过拟合) | 5个 | +17.82U | 53.2% | ⚠️过拟合 |

### 2.3 根本限制

**15m时间框架对所有策略都不友好：**
- 均值回归：信号噪音大，WR天花板~50%
- 趋势跟踪：假突破多，WR~35%
- 动量突破：信号不稳定，WR~33%

## 三、技术架构

### 3.1 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    多时间框架多策略交易系统                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   数据层      │  │   策略层      │  │   风控层      │      │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤      │
│  │ • 15m K线    │  │ • 均值回归    │  │ • 仓位管理    │      │
│  │ • 1h K线     │  │ • 趋势跟踪    │  │ • 止损止盈    │      │
│  │ • 4h K线     │  │ • 自适应切换  │  │ • 部分止盈    │      │
│  │ • 资金费率   │  │ • 信号过滤    │  │ • 冷却期      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │               │                  │                │
│         └───────────────┼──────────────────┘                │
│                         ▼                                   │
│              ┌──────────────────┐                           │
│              │    回测验证层     │                           │
│              ├──────────────────┤                           │
│              │ • 90天回测       │                           │
│              │ • 7天样本外验证   │                           │
│              │ • 参数敏感性测试  │                           │
│              └──────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 数据流

```
15m K线 → 技术指标计算 → 策略信号生成
                            ↓
1h K线  → 趋势确认    → 信号过滤
                            ↓
4h K线  → 大趋势过滤  → 最终信号
                            ↓
                    风控系统 → 开仓/平仓
```

## 四、策略详解

### 4.1 均值回归策略 (震荡市)

**适用条件:** ADX < 20 (震荡市场)

**入场条件 (需满足6/7):**

| # | 条件 | 说明 | 阈值 |
|---|------|------|------|
| 1 | RSI超卖 | RSI深度超卖 | < 30 |
| 2 | BB下轨 | 价格跌破布林带下轨 | price < bb_lower |
| 3 | StochRSI | StochRSI超卖 | stoch_k < 20 |
| 4 | 下影线 | K线有下影线支撑 | lower_wick > atr*0.3 |
| 5 | 放量 | 成交量高于均量 | vol_ratio > 1.3 |
| 6 | 实体 | K线实体足够大 | body > atr*0.3 |
| 7 | DI确认 | 多头DI确认 | di_plus > di_minus |

**Bonus条件 (增加信号强度):**

| # | 条件 | 说明 |
|---|------|------|
| 1 | MACD转向 | MACD柱状图从负转正 |
| 2 | RSI背离 | 价格新低但RSI未新低 |

**出场条件:**
- 止损: 固定2.5% 或 ATR×6
- 止盈: 固定3.75% 或 ATR×9
- 部分止盈: 盈利3%时平50%
- 保本止损: 盈利后止损移至入场价

**做空条件 (需满足6/7):**

| # | 条件 | 说明 | 阈值 |
|---|------|------|------|
| 1 | RSI超买 | RSI深度超买 | > 70 |
| 2 | BB上轨 | 价格突破布林带上轨 | price > bb_upper |
| 3 | StochRSI | StochRSI超买 | stoch_k > 80 |
| 4 | 上影线 | K线有上影线压力 | upper_wick > atr*0.3 |
| 5 | 放量 | 成交量高于均量 | vol_ratio > 1.3 |
| 6 | 实体 | K线实体足够大 | body > atr*0.3 |
| 7 | DI确认 | 空头DI确认 | di_minus > di_plus |

### 4.2 趋势跟踪策略 (趋势市)

**适用条件:** ADX > 30 (强趋势市场)

**做多入场条件 (需满足6/7):**

| # | 条件 | 说明 |
|---|------|------|
| 1 | EMA排列 | fast > medium > slow |
| 2 | 价格在EMA上 | price > ema_slow * 1.002 |
| 3 | RSI动量 | rsi > 50 且上升 |
| 4 | 成交量 | vol_ratio > 1.1 |
| 5 | DI确认 | di_plus > di_minus |
| 6 | MACD动量 | macd_hist > 0 或转正 |
| 7 | 前K线收阳 | prev_close > prev_open |

**做空入场条件 (需满足6/7):**

| # | 条件 | 说明 |
|---|------|------|
| 1 | EMA排列 | fast < medium < slow |
| 2 | 价格在EMA下 | price < ema_slow * 0.998 |
| 3 | RSI动量 | rsi < 50 且下降 |
| 4 | 成交量 | vol_ratio > 1.1 |
| 5 | DI确认 | di_minus > di_plus |
| 6 | MACD动量 | macd_hist < 0 或转负 |
| 7 | 前K线收阴 | prev_close < prev_open |

### 4.3 自适应切换逻辑

```python
def detect_market_regime(df):
    """检测市场状态"""
    adx = df['adx'].iloc[-1]
    price = df['close'].iloc[-1]
    ema_slow = df['ema_slow'].iloc[-1]
    
    # 趋势方向
    if price > ema_slow * 1.03:
        trend = 'bullish'
    elif price < ema_slow * 0.97:
        trend = 'bearish'
    else:
        trend = 'neutral'
    
    # 策略选择
    if adx < 20:
        strategy = 'mean_reversion'
        confidence = 0.7
    elif adx > 30 and trend != 'neutral':
        strategy = 'trend_following'
        confidence = min(0.5 + adx/40, 0.9)
    else:
        strategy = 'both'  # 两种都试
        confidence = 0.5
    
    return {
        'strategy': strategy,
        'confidence': confidence,
        'trend': trend,
        'adx': adx,
    }
```

## 五、风控系统

### 5.1 仓位管理

```python
def calculate_position_size(balance, signal_strength, max_risk=0.02):
    """
    根据信号强度计算仓位
    
    参数:
        balance: 当前余额
        signal_strength: 信号强度 (0-1)
        max_risk: 最大风险比例 (默认2%)
    
    返回:
        position_size: 仓位大小 (USDT)
    """
    base_size = balance * max_risk
    adjusted_size = base_size * signal_strength
    return min(adjusted_size, balance * 0.10)  # 最大10%仓位
```

### 5.2 止损止盈

```python
def calculate_sl_tp(entry_price, direction, atr):
    """
    根据ATR计算止损止盈
    
    参数:
        entry_price: 入场价格
        direction: 'long' 或 'short'
        atr: 平均真实波幅
    
    返回:
        sl: 止损价格
        tp: 止盈价格
    """
    # ATR倍数
    sl_multiplier = 6
    tp_multiplier = 9
    
    # 计算百分比
    sl_pct = min(atr * sl_multiplier / entry_price, 0.05)  # 最大5%
    tp_pct = min(atr * tp_multiplier / entry_price, 0.08)  # 最大8%
    
    if direction == 'long':
        sl = entry_price * (1 - sl_pct)
        tp = entry_price * (1 + tp_pct)
    else:
        sl = entry_price * (1 + sl_pct)
        tp = entry_price * (1 - tp_pct)
    
    return sl, tp
```

### 5.3 部分止盈

```python
def check_partial_close(position, current_price):
    """
    检查部分止盈
    
    规则:
        - 盈利3%时平50%仓位
        - 剩余仓位设置保本止损
    """
    if position['direction'] == 'long':
        pnl_pct = (current_price - position['entry']) / position['entry']
    else:
        pnl_pct = (position['entry'] - current_price) / position['entry']
    
    if pnl_pct >= 0.03 and not position['partial_closed']:
        return 'partial_close'
    return None
```

### 5.4 风险参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 单笔最大风险 | 2% | 每笔交易最大亏损 |
| 最大仓位 | 10% | 单币种最大仓位 |
| 最大回撤 | 10% | 触发全部平仓 |
| 冷却期 | 2小时 | 平仓后等待时间 |
| 最大持仓数 | 5 | 同时持仓上限 |

## 六、回测验证

### 6.1 90天回测

```python
def backtest_90d(config, coins):
    """90天回测"""
    results = {}
    for coin in coins:
        data = load_data(coin)
        result = run_backtest(config, data, period='90d')
        results[coin] = result
    
    # 计算整体指标
    total_pnl = sum(r['pnl'] for r in results.values())
    avg_wr = np.mean([r['wr'] for r in results.values()])
    profitable = [r for r in results if r['pnl'] > 0]
    
    return {
        'total_pnl': total_pnl,
        'avg_wr': avg_wr,
        'profitable_coins': len(profitable),
        'results': results,
    }
```

### 6.2 7天样本外验证

```python
def validate_7d(config, coins):
    """7天样本外验证"""
    results = {}
    for coin in coins:
        data = load_data(coin)
        result = run_backtest(config, data, period='7d')
        results[coin] = result
    
    total_pnl = sum(r['pnl'] for r in results.values())
    avg_wr = np.mean([r['wr'] for r in results.values()])
    
    # 验证标准
    passed = total_pnl > 0 and avg_wr > 0.55
    
    return {
        'total_pnl': total_pnl,
        'avg_wr': avg_wr,
        'passed': passed,
    }
```

### 6.3 参数敏感性测试

```python
def sensitivity_test(config, coins):
    """测试参数敏感性"""
    results = []
    
    # 测试不同SL/TP组合
    for sl in [0.02, 0.025, 0.03]:
        for tp in [0.03, 0.0375, 0.045]:
            for conditions in [5, 6, 7]:
                test_config = config.copy()
                test_config['stop_loss']['fixed_pct'] = sl
                test_config['take_profit']['fixed_pct'] = tp
                test_config['rules']['min_conditions'] = conditions
                
                result = backtest_90d(test_config, coins)
                results.append({
                    'sl': sl,
                    'tp': tp,
                    'conditions': conditions,
                    'pnl': result['total_pnl'],
                    'wr': result['avg_wr'],
                })
    
    # 选择最优参数
    best = max(results, key=lambda x: x['pnl'] if x['wr'] > 0.55 else -999)
    return best
```

## 七、实施步骤

### 阶段1: 数据准备 (0.5天)

- [ ] 拉取16个币种的15m数据
- [ ] 生成1h和4h数据
- [ ] 拉取资金费率数据
- [ ] 计算技术指标

### 阶段2: 策略开发 (2天)

- [ ] 完善均值回归策略
- [ ] 完善趋势跟踪策略
- [ ] 实现市场状态检测
- [ ] 实现策略自动切换
- [ ] 添加RSI背离检测
- [ ] 添加MACD转向确认

### 阶段3: 风控系统 (1天)

- [ ] 实现仓位管理
- [ ] 实现止损止盈
- [ ] 实现部分止盈
- [ ] 实现冷却期
- [ ] 实现最大回撤保护

### 阶段4: 回测验证 (1天)

- [ ] 90天回测
- [ ] 7天样本外验证
- [ ] 参数敏感性测试
- [ ] 过拟合检测

### 阶段5: 优化调整 (1天)

- [ ] 根据回测结果调整参数
- [ ] 优化策略组合
- [ ] 最终验证

**总计: 5.5天**

## 八、成功标准

| 指标 | 目标 | 验证方式 | 权重 |
|------|------|----------|------|
| 币种数量 | 10-15个 | 回测统计 | 20% |
| 胜率 | >55% | 回测统计 | 30% |
| 盈利 | +5-10U | 回测统计 | 25% |
| 过拟合 | 7天验证通过 | 样本外测试 | 15% |
| 最大回撤 | <10% | 回测统计 | 10% |

## 九、风险控制

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|----------|
| 15m噪音 | 高 | 中 | 多时间框架确认 |
| 市场状态误判 | 中 | 高 | 保守切换阈值 |
| 参数过拟合 | 中 | 高 | 7天样本外验证 |
| 极端行情 | 低 | 高 | 最大回撤10%止损 |
| 流动性风险 | 低 | 中 | 只交易主流币 |
| 系统故障 | 低 | 高 | 定期备份+监控 |

## 十、预期结果

| 场景 | 概率 | 结果 |
|------|------|------|
| 最佳 | 30% | 12个币, 60% WR, +10U |
| 基准 | 50% | 8个币, 55% WR, +5U |
| 最差 | 20% | 5个币, 50% WR, +2U |

## 十一、文件结构

```
Quantify/
├── config/
│   ├── settings.yaml          # 全局设置
│   └── strategy.yaml          # 策略参数
├── data/
│   ├── BTC_15m.csv            # BTC 15分钟数据
│   ├── BTC_1h.csv             # BTC 1小时数据
│   ├── BTC_4h.csv             # BTC 4小时数据
│   └── ...                    # 其他币种数据
├── strategy/
│   ├── signals.py             # 信号生成器
│   ├── hybrid.py              # 混合策略
│   ├── indicators.py          # 技术指标
│   └── ml_model.py            # ML模型
├── backtest/
│   ├── engine.py              # 回测引擎
│   └── analyzer.py            # 结果分析
├── risk/
│   ├── position_sizer.py      # 仓位管理
│   ├── stop_loss.py           # 止损管理
│   └── take_profit.py         # 止盈管理
├── docs/
│   └── MULTI_STRATEGY_PLAN.md # 本文档
└── main.py                    # 主入口
```

## 十二、关键配置

```yaml
# config/strategy.yaml
rules:
  strategy_mode: "auto"        # 自动切换策略
  min_conditions: 6            # 需满足6/7条件
  long_conditions_min: 6       # 做多条件数

stop_loss:
  fixed_pct: 0.025             # 固定止损2.5%
  atr_multiplier: 6            # 或ATR×6
  trailing_activate: 0         # 禁用追踪止损
  trailing_callback: 0

take_profit:
  fixed_pct: 0.0375            # 固定止盈3.75%
  atr_multiplier: 9            # 或ATR×9
  partial_close_trigger: 0.03  # 3%部分止盈
  partial_close_pct: 0.5       # 平50%仓位

risk:
  max_position_pct: 0.02       # 单笔最大风险2%
  max_concurrent_positions: 5  # 最大持仓5个
  max_daily_loss: 0.10         # 日最大亏损10%

adx:
  period: 14
  threshold: 20                # ADX阈值
```

## 十三、注意事项

1. **避免过拟合**: 使用7天样本外验证，不针对特定时间段优化参数
2. **保守的仓位管理**: 单笔风险不超过2%，最大回撤10%
3. **多时间框架确认**: 15m信号需1h趋势确认
4. **市场状态适应**: 根据ADX自动切换策略
5. **实时监控**: 监控持仓和风险指标

## 十四、更新日志

| 日期 | 更新内容 |
|------|----------|
| 2026-05-30 | 初始版本 |

---

**文档维护者**: Quantify Trading System
**最后更新**: 2026-05-30
