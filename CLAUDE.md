# Tradeing - Multi-Strategy Trading System

## Project Overview
OKX quantitative trading system with multi-timeframe, multi-strategy support.
Supports 10-15 coins across uptrend, downtrend, and sideways market conditions.

## Architecture
```
tradeing/
├── config/          # Settings and strategy parameters (YAML)
├── data/            # OHLCV data (15m, 1h, 4h)
├── strategy/        # Signal generation, hybrid strategy, ML model
├── backtest/        # Backtest engine, analyzer, reports
├── risk/            # Position sizing, stop loss, take profit
├── execution/       # Order execution, paper trading, account management
├── monitor/         # Real-time monitoring
└── docs/            # Documentation
```

## Key Commands
- `python test_okx_full.py` - Run full OKX backtest
- `python run_full_report.py` - Generate comprehensive report
- `python run_90d_report.py` - 90-day backtest report
- `python param_sweep.py` - Parameter optimization sweep

## Code Standards
- Python 3.10+, type hints on all public functions
- Use loguru for logging, not print()
- Config via YAML files in config/
- All strategies must support both long and short
- Backtest must include fees (taker 0.05%), slippage (0.05%)
- Sharpe ratio calculation must be annualized

## Trading Rules
- Initial capital: 10 USDT
- Max leverage: 20x (OKX limit for small caps)
- Fee rate: 0.05% taker
- Slippage: 0.05%
- Max concurrent positions: 5
- Cooldown after close: 6 bars (1.5h on 15m)

## Strategy Modes
- `mean_reversion`: ADX < 18, RSI/BB/StochRSI reversal signals
- `trend`: ADX > 30, EMA alignment + momentum
- `auto`: Auto-switch based on ADX regime detection

## Validation Criteria (STRICT)
1. Support 10-15 coins
2. Win rate > 55%
3. Sharpe ratio > 2.0
4. Positive profit (all coins combined)
5. Pass overfitting test (train on 87 days, validate on last 7 days)
6. 7-day out-of-sample backtest (May 24-30, 2026) must also pass:
   - Win rate > 55%
   - Sharpe > 2.0
   - Positive profit

## Important Notes
- Never optimize parameters on the validation period
- Use walk-forward analysis to prevent overfitting
- Each strategy must work independently before combining
- Always run both in-sample and out-of-sample tests
