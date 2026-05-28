"""OKX 量化自动交易系统 - 主入口"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

import yaml
import pandas as pd
from loguru import logger

from data.fetcher import DataFetcher
from data.storage import DataStorage
from data.hot_coins import HotCoinSelector
from strategy.hybrid import HybridStrategy
from backtest.engine import BacktestEngine
from backtest.analyzer import BacktestAnalyzer
from backtest.report import BacktestReport


def load_config() -> dict:
    config_dir = Path("config")

    with open(config_dir / "settings.yaml", "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)

    with open(config_dir / "strategy.yaml", "r", encoding="utf-8") as f:
        strategy = yaml.safe_load(f)

    merged = {**settings, **strategy}
    return merged


def setup_logging(config: dict):
    log_cfg = config.get("logging", {})
    logger.add(
        log_cfg.get("file", "logs/trading.log"),
        level=log_cfg.get("level", "INFO"),
        rotation=log_cfg.get("rotation", "10 MB"),
        retention=log_cfg.get("retention", "30 days"),
        encoding="utf-8",
    )


def cmd_backtest(args, config: dict):
    logger.info("=" * 60)
    logger.info("开始回测")
    logger.info("=" * 60)

    storage = DataStorage()
    fetcher = DataFetcher(storage)
    selector = HotCoinSelector(fetcher)

    days = args.days if args.days else 90
    symbols = []

    if args.symbols:
        symbols = args.symbols.split(",")
    else:
        symbols = selector.get_top_coins(top_n=5)

    logger.info(f"回测币种: {symbols}")
    logger.info(f"回测周期: {days}天")

    all_data = {}
    train_data = {}

    csv_files = {
        "BTC/USDT:USDT": "data/BTC_15m.csv",
        "ETH/USDT:USDT": "data/ETH_15m.csv",
    }

    for symbol in symbols:
        if symbol in csv_files and Path(csv_files[symbol]).exists():
            df = pd.read_csv(csv_files[symbol])
            logger.info(f"{symbol}: 从CSV加载 {len(df)} 条K线")
        else:
            df = fetcher.sync_klines(symbol, config["general"]["timeframe"], days=days)

        if df.empty or len(df) < 100:
            logger.warning(f"{symbol} 数据不足，跳过")
            continue

        split = int(len(df) * 0.7)
        train_data[symbol] = df.iloc[:split]
        all_data[symbol] = df.iloc[split:]
        logger.info(f"{symbol}: 训练集={split}条, 测试集={len(df)-split}条")

    if not all_data:
        logger.error("没有可用数据")
        return

    engine = BacktestEngine(config)
    stats = engine.run_multi(all_data, train_data)

    analyzer = BacktestAnalyzer(config["general"]["initial_capital"])
    if "trades" in stats and stats["trades"]:
        from execution.account import TradeRecord
        trades = []
        for t in stats.get("trades", []):
            if t.get("action") == "close":
                trades.append(TradeRecord(
                    symbol=t["symbol"],
                    direction=t["direction"],
                    entry_price=t["entry_price"],
                    exit_price=t["exit_price"],
                    size=0,
                    leverage=0,
                    pnl=t["pnl"],
                    pnl_pct=t["pnl_pct"],
                    open_time=datetime.now(),
                    close_time=datetime.now(),
                    close_reason=t["reason"],
                ))
        result = analyzer.analyze(trades)
        analyzer.print_summary(result)
        report = BacktestReport()
        report.generate_chart(result, "BACKTEST")
        report.generate_json(result, "BACKTEST")
        report.print_target_check(result)
    else:
        logger.warning("没有产生交易")


def cmd_train(args, config: dict):
    logger.info("训练ML模型...")

    storage = DataStorage()
    fetcher = DataFetcher(storage)
    selector = HotCoinSelector(fetcher)

    symbols = selector.get_top_coins(top_n=10)
    data = {}

    for symbol in symbols:
        df = fetcher.sync_klines(symbol, config["general"]["timeframe"], days=90)
        if not df.empty and len(df) >= 100:
            data[symbol] = df

    if not data:
        logger.error("没有足够的训练数据")
        return

    strategy = HybridStrategy(config)
    metrics = strategy.train_ml(data, force=True)

    if metrics:
        logger.info(f"训练完成: {metrics}")
    else:
        logger.warning("训练失败")


def cmd_live(args, config: dict):
    import time
    import signal as sig
    from execution.order_executor import OrderExecutor
    from strategy.hybrid import HybridStrategy
    from strategy.indicators import TechnicalIndicators
    from risk.stop_loss import StopLossManager
    from risk.take_profit import TakeProfitManager
    from risk.position_sizer import PositionSizer

    logger.info("=" * 60)
    logger.info("启动 OKX 模拟盘自动交易")
    logger.info("=" * 60)

    executor = OrderExecutor(config)

    if not executor.test_connection():
        logger.error("无法连接OKX，请检查API配置")
        return

    strategy = HybridStrategy(config)
    indicators = TechnicalIndicators(config)
    stop_loss_mgr = StopLossManager(config)
    take_profit_mgr = TakeProfitManager(config)
    position_sizer = PositionSizer(config)

    storage = DataStorage()
    fetcher = DataFetcher(storage)
    selector = HotCoinSelector(fetcher)

    # 加载历史交易，显示累计盈亏
    initial_capital = config.get("trading_capital", 10)
    cumulative_pnl = 0.0
    try:
        history = storage.load_trades()
        if history:
            cumulative_pnl = sum(t.pnl for t in history)
            wins = len([t for t in history if t.pnl > 0])
            losses = len([t for t in history if t.pnl <= 0])
            logger.info(f"历史交易: {len(history)}笔 | 盈{wins}/亏{losses} | 累计盈亏={cumulative_pnl:+.4f}U")
        else:
            logger.info("无历史交易记录")
    except Exception as e:
        logger.warning(f"加载历史交易失败: {e}")

    # 动态策略资金 = 初始资金 + 累计盈亏
    current_capital = [max(initial_capital + cumulative_pnl, 1.0)]  # 用列表做可变引用
    logger.info(f"策略资金: {initial_capital}U + {cumulative_pnl:+.4f}U = {current_capital[0]:.2f}U")

    symbols = config.get("live_symbols", ["BTC/USDT:USDT", "ETH/USDT:USDT"])
    timeframe = config.get("general", {}).get("timeframe", "15m")
    scan_interval = config.get("general", {}).get("scan_interval", 60)

    logger.info(f"监控币种: {symbols}")
    logger.info(f"时间框架: {timeframe}")
    logger.info(f"扫描间隔: {scan_interval}秒")

    running = True
    def _stop(signum, frame):
        nonlocal running
        logger.info("收到停止信号，准备退出...")
        running = False
    sig.signal(sig.SIGINT, _stop)
    sig.signal(sig.SIGTERM, _stop)

    while running:
        try:
            _live_tick(
                executor=executor,
                strategy=strategy,
                indicators=indicators,
                stop_loss_mgr=stop_loss_mgr,
                take_profit_mgr=take_profit_mgr,
                position_sizer=position_sizer,
                fetcher=fetcher,
                symbols=symbols,
                timeframe=timeframe,
                config=config,
                storage=storage,
                current_capital=current_capital,
            )
        except Exception as e:
            logger.error(f"交易循环异常: {e}")

        for _ in range(scan_interval):
            if not running:
                break
            time.sleep(1)

    logger.info("交易已停止")


def _calc_used_margin(positions: list) -> float:
    """计算现有持仓占用的保证金总额
    OKX: contracts = 合约张数, 1张 = contractSize(0.01 BTC / 0.1 ETH)
    名义价值 = 张数 × contractSize × 价格
    保证金 = 名义价值 / 杠杆
    """
    total = 0.0
    for p in positions:
        try:
            contracts = abs(float(p.get("contracts", 0)))
            entry = float(p.get("entryPrice", 0))
            leverage = float(p.get("leverage", 1))
            contract_size = float(p.get("contractSize", 0.01))
            if contracts > 0 and entry > 0 and leverage > 0:
                notional = contracts * contract_size * entry
                total += notional / leverage
        except Exception as e:
            logger.warning(f"计算保证金异常({p.get('symbol','?')}): {e}")
    return total


def _live_tick(
    executor, strategy, indicators, stop_loss_mgr, take_profit_mgr,
    position_sizer, fetcher, symbols, timeframe, config, storage=None,
    current_capital=None,
):
    real_balance = executor.get_balance()
    positions = executor.get_positions()

    # HIGH #6: 获取持仓失败时，positions为空但实际可能有持仓
    # 此时标记api_ok=False，禁止开新仓
    try:
        # 用一次额外的API调用验证连接正常
        executor.exchange.fetch_balance(params={"type": "swap"})
        api_ok = True
    except Exception:
        api_ok = False
        logger.warning("API验证失败，本轮不开新仓")

    position_symbols = {p["symbol"] for p in positions}

    # --- 资金隔离 ---
    if current_capital is not None:
        trading_capital = current_capital[0]
    else:
        trading_capital = config.get("trading_capital", real_balance)
    used_margin = _calc_used_margin(positions)
    available = max(trading_capital - used_margin, 0)

    logger.info(
        f"[Tick] 模拟盘={real_balance:.2f}U "
        f"策略资金={trading_capital:.2f}U "
        f"已用保证金={used_margin:.2f}U "
        f"可用={available:.2f}U "
        f"持仓={len(positions)}个"
    )

    for symbol in symbols:
        try:
            df = fetcher.sync_klines(symbol, timeframe, days=7)
            if df.empty or len(df) < 60:
                continue

            df = indicators.calculate(df)

            if symbol in position_symbols:
                _manage_position(executor, symbol, positions, df, config, storage=storage)
            elif api_ok:
                # HIGH #4: 每次开仓后扣减available，避免超支
                opened = _try_open(executor, strategy, position_sizer, stop_loss_mgr,
                          take_profit_mgr, symbol, df, available, config, storage=storage)
                if opened:
                    available = max(available - opened, 0)

        except Exception as e:
            logger.error(f"{symbol} 处理异常: {e}")


def _manage_position(executor, symbol, positions, df, config, storage=None):
    pos = next((p for p in positions if p["symbol"] == symbol), None)
    if not pos:
        return

    # 用实时价格判断止损止盈，而不是K线收盘价
    try:
        ticker = executor.exchange.fetch_ticker(symbol)
        current_price = float(ticker["last"])
    except Exception:
        current_price = float(df["close"].iloc[-1])

    entry_price = float(pos["entryPrice"])
    side = pos["side"]
    contracts = abs(float(pos.get("contracts", 0)))

    fixed_sl_pct = config.get("stop_loss", {}).get("fixed_pct", 0.008)
    fixed_tp_pct = config.get("take_profit", {}).get("fixed_pct", 0.005)

    if side == "long":
        pnl_pct = (current_price - entry_price) / entry_price
        sl_price = entry_price * (1 - fixed_sl_pct)
        tp_price = entry_price * (1 + fixed_tp_pct)
    else:
        pnl_pct = (entry_price - current_price) / entry_price
        sl_price = entry_price * (1 + fixed_sl_pct)
        tp_price = entry_price * (1 - fixed_tp_pct)

    if pnl_pct <= -fixed_sl_pct:
        logger.info(f"[止损] {symbol} {side} 盈亏={pnl_pct:.2%}")
        executor.close_position(symbol, side)
        _record_trade(storage, symbol, side, pos, current_price, "止损")
    elif pnl_pct >= fixed_tp_pct:
        logger.info(f"[止盈] {symbol} {side} 盈亏={pnl_pct:.2%}")
        executor.close_position(symbol, side)
        _record_trade(storage, symbol, side, pos, current_price, "止盈")
    else:
        logger.debug(f"[持仓] {symbol} {side} 盈亏={pnl_pct:.2%} 入场={entry_price} 当前={current_price}")


def _try_open(executor, strategy, position_sizer, stop_loss_mgr,
              take_profit_mgr, symbol, df, available, config, storage=None):
    """尝试开仓，返回实际使用的保证金金额（用于available扣减），失败返回0"""
    fixed_sl_pct = config.get("stop_loss", {}).get("fixed_pct", 0.008)
    fixed_tp_pct = config.get("take_profit", {}).get("fixed_pct", 0.005)
    leverage = config.get("leverage", {}).get("min", 5)

    # CRITICAL #3: 用PositionSizer做风控检查
    can_open, reason = position_sizer.can_open_position(
        balance=available,
        daily_pnl=0,  # TODO: 接入日内盈亏
        current_positions=0,  # 由调用方传入更准确，但先用0
    )
    if not can_open:
        logger.info(f"[风控拒绝] {symbol}: {reason}")
        return 0

    signal = strategy.analyze(symbol, df)
    if signal is None:
        return 0
    if signal.direction != "long":
        return 0

    entry_price = float(df["close"].iloc[-1])

    # CRITICAL #3: 用PositionSizer计算仓位，而不是固定50%
    entry_price_for_sizer = entry_price
    pos_size = position_sizer.calculate_position(
        balance=available,
        entry_price=entry_price_for_sizer,
        stop_distance_pct=fixed_sl_pct,
        signal_strength=signal.strength if hasattr(signal, 'strength') else 0.5,
        current_positions=0,
    )
    amount_usdt = pos_size.amount_usdt if pos_size.amount_usdt > 0 else available * 0.50

    if amount_usdt < 1:
        logger.warning(f"可用资金不足({available:.2f}U)，跳过 {symbol}")
        return 0

    order = executor.open_long(symbol, amount_usdt, leverage, entry_price)
    if order:
        # Use actual fill price from order, fallback to current ticker
        fill_price = order.get("average") or order.get("price") or entry_price
        if not fill_price or fill_price <= 0:
            try:
                ticker = executor.exchange.fetch_ticker(symbol)
                fill_price = ticker["last"]
            except Exception:
                fill_price = entry_price

        # 获取实际持仓张数，传给止损/止盈
        try:
            pos = executor.exchange.fetch_position(symbol)
            actual_contracts = abs(float(pos.get("contracts", 0))) if pos else 0
        except Exception:
            actual_contracts = 0

        sl_price = fill_price * (1 - fixed_sl_pct)
        tp_price = fill_price * (1 + fixed_tp_pct)
        # CRITICAL #1: 传入实际持仓张数
        executor.set_stop_loss(symbol, "long", sl_price, contracts=actual_contracts)
        executor.set_take_profit(symbol, "long", tp_price, contracts=actual_contracts)
        logger.info(f"[开仓] {symbol} LONG 入场={fill_price} SL={sl_price} TP={tp_price} 金额={amount_usdt:.2f}U 张数={actual_contracts}")
        return amount_usdt  # 返回实际使用的保证金，供available扣减

    return 0



def _record_trade(storage, symbol, side, pos, exit_price, reason):
    """平仓后记录交易到数据库"""
    if storage is None:
        return
    try:
        from execution.account import TradeRecord
        from datetime import datetime
        entry_price = float(pos["entryPrice"])
        leverage = int(pos.get("leverage", 1))
        contracts = abs(float(pos.get("contracts", 0)))
        # CRITICAL #2: 加入contractSize，修复盈亏膨胀
        contract_size = float(pos.get("contractSize", 0.01))
        # 名义价值 = 张数 × contractSize × 价格
        notional = contracts * contract_size * entry_price
        # 保证金 = 名义价值 / 杠杆
        size = notional / leverage if leverage > 0 else notional
        if side == "long":
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price
        pnl_pct *= leverage
        pnl = size * pnl_pct
        # 手续费按名义价值收，不是保证金
        fee = notional * 0.0005
        pnl -= fee
        record = TradeRecord(
            symbol=symbol, direction=side,
            entry_price=entry_price, exit_price=exit_price,
            size=round(size, 4), leverage=leverage,
            pnl=round(pnl, 6), pnl_pct=round(pnl_pct, 6),
            open_time=datetime.utcnow(), close_time=datetime.utcnow(),
            close_reason=reason,
        )
        storage.save_trade(record)
        emoji = "+" if pnl >= 0 else ""
        logger.info(f"[记录] {symbol} {side} 盈亏={emoji}{pnl:.4f}U ({emoji}{pnl_pct:.2%}) | {reason}")
    except Exception as e:
        logger.warning(f"记录交易失败: {e}")


def cmd_paper(args, config: dict):
    logger.info("启动纸上交易模式...")
    logger.info("使用实时行情进行模拟交易")

    storage = DataStorage()
    fetcher = DataFetcher(storage)
    selector = HotCoinSelector(fetcher)

    from execution.paper_trader import PaperTrader
    trader = PaperTrader(config)

    symbols = selector.get_top_coins(top_n=10)
    logger.info(f"监控币种: {symbols}")

    for symbol in symbols:
        df = fetcher.sync_klines(symbol, config["general"]["timeframe"], days=7)
        if not df.empty:
            trader.on_candle(symbol, df)

    stats = trader.get_stats()
    logger.info(f"纸上交易结果: {stats}")


def main():
    parser = argparse.ArgumentParser(description="OKX 量化自动交易系统")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    bt_parser = subparsers.add_parser("backtest", help="运行回测")
    bt_parser.add_argument("--days", type=int, default=90, help="回测天数")
    bt_parser.add_argument("--symbols", type=str, help="币种列表(逗号分隔)")

    subparsers.add_parser("train", help="训练ML模型")

    subparsers.add_parser("live", help="启动实盘")
    subparsers.add_parser("paper", help="启动纸上交易")

    args = parser.parse_args()

    config = load_config()
    setup_logging(config)

    if args.command == "backtest":
        cmd_backtest(args, config)
    elif args.command == "train":
        cmd_train(args, config)
    elif args.command == "live":
        cmd_live(args, config)
    elif args.command == "paper":
        cmd_paper(args, config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
