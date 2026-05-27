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
            )
        except Exception as e:
            logger.error(f"交易循环异常: {e}")

        for _ in range(scan_interval):
            if not running:
                break
            time.sleep(1)

    logger.info("交易已停止")


def _live_tick(
    executor, strategy, indicators, stop_loss_mgr, take_profit_mgr,
    position_sizer, fetcher, symbols, timeframe, config,
):
    balance = executor.get_balance()
    positions = executor.get_positions()
    position_symbols = {p["symbol"] for p in positions}

    logger.info(f"[Tick] 余额={balance:.2f}U 持仓={len(positions)}个")

    for symbol in symbols:
        try:
            df = fetcher.sync_klines(symbol, timeframe, days=7)
            if df.empty or len(df) < 60:
                continue

            df = indicators.calculate(df)

            if symbol in position_symbols:
                _manage_position(executor, symbol, positions, df, config)
            else:
                _try_open(executor, strategy, position_sizer, stop_loss_mgr,
                          take_profit_mgr, symbol, df, balance, config)

        except Exception as e:
            logger.error(f"{symbol} 处理异常: {e}")


def _manage_position(executor, symbol, positions, df, config):
    pos = next((p for p in positions if p["symbol"] == symbol), None)
    if not pos:
        return

    current_price = float(df["close"].iloc[-1])
    entry_price = float(pos["entryPrice"])
    side = pos["side"]

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
    elif pnl_pct >= fixed_tp_pct:
        logger.info(f"[止盈] {symbol} {side} 盈亏={pnl_pct:.2%}")
        executor.close_position(symbol, side)
    else:
        logger.debug(f"[持仓] {symbol} {side} 盈亏={pnl_pct:.2%} 入场={entry_price} 当前={current_price}")


def _try_open(executor, strategy, position_sizer, stop_loss_mgr,
              take_profit_mgr, symbol, df, balance, config):
    fixed_sl_pct = config.get("stop_loss", {}).get("fixed_pct", 0.008)
    fixed_tp_pct = config.get("take_profit", {}).get("fixed_pct", 0.005)
    leverage = config.get("leverage", {}).get("min", 5)

    signal = strategy.analyze(symbol, df)
    if signal is None:
        return
    if signal.direction != "long":
        return

    entry_price = float(df["close"].iloc[-1])
    amount_usdt = balance * 0.10
    if amount_usdt < 1:
        logger.warning(f"余额不足，跳过 {symbol}")
        return

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

        sl_price = fill_price * (1 - fixed_sl_pct)
        tp_price = fill_price * (1 + fixed_tp_pct)
        executor.set_stop_loss(symbol, "long", sl_price)
        executor.set_take_profit(symbol, "long", tp_price)
        logger.info(f"[开仓] {symbol} LONG 入场={fill_price} SL={sl_price} TP={tp_price} 金额={amount_usdt:.2f}U")


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
