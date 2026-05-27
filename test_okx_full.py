"""OKX 完整下单流程测试"""
import yaml
from pathlib import Path
from execution.order_executor import OrderExecutor

def main():
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    executor = OrderExecutor(config)

    # 1. 测试连接
    if not executor.test_connection():
        print("连接失败")
        return

    # 2. 获取余额
    balance = executor.get_balance()
    print(f"余额: {balance:.2f} USDT")

    # 3. 获取持仓
    positions = executor.get_positions()
    print(f"当前持仓: {positions}")

    # 4. 开多 BTC/USDT:USDT (200 USDT, 5x)
    symbol = "BTC/USDT:USDT"
    leverage = 5
    amount_usdt = 200.0

    import ccxt
    ticker = executor.exchange.fetch_ticker(symbol)
    entry_price = ticker['last']
    print(f"当前价格: {entry_price}")

    order = executor.open_long(symbol, amount_usdt, leverage, entry_price)
    if not order:
        print("开仓失败")
        return

    print(f"开仓成功: {order['id']}")

    # 5. 设置止损 (入场价 -0.8%)
    sl_price = entry_price * (1 - 0.008)
    executor.set_stop_loss(symbol, "long", sl_price)

    # 6. 设置止盈 (入场价 +0.5%)
    tp_price = entry_price * (1 + 0.005)
    executor.set_take_profit(symbol, "long", tp_price)

    # 7. 检查持仓
    import time
    time.sleep(2)
    positions = executor.get_positions()
    print(f"持仓状态: {positions}")

    # 8. 平仓
    print("5秒后平仓...")
    time.sleep(5)
    close_result = executor.close_position(symbol, "long")
    if close_result:
        print(f"平仓成功: {close_result['id']}")
    else:
        print("平仓失败或无持仓")

    # 9. 最终余额
    final_balance = executor.get_balance()
    print(f"最终余额: {final_balance:.2f} USDT")

if __name__ == "__main__":
    main()
