"""关闭所有持仓"""
import yaml
from execution.order_executor import OrderExecutor

config = yaml.safe_load(open("config/settings.yaml", "r", encoding="utf-8"))
executor = OrderExecutor(config)

positions = executor.get_positions()
print(f"当前持仓: {len(positions)} 个")
for p in positions:
    print(f"  {p['symbol']} {p['side']} {p['contracts']}张")
    executor.close_position(p["symbol"], p["side"])

print(f"最终余额: {executor.get_balance():.2f} USDT")
