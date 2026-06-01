"""平掉OKX模拟盘所有持仓 - 使用OKX原生API"""
import ccxt
import yaml

with open('config/settings.yaml') as f:
    config = yaml.safe_load(f)

okx_cfg = config['okx']
exchange = ccxt.okx({
    'apiKey': okx_cfg['api_key'],
    'secret': okx_cfg['secret_key'],
    'password': okx_cfg['passphrase'],
    'enableRateLimit': True,
})
exchange.set_sandbox_mode(True)

# 获取OKX原始持仓
raw = exchange.private_get_account_positions(params={'instType': 'SWAP'})
positions = raw.get('data', [])

print(f"当前持仓: {len(positions)}个")

for p in positions:
    inst_id = p['instId']
    pos = float(p['pos'])
    
    if pos == 0:
        continue
    
    side = 'long' if pos > 0 else 'short'
    contracts = abs(pos)
    mgn = p.get('mgnMode', 'cross')
    
    print(f"  {inst_id} {side} {contracts}张 mgnMode={mgn}")
    
    try:
        # 使用OKX原生平仓接口
        result = exchange.private_post_trade_close_position({
            'instId': inst_id,
            'mgnMode': mgn,
            'posSide': 'net',
        })
        print(f"    -> 平仓结果: {result.get('msg', result)}")
    except Exception as e:
        print(f"    -> 失败: {e}")

import time
time.sleep(3)

# 确认
raw2 = exchange.private_get_account_positions(params={'instType': 'SWAP'})
active = [p for p in raw2.get('data', []) if float(p.get('pos', 0)) != 0]
print(f"\n平仓后持仓: {len(active)}个")

balance = exchange.fetch_balance(params={'type': 'swap'})
usdt = balance.get('USDT', {})
print(f"模拟盘可用: {usdt.get('free', 0):.2f}U")
