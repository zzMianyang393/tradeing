"""OKX实盘/模拟盘下单模块"""

import ccxt
from typing import Optional
from loguru import logger


class OrderExecutor:
    def __init__(self, config: dict):
        okx_cfg = config.get("okx", {})
        self.exchange = ccxt.okx({
            "apiKey": okx_cfg.get("api_key", ""),
            "secret": okx_cfg.get("secret_key", ""),
            "password": okx_cfg.get("passphrase", ""),
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })

        self.sandbox = okx_cfg.get("sandbox", True)
        if self.sandbox:
            self.exchange.set_sandbox_mode(True)
            logger.info("OKX 模拟盘模式 (sandbox)")
        else:
            logger.warning("OKX 实盘模式 - 请确认!")

    def set_leverage(self, symbol: str, leverage: int):
        try:
            self.exchange.set_leverage(leverage, symbol, params={"mgnMode": "cross"})
            logger.debug(f"设置杠杆: {symbol} {leverage}x")
        except Exception as e:
            logger.warning(f"设置杠杆失败 (可能已设置): {e}")

    def open_long(self, symbol: str, amount_usdt: float, leverage: int, entry_price: float) -> Optional[dict]:
        try:
            self.set_leverage(symbol, leverage)

            market = self.exchange.market(symbol)
            base_amount = amount_usdt * leverage / entry_price
            base_amount = self.exchange.amount_to_precision(symbol, base_amount)

            if float(base_amount) <= 0:
                logger.warning(f"计算数量为0，跳过: {symbol}")
                return None

            order = self.exchange.create_market_order(
                symbol, "buy", float(base_amount)
            )
            logger.info(f"开多成功: {order['id']} {symbol} 数量={base_amount} 杠杆={leverage}x")
            return order

        except Exception as e:
            logger.error(f"开多失败 {symbol}: {e}")
            return None

    def open_short(self, symbol: str, amount_usdt: float, leverage: int, entry_price: float) -> Optional[dict]:
        try:
            self.set_leverage(symbol, leverage)

            market = self.exchange.market(symbol)
            base_amount = amount_usdt * leverage / entry_price
            base_amount = self.exchange.amount_to_precision(symbol, base_amount)

            if float(base_amount) <= 0:
                logger.warning(f"计算数量为0，跳过: {symbol}")
                return None

            order = self.exchange.create_market_order(
                symbol, "sell", float(base_amount)
            )
            logger.info(f"开空成功: {order['id']} {symbol} 数量={base_amount} 杠杆={leverage}x")
            return order

        except Exception as e:
            logger.error(f"开空失败 {symbol}: {e}")
            return None

    def close_position(self, symbol: str, side: str) -> Optional[dict]:
        try:
            position = self.exchange.fetch_position(symbol)
            if not position or float(position.get("contracts", 0)) == 0:
                return None

            close_side = "sell" if side == "long" else "buy"
            amount = abs(float(position.get("contracts", 0)))

            order = self.exchange.create_market_order(
                symbol, close_side, amount
            )
            logger.info(f"平仓成功: {order['id']} {symbol}")
            return order

        except Exception as e:
            logger.error(f"平仓失败 {symbol}: {e}")
            return None

    def set_stop_loss(self, symbol: str, side: str, stop_price: float):
        try:
            trigger_side = "sell" if side == "long" else "buy"
            self.exchange.create_order(
                symbol,
                type="market",
                side=trigger_side,
                amount=1,
                params={
                    "tdMode": "cross",
                    "ordType": "conditional",
                    "slTriggerPx": str(stop_price),
                    "slOrdPx": "-1",
                },
            )
            logger.info(f"设置止损: {symbol} @ {stop_price}")
        except Exception as e:
            logger.warning(f"设置止损失败: {e}")

    def set_take_profit(self, symbol: str, side: str, tp_price: float):
        try:
            trigger_side = "sell" if side == "long" else "buy"
            self.exchange.create_order(
                symbol,
                type="market",
                side=trigger_side,
                amount=1,
                params={
                    "tdMode": "cross",
                    "ordType": "conditional",
                    "tpTriggerPx": str(tp_price),
                    "tpOrdPx": "-1",
                },
            )
            logger.info(f"设置止盈: {symbol} @ {tp_price}")
        except Exception as e:
            logger.warning(f"设置止盈失败: {e}")

    def get_balance(self) -> float:
        try:
            balance = self.exchange.fetch_balance(params={"type": "swap"})
            return float(balance.get("USDT", {}).get("free", 0))
        except Exception as e:
            logger.error(f"获取余额失败: {e}")
            return 0.0

    def get_positions(self) -> list:
        try:
            positions = self.exchange.fetch_positions()
            return [
                {
                    "symbol": p["symbol"],
                    "side": p["side"],
                    "contracts": p.get("contracts", 0),
                    "entryPrice": p.get("entryPrice", 0),
                    "unrealizedPnl": p.get("unrealizedPnl", 0),
                    "leverage": p.get("leverage", 1),
                }
                for p in positions
                if float(p.get("contracts", 0)) != 0
            ]
        except Exception as e:
            logger.error(f"获取持仓失败: {e}")
            return []

    def test_connection(self) -> bool:
        try:
            balance = self.get_balance()
            logger.info(f"连接成功! 模拟盘余额: {balance} USDT")
            return True
        except Exception as e:
            logger.error(f"连接失败: {e}")
            return False
