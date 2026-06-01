"""OKX实盘/模拟盘下单模块"""

import time
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
            self.exchange.set_leverage(leverage, symbol, params={"mgnMode": "isolated"})
            logger.debug(f"设置杠杆: {symbol} {leverage}x")
        except Exception as e:
            logger.warning(f"设置杠杆失败 (可能已设置): {e}")

    def open_long(self, symbol: str, amount_usdt: float, leverage: int, entry_price: float) -> Optional[dict]:
        try:
            self.set_leverage(symbol, leverage)

            market = self.exchange.market(symbol)
            # OKX USDT保证金合约：sz = 合约张数 (不是BTC数量)
            # 1张BTC = contractSize(0.01) BTC, 1张ETH = contractSize(0.1) ETH
            contract_size = float(market.get("contractSize", 1))
            notional = amount_usdt * leverage
            # 合约张数 = 名义价值(USDT) / 每张价值(USDT)
            contracts = notional / (contract_size * entry_price)
            contracts = self.exchange.amount_to_precision(symbol, contracts)
            min_contracts = float(market.get("limits", {}).get("amount", {}).get("min", 1))
            contracts_f = float(contracts) if contracts else 0
            if contracts_f < min_contracts:
                min_margin = min_contracts * contract_size * entry_price / leverage
                logger.warning(f"合约张数{contracts} < 最小{min_contracts}张，资金不足跳过: {symbol} (需~{min_margin:.1f}U保证金)")
                return None

            order = self.exchange.create_market_order(
                symbol, "buy", float(contracts),
                params={"tdMode": "isolated"}
            )
            logger.info(f"开多成功: {order['id']} {symbol} {contracts}张 保证金={amount_usdt:.2f}U 杠杆={leverage}x")
            return order

        except Exception as e:
            logger.error(f"开多失败 {symbol}: {e}")
            return None

    def open_short(self, symbol: str, amount_usdt: float, leverage: int, entry_price: float) -> Optional[dict]:
        try:
            self.set_leverage(symbol, leverage)

            market = self.exchange.market(symbol)
            contract_size = float(market.get("contractSize", 1))
            notional = amount_usdt * leverage
            contracts = notional / (contract_size * entry_price)
            contracts = self.exchange.amount_to_precision(symbol, contracts)
            min_contracts = float(market.get("limits", {}).get("amount", {}).get("min", 1))
            contracts_f = float(contracts) if contracts else 0
            if contracts_f < min_contracts:
                min_margin = min_contracts * contract_size * entry_price / leverage
                logger.warning(f"合约张数{contracts} < 最小{min_contracts}张，资金不足跳过: {symbol} (需~{min_margin:.1f}U保证金)")
                return None

            order = self.exchange.create_market_order(
                symbol, "sell", float(contracts),
                params={"tdMode": "isolated"}
            )
            logger.info(f"开空成功: {order['id']} {symbol} {contracts}张 保证金={amount_usdt:.2f}U 杠杆={leverage}x")
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

    def set_stop_loss(self, symbol: str, side: str, stop_price: float, contracts: float = 0):
        """设置止损，传入实际持仓张数。失败最多重试3次。"""
        trigger_side = "sell" if side == "long" else "buy"
        # 如果没传contracts，尝试从OKX获取当前持仓
        if contracts <= 0:
            try:
                pos = self.exchange.fetch_position(symbol)
                contracts = abs(float(pos.get("contracts", 0))) if pos else 0
            except Exception:
                pass
        if contracts <= 0:
            logger.warning(f"设置止损跳过: {symbol} 无法获取持仓张数")
            return

        for attempt in range(3):
            try:
                self.exchange.create_order(
                    symbol,
                    type="market",
                    side=trigger_side,
                    amount=contracts,
                    params={
                        "tdMode": "isolated",
                        "ordType": "conditional",
                        "slTriggerPx": str(stop_price),
                        "slOrdPx": "-1",
                    },
                )
                logger.info(f"设置止损: {symbol} @ {stop_price} 平{contracts}张")
                return
            except Exception as e:
                if attempt < 2:
                    logger.warning(f"设置止损失败(第{attempt+1}次): {e}，重试...")
                    time.sleep(1)
                else:
                    logger.error(f"设置止损最终失败: {symbol} {e}")

    def set_take_profit(self, symbol: str, side: str, tp_price: float, contracts: float = 0):
        """设置止盈，传入实际持仓张数。失败最多重试3次。"""
        trigger_side = "sell" if side == "long" else "buy"
        if contracts <= 0:
            try:
                pos = self.exchange.fetch_position(symbol)
                contracts = abs(float(pos.get("contracts", 0))) if pos else 0
            except Exception:
                pass
        if contracts <= 0:
            logger.warning(f"设置止盈跳过: {symbol} 无法获取持仓张数")
            return

        for attempt in range(3):
            try:
                self.exchange.create_order(
                    symbol,
                    type="market",
                    side=trigger_side,
                    amount=contracts,
                    params={
                        "tdMode": "isolated",
                        "ordType": "conditional",
                        "tpTriggerPx": str(tp_price),
                        "tpOrdPx": "-1",
                    },
                )
                logger.info(f"设置止盈: {symbol} @ {tp_price} 平{contracts}张")
                return
            except Exception as e:
                if attempt < 2:
                    logger.warning(f"设置止盈失败(第{attempt+1}次): {e}，重试...")
                    time.sleep(1)
                else:
                    logger.error(f"设置止盈最终失败: {symbol} {e}")

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
            result = []
            for p in positions:
                if float(p.get("contracts", 0)) != 0:
                    # 从 market info 获取 contractSize
                    try:
                        market = self.exchange.market(p["symbol"])
                        cs = float(market.get("contractSize", 0.01))
                    except Exception:
                        cs = 0.01
                    result.append({
                        "symbol": p["symbol"],
                        "side": p["side"],
                        "contracts": p.get("contracts", 0),
                        "entryPrice": p.get("entryPrice", 0),
                        "unrealizedPnl": p.get("unrealizedPnl", 0),
                        "leverage": p.get("leverage", 1),
                        "contractSize": cs,
                    })
            return result
        except Exception as e:
            logger.error(f"获取持仓失败: {e}")
            return []

    def get_recent_fills(self, symbol: str, limit: int = 5) -> list:
        """获取最近成交记录，用于检测OKX条件单触发的平仓"""
        try:
            trades = self.exchange.fetch_my_trades(symbol, limit=limit)
            return trades
        except Exception as e:
            logger.warning(f"获取成交记录失败 {symbol}: {e}")
            return []

    def test_connection(self) -> bool:
        try:
            balance = self.get_balance()
            logger.info(f"连接成功! 模拟盘余额: {balance} USDT")
            return True
        except Exception as e:
            logger.error(f"连接失败: {e}")
            return False
