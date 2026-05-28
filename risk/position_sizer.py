"""仓位管理模块 - 动态仓位和杠杆计算"""

from __future__ import annotations

from dataclasses import dataclass



@dataclass
class PositionSize:
    amount_usdt: float  # 开仓金额(USDT)
    leverage: int        # 杠杆倍数
    risk_amount: float   # 本笔交易最大亏损金额
    position_pct: float  # 占总资金比例


class PositionSizer:
    def __init__(self, config: dict = None):
        if config is None:
            config = {}
        risk_cfg = config.get("risk", {})
        lev_cfg = config.get("leverage", {})

        self.max_position_pct = risk_cfg.get("max_position_pct", 0.02)
        self.max_concurrent = risk_cfg.get("max_concurrent_positions", 5)
        self.max_daily_loss = risk_cfg.get("max_daily_loss", 0.05)

        self.min_leverage = lev_cfg.get("min", 3)
        self.max_leverage = lev_cfg.get("max", 20)
        self.thresholds = lev_cfg.get("signal_thresholds", {
            "strong": 0.8,
            "medium": 0.6,
            "weak": 0.4,
        })

    def calculate_leverage(self, signal_strength: float) -> int:
        if signal_strength >= self.thresholds["strong"]:
            return self.max_leverage
        elif signal_strength >= self.thresholds["medium"]:
            mid = (self.min_leverage + self.max_leverage) // 2
            return mid
        elif signal_strength >= self.thresholds["weak"]:
            return self.min_leverage + 2
        else:
            return self.min_leverage

    def calculate_position(
        self,
        balance: float,
        entry_price: float,
        stop_distance_pct: float,
        signal_strength: float,
        current_positions: int = 0,
    ) -> PositionSize:
        if current_positions >= self.max_concurrent:
            return PositionSize(0, 0, 0, 0)

        leverage = self.calculate_leverage(signal_strength)

        max_risk = balance * self.max_position_pct
        risk_amount = max_risk * (1 - current_positions / self.max_concurrent * 0.5)

        if stop_distance_pct <= 0:
            stop_distance_pct = 0.01

        notional = risk_amount / stop_distance_pct
        amount_usdt = notional / leverage
        amount_usdt = min(amount_usdt, balance * 0.3)

        if amount_usdt < 1:
            return PositionSize(0, 0, 0, 0)

        position_pct = amount_usdt / balance

        return PositionSize(
            amount_usdt=round(amount_usdt, 4),
            leverage=leverage,
            risk_amount=round(risk_amount, 4),
            position_pct=round(position_pct, 4),
        )

    def can_open_position(
        self,
        balance: float,
        daily_pnl: float,
        current_positions: int,
    ) -> tuple[bool, str]:
        if current_positions >= self.max_concurrent:
            return False, f"已达最大持仓数 {self.max_concurrent}"

        if balance < 1:
            return False, "余额不足"

        max_daily_loss = balance * self.max_daily_loss
        if daily_pnl < -max_daily_loss:
            return False, f"已达日最大亏损限制 {self.max_daily_loss:.1%}"

        return True, "可以开仓"
