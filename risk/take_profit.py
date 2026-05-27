"""止盈管理模块"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TakeProfitResult:
    target_price: float
    target_pct: float
    method: str
    partial_close_price: Optional[float] = None
    partial_close_pct: Optional[float] = None


class TakeProfitManager:
    def __init__(self, config: dict = None):
        if config is None:
            config = {}
        tp_cfg = config.get("take_profit", {})
        sl_cfg = config.get("stop_loss", {})

        self.risk_reward_ratio = tp_cfg.get("risk_reward_ratio", 2.0)
        self.partial_close_pct = tp_cfg.get("partial_close_pct", 0.5)
        self.partial_close_trigger = tp_cfg.get("partial_close_trigger", 0.025)

    def calculate_target(
        self, entry_price: float, stop_distance_pct: float, direction: str
    ) -> TakeProfitResult:
        target_pct = stop_distance_pct * self.risk_reward_ratio

        if direction == "long":
            target_price = entry_price * (1 + target_pct)
            partial_price = entry_price * (1 + self.partial_close_trigger)
        else:
            target_price = entry_price * (1 - target_pct)
            partial_price = entry_price * (1 - self.partial_close_trigger)

        return TakeProfitResult(
            target_price=round(target_price, 8),
            target_pct=round(target_pct, 6),
            method="RISK_REWARD",
            partial_close_price=round(partial_price, 8),
            partial_close_pct=self.partial_close_pct,
        )

    def should_take_profit(
        self, current_price: float, entry_price: float, direction: str
    ) -> tuple[bool, str]:
        if direction == "long":
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price

        if pnl_pct >= self.partial_close_trigger:
            return True, f"触发部分止盈 ({pnl_pct:.2%})"

        return False, ""

    def should_full_close(
        self, current_price: float, target_price: float, direction: str
    ) -> bool:
        if direction == "long":
            return current_price >= target_price
        else:
            return current_price <= target_price

    def get_partial_close_size(self, position_size: float) -> float:
        return position_size * self.partial_close_pct
