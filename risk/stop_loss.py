"""止损管理模块"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class StopLossResult:
    stop_price: float
    stop_pct: float
    method: str


class StopLossManager:
    def __init__(self, config: dict = None):
        if config is None:
            config = {}
        sl_cfg = config.get("stop_loss", {})
        self.atr_multiplier = sl_cfg.get("atr_multiplier", 1.5)
        self.max_pct = sl_cfg.get("max_pct", 0.03)
        self.trailing_activate = sl_cfg.get("trailing_activate", 0.02)
        self.trailing_callback = sl_cfg.get("trailing_callback", 0.012)

    def calculate_atr_stop(
        self, entry_price: float, atr: float, direction: str
    ) -> StopLossResult:
        stop_distance = atr * self.atr_multiplier
        stop_pct = stop_distance / entry_price

        if stop_pct > self.max_pct:
            stop_pct = self.max_pct

        if direction == "long":
            stop_price = entry_price * (1 - stop_pct)
        else:
            stop_price = entry_price * (1 + stop_pct)

        return StopLossResult(
            stop_price=round(stop_price, 8),
            stop_pct=round(stop_pct, 6),
            method="ATR",
        )

    def calculate_fixed_stop(
        self, entry_price: float, direction: str
    ) -> StopLossResult:
        if direction == "long":
            stop_price = entry_price * (1 - self.max_pct)
        else:
            stop_price = entry_price * (1 + self.max_pct)

        return StopLossResult(
            stop_price=round(stop_price, 8),
            stop_pct=self.max_pct,
            method="FIXED",
        )

    def should_move_stop(
        self,
        entry_price: float,
        current_price: float,
        current_stop: float,
        direction: str,
    ) -> Optional[float]:
        if direction == "long":
            unrealized_pct = (current_price - entry_price) / entry_price
            if unrealized_pct >= self.trailing_activate:
                new_stop = current_price * (1 - self.trailing_callback)
                if new_stop > current_stop:
                    return round(new_stop, 8)
        else:
            unrealized_pct = (entry_price - current_price) / entry_price
            if unrealized_pct >= self.trailing_activate:
                new_stop = current_price * (1 + self.trailing_callback)
                if new_stop < current_stop:
                    return round(new_stop, 8)

        return None

    def check_stop_hit(
        self, current_price: float, stop_price: float, direction: str
    ) -> bool:
        if direction == "long":
            return current_price <= stop_price
        else:
            return current_price >= stop_price
