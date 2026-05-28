"""仓位管理模块 - 动态仓位和杠杆计算（含凯利公式）"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


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

        # 凯利公式参数
        self.kelly_fraction = risk_cfg.get("kelly_fraction", 0.5)  # 半凯利
        self.min_position_pct = risk_cfg.get("min_position_pct", 0.05)
        self.max_position_pct_cap = risk_cfg.get("max_position_pct_cap", 0.30)

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

    def kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """凯利公式计算最优仓位比例
        
        Kelly % = (W * R - (1-W)) / R
        其中 W = 胜率, R = 盈亏比 (avg_win / avg_loss)
        
        使用半凯利（kelly_fraction）降低风险
        """
        if avg_loss <= 0 or win_rate <= 0:
            return self.min_position_pct
        
        R = avg_win / avg_loss  # 盈亏比
        W = win_rate
        
        kelly = (W * R - (1 - W)) / R
        
        # 应用凯利分数（半凯利更安全）
        kelly *= self.kelly_fraction
        
        # 限制范围
        kelly = max(kelly, self.min_position_pct)
        kelly = min(kelly, self.max_position_pct_cap)
        
        return kelly

    def calculate_position(
        self,
        balance: float,
        entry_price: float,
        stop_distance_pct: float,
        signal_strength: float,
        current_positions: int = 0,
        win_rate: float = None,
        avg_win: float = None,
        avg_loss: float = None,
    ) -> PositionSize:
        if current_positions >= self.max_concurrent:
            return PositionSize(0, 0, 0, 0)

        leverage = self.calculate_leverage(signal_strength)

        max_risk = balance * self.max_position_pct
        risk_amount = max_risk * (1 - current_positions / self.max_concurrent * 0.5)

        if stop_distance_pct <= 0:
            stop_distance_pct = 0.01

        # 如果有历史交易数据，用凯利公式计算仓位
        if win_rate is not None and avg_win is not None and avg_loss is not None:
            kelly_pct = self.kelly_criterion(win_rate, avg_win, avg_loss)
            amount_usdt = balance * kelly_pct
        else:
            # 默认仓位：25%资金，上限5U
            amount_usdt = min(balance * 0.25, 5.0)
        
        amount_usdt = max(amount_usdt, 1.0)  # 最少1U

        if amount_usdt > balance:
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
