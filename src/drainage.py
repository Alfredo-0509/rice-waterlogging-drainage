"""Drainage pump policies: do-nothing, rule-based hysteresis, automated (prediction-aware).

Constraints:
  - pump capacity enforced in hydrology.WaterBalanceSim
  - minimum on-time (default 30 min = 3 slots) to avoid chatter
  - no GPIO / no live hardware
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

# Rule-based thresholds (document in PROVENANCE)
POND_ON_MM = 40.0
POND_OFF_MM = 12.0
VWC_ON = 0.50
VWC_OFF = 0.40
MIN_ON_SLOTS = 3  # 30 minutes

# Automated: start earlier / stop later using predicted severity
AUTO_POND_ON_MM = 15.0
AUTO_POND_OFF_MM = 5.0
AUTO_VWC_ON = 0.42
SEVERITY_EARLY = {"watch", "waterlogging", "severe"}
SEVERITY_HOLD = {"waterlogging", "severe"}


@dataclass
class PumpController:
    """Tracks min-on-time and last action for hysteresis helpers."""

    min_on_slots: int = MIN_ON_SLOTS
    _on_slots: int = 0
    _pump_on: bool = False

    def apply(self, want_on: bool) -> bool:
        if want_on:
            self._pump_on = True
            self._on_slots += 1
            return True
        # want off — respect min on-time
        if self._pump_on and self._on_slots < self.min_on_slots:
            self._on_slots += 1
            return True
        self._pump_on = False
        self._on_slots = 0
        return False


def policy_do_nothing(ctx: dict) -> bool:
    return False


def make_rule_policy(ctrl: PumpController | None = None) -> Callable[[dict], bool]:
    """Ponding-primary farmer rule: react when standing water is obvious.

    VWC is NOT used for ON (avoids early soil-threshold pumping). Hysteresis on
    pond depth only so the pump can idle in a lull once the ditch is cleared.
    """
    ctrl = ctrl or PumpController()

    def policy(ctx: dict) -> bool:
        st = ctx["state"]
        pond = st.pond_mm
        if not ctrl._pump_on:
            want = pond >= POND_ON_MM
        else:
            want = pond > POND_OFF_MM
        return ctrl.apply(want)

    return policy


def make_automated_policy(ctrl: PumpController | None = None) -> Callable[[dict], bool]:
    """Use predicted severity + current state to pump earlier / stop later.

    Readable one-function decision:
      - if predicted severe/waterlogging → lower ON thresholds (act early)
      - if predicted watch and soil rising → prep-pump when pond > soft threshold
      - if predicted none and pond/VWC low → allow OFF (after min on-time)
      - hold ON longer while prediction stays elevated
    """
    ctrl = ctrl or PumpController()

    def policy(ctx: dict) -> bool:
        st = ctx["state"]
        pred = str(ctx.get("predicted_severity", "none")).lower()
        score = float(ctx.get("predicted_score", 0.0) or 0.0)
        pond = st.pond_mm
        theta = st.theta
        rain_1h = float(ctx["row"].get("rain_1h_mm", 0.0) or 0.0)

        # early-action thresholds when model anticipates trouble
        if pred in ("severe",) or score >= 0.65:
            on_pond, off_pond, on_vwc = 8.0, 3.0, 0.40
        elif pred in ("waterlogging",) or score >= 0.35:
            on_pond, off_pond, on_vwc = AUTO_POND_ON_MM, AUTO_POND_OFF_MM, AUTO_VWC_ON
        elif pred in ("watch",) or score >= 0.15:
            on_pond, off_pond, on_vwc = 20.0, 8.0, 0.44
        else:
            on_pond, off_pond, on_vwc = POND_ON_MM, POND_OFF_MM, VWC_ON

        rain_6h = float(ctx["row"].get("rain_6h_mm", 0.0) or 0.0)
        rain_rate = float(ctx["row"].get("rain_mm_h", 0.0) or 0.0)

        if not ctrl._pump_on:
            # Early action: clear ditch before next band when model sees risk
            early = pred in SEVERITY_EARLY or score >= 0.15
            intensifying = rain_1h >= 5.0 or rain_rate >= 12.0 or rain_6h >= 25.0
            want = (
                (pond >= on_pond)
                or (theta >= on_vwc)
                or (early and pond >= 2.0)
                or (early and intensifying and theta >= 0.38)
                or (pred in SEVERITY_HOLD and pond >= 1.0)
            )
        else:
            # stop later while prediction stays elevated (clear ditch through lull)
            calm = pred in ("none",) and score < 0.15 and rain_1h < 2.0
            want = not ((pond <= off_pond) and (theta <= 0.42) and calm)

        return ctrl.apply(want)

    return policy


def reason_for_action(
    pump_on: bool,
    prev_on: bool,
    pred: str,
    pond: float,
    theta: float,
    policy_name: str,
) -> str:
    if pump_on and not prev_on:
        return f"{policy_name}: ON (pred={pred}, pond={pond:.1f}mm, VWC={theta:.2f})"
    if (not pump_on) and prev_on:
        return f"{policy_name}: OFF (pred={pred}, pond={pond:.1f}mm, VWC={theta:.2f})"
    if pump_on:
        return f"{policy_name}: hold ON (pred={pred}, pond={pond:.1f}mm)"
    return f"{policy_name}: stay OFF (pred={pred})"


def build_workflow_log(sim_df: pd.DataFrame, policy_name: str) -> pd.DataFrame:
    """timestamp, predicted_severity, action, reason."""
    rows = []
    prev = False
    for _, r in sim_df.iterrows():
        on = bool(r["pump_on"])
        action = "PUMP_ON" if on else "PUMP_OFF"
        # log every state change + periodic heartbeat every 6 slots when on
        change = on != prev
        rows.append(
            {
                "timestamp": r["timestamp_local"],
                "predicted_severity": r.get("predicted_severity", "none"),
                "action": action,
                "reason": reason_for_action(
                    on,
                    prev,
                    str(r.get("predicted_severity", "none")),
                    float(r["pond_mm"]),
                    float(r["theta"]),
                    policy_name,
                ),
                "state_change": int(change),
            }
        )
        prev = on
    return pd.DataFrame(rows)
