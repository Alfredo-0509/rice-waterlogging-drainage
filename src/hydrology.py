"""Water-balance bucket + crop-damage PROXY for one rice paddy plot.

Scale defaults (see PROVENANCE.md):
  Area A ≈ 10 ha = 100_000 m²
  Root-zone depth Zr = 300 mm
  Field capacity FC = 0.32 m³/m³
  Saturation θs = 0.52 m³/m³
  Wilting / residual θr = 0.15 m³/m³
  Pump capacity Qp = 25 mm/h equivalent over the plot
      (= 25e-3 m/h × 1e5 m² = 2500 m³/h)

Formulas (explicit Euler, Δt = 10 min = 1/6 h):

  1) Soil moisture bucket
       θ_{t+1} = clip( θ_t + (R_infil - D_perc - E_crop) / Zr , θr, θs )
     where R_infil = min(R_slot, (θs - θ_t)*Zr)   [mm]
           D_perc  = k_perc * max(θ_t - FC, 0) * Zr   [mm]  (slow deep perc.)
           E_crop  ≈ 0 for storm window (negligible vs rain)

  2) Excess → surface ponding / ditch
       excess_mm = R_slot - R_infil
       h_{t+1} = h_t + excess_mm - Q_pump_mm - Q_weir_mm
     h is ponding / ditch stage in mm (0 = dry ditch floor).

  3) Pump
       Q_pump_mm = Qp_mm_h * Δt   if pump ON, else 0
       (capped so h never goes negative)

  4) Crop-damage PROXY (NOT RMB yield):
       score ∈ [0, 1] = 0.45*f_pond + 0.35*f_sat + 0.20*f_peak
       f_pond = piecewise(hours with h > 20 mm)
       f_sat  = piecewise(hours with θ > FC)
       f_peak = piecewise(peak ponding depth mm)

Severity labels from rolling hydrologic state:
  none | watch | waterlogging | severe   + score 0–1 + reasons.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

# ---- defaults ----
AREA_HA = 10.0
ZR_MM = 300.0
FC = 0.32
THETA_S = 0.52
THETA_R = 0.15
PUMP_MM_H = 25.0  # mm/h equivalent over plot
K_PERC = 0.08  # fraction of (θ-FC)*Zr drained per hour when wet
DT_H = 10.0 / 60.0
POND_HARM_MM = 20.0  # depth above which ponding harms rice canopy/roots


@dataclass
class HydroState:
    theta: float = 0.22
    pond_mm: float = 0.0
    pump_on: bool = False
    hours_ponded: float = 0.0
    hours_above_fc: float = 0.0
    peak_pond_mm: float = 0.0
    pumped_mm_cum: float = 0.0


@dataclass
class StepResult:
    theta: float
    pond_mm: float
    pump_on: bool
    rain_mm: float
    pumped_mm: float
    excess_mm: float
    severity_label: str
    severity_score: float
    reasons: str


def _piecewise_hours(h: float, knots: list[tuple[float, float]]) -> float:
    """Linear interpolate damage factor from (hours, factor) knots, clip 0–1."""
    if h <= knots[0][0]:
        return knots[0][1]
    for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
        if h <= x1:
            t = (h - x0) / max(x1 - x0, 1e-9)
            return float(np.clip(y0 + t * (y1 - y0), 0.0, 1.0))
    return knots[-1][1]


def damage_proxy(hours_ponded: float, hours_above_fc: float, peak_pond_mm: float) -> float:
    """Crop-loss PROXY in [0, 1] — NOT currency or certified yield loss."""
    f_pond = _piecewise_hours(
        hours_ponded,
        [(0, 0.0), (6, 0.15), (12, 0.40), (24, 0.75), (48, 1.0)],
    )
    f_sat = _piecewise_hours(
        hours_above_fc,
        [(0, 0.0), (12, 0.10), (24, 0.35), (48, 0.70), (72, 1.0)],
    )
    # peak depth factor
    if peak_pond_mm <= 20:
        f_peak = 0.0
    elif peak_pond_mm <= 50:
        f_peak = (peak_pond_mm - 20) / 30 * 0.4
    elif peak_pond_mm <= 100:
        f_peak = 0.4 + (peak_pond_mm - 50) / 50 * 0.4
    else:
        f_peak = min(1.0, 0.8 + (peak_pond_mm - 100) / 100 * 0.2)
    return float(np.clip(0.45 * f_pond + 0.35 * f_sat + 0.20 * f_peak, 0.0, 1.0))


def classify_severity(
    pond_mm: float,
    hours_ponded_window: float,
    rain_6h_mm: float,
    theta: float,
) -> tuple[str, float, str]:
    """Instantaneous / rolling severity for a time step.

    Returns (label, score 0–1, reasons).
    """
    reasons = []
    score = 0.0

    # rain contribution
    if rain_6h_mm >= 120:
        score += 0.35
        reasons.append(f"6h_rain={rain_6h_mm:.0f}mm")
    elif rain_6h_mm >= 60:
        score += 0.20
        reasons.append(f"6h_rain={rain_6h_mm:.0f}mm")
    elif rain_6h_mm >= 25:
        score += 0.08
        reasons.append(f"6h_rain={rain_6h_mm:.0f}mm")

    # ponding
    if pond_mm >= 80:
        score += 0.40
        reasons.append(f"pond={pond_mm:.0f}mm")
    elif pond_mm >= 40:
        score += 0.25
        reasons.append(f"pond={pond_mm:.0f}mm")
    elif pond_mm >= POND_HARM_MM:
        score += 0.12
        reasons.append(f"pond={pond_mm:.0f}mm")

    # duration in recent window
    if hours_ponded_window >= 12:
        score += 0.25
        reasons.append(f"ponded_h={hours_ponded_window:.1f}")
    elif hours_ponded_window >= 4:
        score += 0.12
        reasons.append(f"ponded_h={hours_ponded_window:.1f}")

    if theta >= THETA_S - 0.02:
        score += 0.10
        reasons.append(f"VWC≈sat({theta:.2f})")
    elif theta >= FC:
        score += 0.05
        reasons.append(f"VWC>FC({theta:.2f})")

    score = float(np.clip(score, 0.0, 1.0))
    if score < 0.15:
        label = "none"
    elif score < 0.35:
        label = "watch"
    elif score < 0.65:
        label = "waterlogging"
    else:
        label = "severe"
    return label, score, ";".join(reasons) if reasons else "quiet"


class WaterBalanceSim:
    """One-field soil bucket + ditch + pump simulator."""

    def __init__(
        self,
        pump_mm_h: float = PUMP_MM_H,
        fc: float = FC,
        theta_s: float = THETA_S,
        zr_mm: float = ZR_MM,
        initial_theta: float = 0.22,
    ):
        self.pump_mm_h = pump_mm_h
        self.fc = fc
        self.theta_s = theta_s
        self.zr_mm = zr_mm
        self.state = HydroState(theta=initial_theta)

    def reset(self, theta: float = 0.22, pond_mm: float = 0.0) -> None:
        self.state = HydroState(theta=theta, pond_mm=pond_mm)

    def step(
        self,
        rain_mm: float,
        pump_on: bool,
        rain_6h_mm: float = 0.0,
        recent_pond_hours: float = 0.0,
    ) -> StepResult:
        st = self.state
        rain_mm = max(0.0, float(rain_mm))

        # infiltration capacity this slot
        room = max(0.0, (self.theta_s - st.theta) * self.zr_mm)
        infil = min(rain_mm, room)
        excess = rain_mm - infil

        # percolation when above FC
        perc = K_PERC * max(st.theta - self.fc, 0.0) * self.zr_mm * DT_H

        theta = st.theta + (infil - perc) / self.zr_mm
        theta = float(np.clip(theta, THETA_R, self.theta_s))

        # ditch / ponding
        pond = st.pond_mm + excess
        pumped = 0.0
        if pump_on:
            pumped = min(pond, self.pump_mm_h * DT_H)
            pond -= pumped

        # tiny weir leak when very deep (overflow to canal) — keeps prototype stable
        if pond > 200:
            weir = min(pond - 200, 5.0)
            pond -= weir

        pond = max(0.0, pond)

        hours_ponded = st.hours_ponded + (DT_H if pond > POND_HARM_MM else 0.0)
        hours_fc = st.hours_above_fc + (DT_H if theta >= self.fc else 0.0)
        peak = max(st.peak_pond_mm, pond)

        self.state = HydroState(
            theta=theta,
            pond_mm=pond,
            pump_on=pump_on,
            hours_ponded=hours_ponded,
            hours_above_fc=hours_fc,
            peak_pond_mm=peak,
            pumped_mm_cum=st.pumped_mm_cum + pumped,
        )

        label, score, reasons = classify_severity(
            pond, recent_pond_hours, rain_6h_mm, theta
        )
        return StepResult(
            theta=theta,
            pond_mm=pond,
            pump_on=pump_on,
            rain_mm=rain_mm,
            pumped_mm=pumped,
            excess_mm=excess,
            severity_label=label,
            severity_score=score,
            reasons=reasons,
        )


def run_simulation(
    features: pd.DataFrame,
    policy_fn: Callable[[dict], bool],
    pump_mm_h: float = PUMP_MM_H,
    initial_theta: Optional[float] = None,
) -> pd.DataFrame:
    """Run water-balance with a pump policy.

    policy_fn(ctx) -> pump_on bool
    ctx keys: i, row, state, history (list of StepResult-like dicts), predicted_severity
    """
    feat = features.copy()
    if initial_theta is None:
        initial_theta = float(feat["vwc_m3m3"].iloc[0])
    sim = WaterBalanceSim(pump_mm_h=pump_mm_h, initial_theta=initial_theta)

    history: list[dict] = []
    rows = []
    pond_flag_window: list[float] = []  # last 24h of ponding flags

    for i, row in feat.iterrows():
        # recent ponded hours (~24 h = 144 slots)
        recent = sum(pond_flag_window[-144:]) * DT_H if pond_flag_window else 0.0
        ctx = {
            "i": i,
            "row": row,
            "state": sim.state,
            "history": history,
            "recent_pond_hours": recent,
            "predicted_severity": row.get("predicted_severity", "none"),
            "predicted_score": float(row.get("predicted_score", 0.0) or 0.0),
        }
        pump_on = bool(policy_fn(ctx))
        res = sim.step(
            rain_mm=float(row["rain_mm_10min"]),
            pump_on=pump_on,
            rain_6h_mm=float(row["rain_6h_mm"]),
            recent_pond_hours=recent,
        )
        pond_flag_window.append(1.0 if res.pond_mm > POND_HARM_MM else 0.0)
        rec = {
            "timestamp_local": row["timestamp_local"],
            "rain_mm_10min": res.rain_mm,
            "rain_mm_h": float(row["rain_mm_h"]),
            "rain_6h_mm": float(row["rain_6h_mm"]),
            "vwc_obs": float(row["vwc_m3m3"]),
            "theta": res.theta,
            "pond_mm": res.pond_mm,
            "pump_on": int(res.pump_on),
            "pumped_mm": res.pumped_mm,
            "excess_mm": res.excess_mm,
            "severity_label": res.severity_label,
            "severity_score": res.severity_score,
            "reasons": res.reasons,
            "hours_ponded_cum": sim.state.hours_ponded,
            "hours_above_fc_cum": sim.state.hours_above_fc,
            "peak_pond_mm": sim.state.peak_pond_mm,
            "predicted_severity": ctx["predicted_severity"],
            "predicted_score": ctx["predicted_score"],
        }
        history.append(rec)
        rows.append(rec)

    out = pd.DataFrame(rows)
    out["damage_proxy"] = damage_proxy(
        float(sim.state.hours_ponded),
        float(sim.state.hours_above_fc),
        float(sim.state.peak_pond_mm),
    )
    # also per-row running proxy for charts
    running = []
    for _, r in out.iterrows():
        running.append(
            damage_proxy(r["hours_ponded_cum"], r["hours_above_fc_cum"], r["peak_pond_mm"])
        )
    out["damage_proxy_running"] = running
    return out


def summarize_damage(sim_df: pd.DataFrame) -> dict:
    last = sim_df.iloc[-1]
    return {
        "hours_ponded": round(float(last["hours_ponded_cum"]), 2),
        "hours_above_fc": round(float(last["hours_above_fc_cum"]), 2),
        "peak_pond_mm": round(float(last["peak_pond_mm"]), 1),
        "damage_proxy": round(float(last["damage_proxy"]), 3),
        "max_severity_score": round(float(sim_df["severity_score"].max()), 3),
        "hours_pump_on": round(float(sim_df["pump_on"].sum() * DT_H), 2),
        "total_pumped_mm": round(float(sim_df["pumped_mm"].sum()), 1),
    }
