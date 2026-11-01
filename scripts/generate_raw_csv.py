#!/usr/bin/env python3
"""Generate a physically plausible 60-hour Pi sensor log (10-min).

Storm narrative (Asia/Shanghai) — two typhoon rain bands:
  hours 0–6    dry
  hours 6–14   FIRST rain band (moderate) → soil wets, light ponding risk
  hours 14–20  lull
  hours 20–28  MAIN typhoon burst (intense hyetograph — no wind/surge)
  hours 28–40  waterlogged plateau / residual rain
  hours 40–60  recession

Injects QC edge cases: stuck VWC, rain spike, missing gap.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "pi_sensor_log.csv"

FARM, PLOT, DEVICE = "JH-AH-01", "P-RICE-A", "RPi-01"
RAIN_ID, VWC_ID = "S-RAIN-01", "S-VWC-01"
CAL = "cal_v1"
TZ = timezone(timedelta(hours=8))
START = datetime(2025, 8, 12, 6, 0, 0, tzinfo=TZ)
N_SLOTS = 60 * 6
DT_MIN = 10


def rain_hyetograph_mm10(t_h: float) -> float:
    """mm in a 10-min slot at hour t_h since start."""
    # --- band 1: solid wetting rain (builds pre-burst ponding if no early pump) ---
    if 6.0 <= t_h < 8.0:
        return 1.5 + 2.0 * (t_h - 6.0)  # → ~5.5
    if 8.0 <= t_h < 12.0:
        return 5.0 + 1.2 * math.sin(math.pi * (t_h - 8.0) / 4.0)  # ~5–6.2
    if 12.0 <= t_h < 14.0:
        return max(0.3, 5.0 - 2.2 * (t_h - 12.0))
    # --- lull (standing water can be cleared by early automation) ---
    if 14.0 <= t_h < 20.0:
        return 0.02
    # --- main typhoon burst ---
    if 20.0 <= t_h < 22.0:
        return 1.5 + 3.5 * (t_h - 20.0)  # 1.5 → 8.5
    if 22.0 <= t_h < 24.5:
        return 8.0 + 2.0 * math.sin(math.pi * (t_h - 22.0) / 2.5)  # peak ~10
    if 24.5 <= t_h < 27.0:
        return max(0.8, 8.0 - 2.8 * (t_h - 24.5))
    # --- plateau / residual ---
    if 27.0 <= t_h < 36.0:
        return 0.30 + 0.45 * abs(math.sin(0.35 * t_h))
    if 36.0 <= t_h < 42.0:
        return 0.06 * max(0.0, math.sin(0.25 * t_h))
    return 0.0


def vwc_truth(t_h: float, rain_mm10: float, prev: float) -> float:
    fc, sat, dry = 0.32, 0.52, 0.18
    infil = min(rain_mm10, max(0.0, (sat - prev) * 300.0))
    nxt = prev + infil / 300.0
    if nxt > fc and rain_mm10 < 0.5:
        nxt -= 0.005 * (nxt - fc)
    if t_h > 42 and rain_mm10 < 0.05:
        nxt -= 0.0012
    return float(np.clip(nxt, dry, sat))


def rain_to_adc(mm10: float) -> int:
    return int(np.clip(round(200 + mm10 * 150), 0, 4095))


def vwc_to_adc(vwc: float) -> int:
    return int(np.clip(round(800 + (vwc - 0.05) * 6000), 0, 4095))


def main() -> None:
    rng = np.random.default_rng(42)
    rows = []
    vwc = 0.24
    rain_series, vwc_series = [], []

    for i in range(N_SLOTS):
        t_h = i * DT_MIN / 60.0
        mm10 = rain_hyetograph_mm10(t_h)
        mm10 = max(0.0, mm10 + float(rng.normal(0, 0.05 if mm10 > 0 else 0.01)))
        vwc = vwc_truth(t_h, mm10, vwc)
        rain_series.append(mm10)
        vwc_series.append(vwc)

    # QC edge cases
    stuck_i0 = int(2 * 6)  # early dry — true stuck, not saturation plateau
    stuck_val = vwc_series[stuck_i0]
    for j in range(8):
        vwc_series[stuck_i0 + j] = stuck_val

    spike_i = int(4 * 6)  # dry period spike
    rain_series[spike_i] = 18.0

    gap_idxs = set(range(int(33 * 6), int(33 * 6) + 3))

    for i in range(N_SLOTS):
        t_local = START + timedelta(minutes=DT_MIN * i)
        t_utc = t_local.astimezone(timezone.utc)
        ts_l, ts_u = t_local.isoformat(), t_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        missing = i in gap_idxs
        mm10, vwc = rain_series[i], vwc_series[i]

        if missing:
            for sid, stype, unit in [
                (RAIN_ID, "rain_cap", "mm/10min"),
                (VWC_ID, "soil_vwc_cap", "m3/m3"),
            ]:
                rows.append(
                    dict(
                        timestamp_local=ts_l,
                        timestamp_utc=ts_u,
                        farm_id=FARM,
                        plot_id=PLOT,
                        device_id=DEVICE,
                        sensor_id=sid,
                        sensor_type=stype,
                        raw_adc="",
                        value_si="",
                        unit=unit,
                        cal_version=CAL,
                        quality_flag="missing",
                    )
                )
            continue

        # rain
        adc = rain_to_adc(mm10)
        flag = "ok"
        if adc >= 4090:
            flag = "saturated"
        elif i == spike_i or mm10 > 15:
            flag = "suspect"
        rows.append(
            dict(
                timestamp_local=ts_l,
                timestamp_utc=ts_u,
                farm_id=FARM,
                plot_id=PLOT,
                device_id=DEVICE,
                sensor_id=RAIN_ID,
                sensor_type="rain_cap",
                raw_adc=adc,
                value_si=round(mm10, 3),
                unit="mm/10min",
                cal_version=CAL,
                quality_flag=flag,
            )
        )

        # vwc
        adc = vwc_to_adc(vwc)
        flag = "ok"
        if stuck_i0 <= i < stuck_i0 + 8:
            flag = "stuck"
        elif adc >= 3900 or vwc >= 0.54:
            flag = "saturated"
        rows.append(
            dict(
                timestamp_local=ts_l,
                timestamp_utc=ts_u,
                farm_id=FARM,
                plot_id=PLOT,
                device_id=DEVICE,
                sensor_id=VWC_ID,
                sensor_type="soil_vwc_cap",
                raw_adc=adc,
                value_si=round(vwc, 4),
                unit="m3/m3",
                cal_version=CAL,
                quality_flag=flag,
            )
        )

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT} ({len(df)} rows, {N_SLOTS} time slots × 2 sensors)")


if __name__ == "__main__":
    main()
