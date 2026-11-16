#!/usr/bin/env python3
"""Generate a physically plausible 60-hour Pi sensor log (10-min).

Storm narrative (Asia/Shanghai) — two typhoon rain bands:
  hours 0–6    dry
  hours 6–14   FIRST rain band (moderate) → soil wets, light ponding risk
  hours 14–20  lull
  hours 20–28  MAIN typhoon burst (intense hyetograph — no wind/surge)
  hours 28–40  waterlogged plateau / residual rain
  hours 40–60  recession

Randomness (the RNG is **not** seeded by default — every run is a fresh
"storm", exactly like a real logger):
  --rain-noise          σ (mm/10min) jitter added to the rainfall hyetograph
                        while it is raining (physical variability)
  --rain-noise-dry      σ (mm/10min) jitter during dry slots
  --rain-sensor-noise   σ (mm/10min) measurement noise of the rain sensor
                        (added to the logged value only, not to the physics)
  --vwc-sensor-noise    σ (m³/m³) measurement noise of the soil VWC probe
  --seed                optional integer → makes a run reproducible again

QC edge cases are injected on top of the noise: stuck VWC, rain spike,
30-min missing gap. Their position/size are slightly jittered each run.
"""
from __future__ import annotations

import argparse
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

# Default noise levels (mm/10min unless stated). Override on the CLI / in the GUI.
DEFAULT_RAIN_NOISE = 0.05
DEFAULT_RAIN_NOISE_DRY = 0.01
DEFAULT_RAIN_SENSOR_NOISE = 0.02
DEFAULT_VWC_SENSOR_NOISE = 0.002


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


def _validate_noise(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite number >= 0 (got {value!r})")
    return value


def generate(
    out: Path = OUT,
    *,
    seed: int | None = None,
    rain_noise: float = DEFAULT_RAIN_NOISE,
    rain_noise_dry: float = DEFAULT_RAIN_NOISE_DRY,
    rain_sensor_noise: float = DEFAULT_RAIN_SENSOR_NOISE,
    vwc_sensor_noise: float = DEFAULT_VWC_SENSOR_NOISE,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build the long-format Pi log, write it to *out* and return the frame.

    ``seed=None`` (the default) uses fresh OS entropy → no fixed seed.
    """
    rain_noise = _validate_noise("rain_noise", rain_noise)
    rain_noise_dry = _validate_noise("rain_noise_dry", rain_noise_dry)
    rain_sensor_noise = _validate_noise("rain_sensor_noise", rain_sensor_noise)
    vwc_sensor_noise = _validate_noise("vwc_sensor_noise", vwc_sensor_noise)

    rng = np.random.default_rng(seed)
    vwc = 0.24
    rain_truth, vwc_truth_series = [], []

    # --- physical truth: hyetograph + soil bucket, with rain variability ---
    for i in range(N_SLOTS):
        t_h = i * DT_MIN / 60.0
        base = rain_hyetograph_mm10(t_h)
        sigma = rain_noise if base > 0 else rain_noise_dry
        mm10 = max(0.0, base + float(rng.normal(0.0, sigma)))
        vwc = vwc_truth(t_h, mm10, vwc)
        rain_truth.append(mm10)
        vwc_truth_series.append(vwc)

    # --- sensor measurements: independent measurement noise ---
    rain_meas = [
        max(0.0, v + float(rng.normal(0.0, rain_sensor_noise)))
        for v in rain_truth
    ]
    vwc_meas = [
        float(np.clip(v + rng.normal(0.0, vwc_sensor_noise), 0.02, 0.60))
        for v in vwc_truth_series
    ]

    # --- QC edge cases (positions/magnitudes jittered, narrative preserved) ---
    stuck_i0 = int(rng.integers(int(1.0 * 6), int(3.0 * 6)))  # early dry — true stuck
    stuck_len = int(rng.integers(8, 11))
    stuck_val = vwc_meas[stuck_i0]
    for j in range(stuck_len):
        if stuck_i0 + j < N_SLOTS:
            vwc_meas[stuck_i0 + j] = stuck_val

    spike_i = int(rng.integers(int(3.5 * 6), int(5.5 * 6)))  # dry period spike
    rain_meas[spike_i] = float(rng.uniform(12.0, 24.0))

    gap_start = int(rng.integers(int(32.0 * 6), int(35.0 * 6)))
    gap_idxs = set(range(gap_start, gap_start + 3))

    rows = []
    for i in range(N_SLOTS):
        t_local = START + timedelta(minutes=DT_MIN * i)
        t_utc = t_local.astimezone(timezone.utc)
        ts_l, ts_u = t_local.isoformat(), t_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        missing = i in gap_idxs
        mm10 = rain_meas[i]
        v = vwc_meas[i]

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
        adc = vwc_to_adc(v)
        flag = "ok"
        if stuck_i0 <= i < stuck_i0 + stuck_len:
            flag = "stuck"
        elif adc >= 3900 or v >= 0.54:
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
                value_si=round(v, 4),
                unit="m3/m3",
                cal_version=CAL,
                quality_flag=flag,
            )
        )

    df = pd.DataFrame(rows)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    if verbose:
        seed_txt = "random (no seed)" if seed is None else str(seed)
        print(
            "Noise: rain σ={:.3f} mm/10min (dry {:.3f}), rain sensor σ={:.3f}, "
            "VWC sensor σ={:.4f} m3/m3 | seed: {}".format(
                rain_noise, rain_noise_dry, rain_sensor_noise, vwc_sensor_noise, seed_txt
            )
        )
        print(f"Wrote {out} ({len(df)} rows, {N_SLOTS} time slots × 2 sensors)")
    return df


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate the synthetic (but physically shaped) Pi sensor log.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out", type=Path, default=OUT, help="output CSV path")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="optional integer seed; omit for a different log on every run",
    )
    p.add_argument(
        "--rain-noise",
        type=float,
        default=DEFAULT_RAIN_NOISE,
        help="σ of rainfall hyetograph noise while raining (mm/10min)",
    )
    p.add_argument(
        "--rain-noise-dry",
        type=float,
        default=DEFAULT_RAIN_NOISE_DRY,
        help="σ of rainfall hyetograph noise in dry slots (mm/10min)",
    )
    p.add_argument(
        "--rain-sensor-noise",
        type=float,
        default=DEFAULT_RAIN_SENSOR_NOISE,
        help="σ of rain sensor measurement noise (mm/10min)",
    )
    p.add_argument(
        "--vwc-sensor-noise",
        type=float,
        default=DEFAULT_VWC_SENSOR_NOISE,
        help="σ of soil VWC sensor measurement noise (m3/m3)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        generate(
            args.out,
            seed=args.seed,
            rain_noise=args.rain_noise,
            rain_noise_dry=args.rain_noise_dry,
            rain_sensor_noise=args.rain_sensor_noise,
            vwc_sensor_noise=args.vwc_sensor_noise,
        )
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
