#!/usr/bin/env python3
"""Ingest messy Pi long-format CSV → QC → sensor_clean.csv + features.csv.

QC rules:
  - range checks (rain 0–25 mm/10min; VWC 0.05–0.55)
  - stuck: identical non-missing value ≥ 6 consecutive slots
  - spikes: |Δ| larger than physical max for one slot
  - missing timestamps: reindex to regular 10-min grid; leave
   (never sentinels)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "pi_sensor_log.csv"
CLEAN = ROOT / "data" / "processed" / "sensor_clean.csv"
FEATURES = ROOT / "data" / "processed" / "features.csv"

FC = 0.32  # field capacity m3/m3 (clay_loam rice paddy)
DT_H = 10 / 60.0
RAIN_MAX = 25.0
VWC_MIN, VWC_MAX = 0.05, 0.55
STUCK_N = 6
RAIN_SPIKE = 8.0  # mm jump in one 10-min slot vs previous
VWC_SPIKE = 0.08


def _parse_ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=False, format="ISO8601")


def qc_series(values: pd.Series, kind: str) -> tuple[pd.Series, pd.Series]:
    """Return (cleaned_values, flags)."""
    flags = pd.Series("ok", index=values.index, dtype=object)
    out = values.astype(float).copy()

    missing = out.isna()
    flags.loc[missing] = "missing"

    if kind == "rain":
        bad = (~missing) & ((out < 0) | (out > RAIN_MAX))
        flags.loc[bad] = "suspect"
        # clamp extreme for downstream but keep flag
        out.loc[bad & (out > RAIN_MAX)] = RAIN_MAX
        out.loc[bad & (out < 0)] = 0.0
        # spikes
        d = out.diff().abs()
        spike = (~missing) & (d > RAIN_SPIKE)
        flags.loc[spike] = "suspect"
        # replace spike with previous valid for hydrology (flag kept)
        out.loc[spike] = np.nan
        out = out.ffill(limit=1)
    else:
        bad = (~missing) & ((out < VWC_MIN) | (out > VWC_MAX))
        flags.loc[bad] = "suspect"
        out = out.clip(VWC_MIN, VWC_MAX)
        d = out.diff().abs()
        spike = (~missing) & (d > VWC_SPIKE)
        flags.loc[spike] = "suspect"
        sat = (~missing) & (out >= 0.54)
        flags.loc[sat] = "saturated"

    # stuck detection — skip near-saturation plateaus (physically flat when wet)
    same = out == out.shift(1)
    run = same.groupby((~same).cumsum()).cumsum()
    stuck = run >= (STUCK_N - 1)
    if kind == "vwc":
        stuck = stuck & (out < 0.50)  # flat at saturation is OK, not a sensor fault
    flags.loc[stuck & flags.eq("ok")] = "stuck"

    return out, flags


def run_ingest(
    raw_path: Path = RAW,
    clean_path: Path = CLEAN,
    features_path: Path = FEATURES,
    verbose: bool = True,
) -> tuple[Path, Path]:
    """QC the raw log and write sensor_clean.csv + features.csv."""
    raw = pd.read_csv(raw_path)
    raw["timestamp_local"] = _parse_ts(raw["timestamp_local"])
    # empty strings → NA
    for col in ("raw_adc", "value_si"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    rain = raw[raw["sensor_type"] == "rain_cap"].copy()
    vwc = raw[raw["sensor_type"] == "soil_vwc_cap"].copy()
    rain = rain.set_index("timestamp_local").sort_index()
    vwc = vwc.set_index("timestamp_local").sort_index()

    # full regular grid
    full_idx = pd.date_range(rain.index.min(), rain.index.max(), freq="10min")
    rain = rain.reindex(full_idx)
    vwc = vwc.reindex(full_idx)

    rain_val, rain_flag = qc_series(rain["value_si"], "rain")
    vwc_val, vwc_flag = qc_series(vwc["value_si"], "vwc")

    def to_utc_str(ts_index):
        if getattr(ts_index, "tz", None) is None:
            ts_index = ts_index.tz_localize("Asia/Shanghai")
        return ts_index.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")

    utc_strs = to_utc_str(full_idx)
    local_strs = []
    for t in full_idx:
        if getattr(t, "tzinfo", None) is None:
            t = t.tz_localize("Asia/Shanghai")
        else:
            t = t.tz_convert("Asia/Shanghai")
        local_strs.append(t.isoformat())

    # rebuild long clean
    rain_clean = pd.DataFrame(
        {
            "timestamp_local": local_strs,
            "timestamp_utc": utc_strs,
            "farm_id": "JH-AH-01",
            "plot_id": "P-RICE-A",
            "device_id": "RPi-01",
            "sensor_id": "S-RAIN-01",
            "sensor_type": "rain_cap",
            "raw_adc": rain["raw_adc"].values if "raw_adc" in rain else np.nan,
            "value_si": rain_val.values,
            "unit": "mm/10min",
            "cal_version": "cal_v1",
            "quality_flag": rain_flag.values,
        }
    )
    vwc_clean = pd.DataFrame(
        {
            "timestamp_local": local_strs,
            "timestamp_utc": utc_strs,
            "farm_id": "JH-AH-01",
            "plot_id": "P-RICE-A",
            "device_id": "RPi-01",
            "sensor_id": "S-VWC-01",
            "sensor_type": "soil_vwc_cap",
            "raw_adc": vwc["raw_adc"].values if "raw_adc" in vwc else np.nan,
            "value_si": vwc_val.values,
            "unit": "m3/m3",
            "cal_version": "cal_v1",
            "quality_flag": vwc_flag.values,
        }
    )

    clean = pd.concat([rain_clean, vwc_clean], ignore_index=True)
    clean = clean.sort_values(["timestamp_local", "sensor_type"]).reset_index(drop=True)
    clean_path = Path(clean_path)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(clean_path, index=False)

    # --- features (wide) ---
    # fill remaining rain/vwc gaps lightly for feature engineering
    rain_f = rain_val.interpolate(limit=3).fillna(0.0)
    vwc_f = vwc_val.interpolate(limit=3).ffill().bfill()

    rain_mm_h = rain_f * 6.0  # mm/10min → mm/h
    rain_1h = rain_f.rolling(6, min_periods=1).sum()
    rain_6h = rain_f.rolling(36, min_periods=1).sum()
    above_fc = (vwc_f >= FC).astype(float)
    hours_above = above_fc.cumsum() * DT_H

    qsum = []
    for i in range(len(full_idx)):
        rf, vf = rain_flag.iloc[i], vwc_flag.iloc[i]
        if rf == "missing" or vf == "missing":
            qsum.append(
                "gap_filled"
                if not (pd.isna(rain_f.iloc[i]) or pd.isna(vwc_f.iloc[i]))
                else "missing"
            )
        elif rf == "suspect" or vf == "suspect":
            qsum.append("spike_flagged")
        elif vf == "stuck":
            qsum.append("stuck_flagged")
        elif vf == "saturated":
            qsum.append("saturated")
        else:
            qsum.append("ok")

    feat = pd.DataFrame(
        {
            "timestamp_local": local_strs,
            "rain_mm_h": rain_mm_h.values.round(3),
            "rain_1h_mm": rain_1h.values.round(3),
            "rain_6h_mm": rain_6h.values.round(3),
            "vwc_m3m3": vwc_f.values.round(4),
            "hours_above_field_capacity": hours_above.values.round(3),
            "quality_summary": qsum,
            # keep interval rain for simulator
            "rain_mm_10min": rain_f.values.round(3),
        }
    )
    features_path = Path(features_path)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    feat.to_csv(features_path, index=False)

    if verbose:
        print(f"Wrote {clean_path} ({len(clean)} rows)")
        print(f"Wrote {features_path} ({len(feat)} rows)")
        print("QC flag counts (rain):", rain_flag.value_counts().to_dict())
        print("QC flag counts (vwc):", vwc_flag.value_counts().to_dict())
    return clean_path, features_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QC raw Pi log → clean sensor + features CSVs.")
    parser.add_argument("--raw", type=Path, default=RAW, help="input raw log")
    parser.add_argument("--clean", type=Path, default=CLEAN, help="output clean long CSV")
    parser.add_argument("--features", type=Path, default=FEATURES, help="output wide features CSV")
    args = parser.parse_args(argv)
    run_ingest(args.raw, args.clean, args.features)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
