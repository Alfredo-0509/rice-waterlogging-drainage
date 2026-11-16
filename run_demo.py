#!/usr/bin/env python3
"""One-command laptop demo for rice waterlogging drainage.

Steps:
  1. Generate raw Pi CSV (fresh randomness every run unless --seed is given)
  2. Ingest + QC → sensor_clean.csv + features.csv
  3. Train severity predictor (sklearn) on physics labels
  4. Run the selected hydrology model(s) × pump policies
  5. Write outputs/ CSVs + matplotlib figures
  6. Print how to open the Tkinter GUI / Streamlit app

Examples
--------
    python3 run_demo.py                          # baseline bucket model
    python3 run_demo.py --model two_layer        # pick a hydrology model
    python3 run_demo.py --all-models             # compare all models
    python3 run_demo.py --seed 7 \
        --rain-noise 0.2 --vwc-sensor-noise 0.01 # reproducible + noisier sensors
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import pipeline  # noqa: E402
from src.hydrology import DEFAULT_MODEL, HYDRO_MODELS  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the full rice-waterlogging drainage demo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        action="append",
        choices=list(HYDRO_MODELS),
        default=None,
        help="hydrology simulation model (repeatable)",
    )
    p.add_argument(
        "--all-models",
        action="store_true",
        help="simulate every model in src/hydrology.py",
    )
    p.add_argument(
        "--policy",
        action="append",
        choices=list(pipeline.POLICY_ORDER),
        default=None,
        help="drainage policy to simulate (repeatable)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="optional RNG seed; omit for a fresh random sensor log each run",
    )
    p.add_argument(
        "--rain-noise",
        type=float,
        default=pipeline.DEFAULT_RAIN_NOISE,
        help="σ of rainfall hyetograph noise while raining (mm/10min)",
    )
    p.add_argument(
        "--rain-noise-dry",
        type=float,
        default=pipeline.DEFAULT_RAIN_NOISE_DRY,
        help="σ of rainfall hyetograph noise in dry slots (mm/10min)",
    )
    p.add_argument(
        "--rain-sensor-noise",
        type=float,
        default=pipeline.DEFAULT_RAIN_SENSOR_NOISE,
        help="σ of rain sensor measurement noise (mm/10min)",
    )
    p.add_argument(
        "--vwc-sensor-noise",
        type=float,
        default=pipeline.DEFAULT_VWC_SENSOR_NOISE,
        help="σ of soil VWC sensor measurement noise (m3/m3)",
    )
    p.add_argument("--no-figures", action="store_true", help="skip matplotlib figures")
    return p


def _models_from_args(args: argparse.Namespace) -> tuple[str, ...]:
    chosen: list[str] = []
    if args.all_models:
        chosen.extend(HYDRO_MODELS)
    for m in args.model or []:
        if m not in chosen:
            chosen.append(m)
    return tuple(chosen) if chosen else (DEFAULT_MODEL,)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = pipeline.PipelineConfig(
        seed=args.seed,
        rain_noise=args.rain_noise,
        rain_noise_dry=args.rain_noise_dry,
        rain_sensor_noise=args.rain_sensor_noise,
        vwc_sensor_noise=args.vwc_sensor_noise,
        models=_models_from_args(args),
        policies=tuple(args.policy) if args.policy else pipeline.POLICY_ORDER,
        figures=not args.no_figures,
    )
    try:
        pipeline.run_pipeline(cfg, log=print)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}")
        return 2

    print("\n=== Demo complete ===")
    print("Interactive GUI (Tkinter):")
    print(f"  cd {ROOT} && python3 app/gui.py")
    print("Optional Streamlit UI:")
    print(f"  cd {ROOT} && .venv/bin/streamlit run app/dashboard.py")
    print("Headless figures already saved — open the PNGs in outputs/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
