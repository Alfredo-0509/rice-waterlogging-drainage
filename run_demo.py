#!/usr/bin/env python3
"""One-command laptop demo for rice waterlogging drainage.

Steps:
  1. Generate raw Pi CSV if missing
  2. Ingest + QC → sensor_clean.csv + features.csv
  3. Train severity predictor (sklearn) on physics labels
  4. Run three pump policies (do-nothing, rule, automated)
  5. Write outputs/ CSVs + matplotlib figures
  6. Print how to open Streamlit
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.drainage import (  # noqa: E402
    build_workflow_log,
    make_automated_policy,
    make_rule_policy,
    policy_do_nothing,
)
from src.hydrology import run_simulation, summarize_damage  # noqa: E402
from src.predict import train_and_predict  # noqa: E402

OUT = ROOT / "outputs"


def _run(script: Path) -> None:
    print(f"→ {script.relative_to(ROOT)}")
    subprocess.check_call([sys.executable, str(script)], cwd=str(ROOT))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = ROOT / "data" / "raw" / "pi_sensor_log.csv"
    if not raw.exists():
        _run(ROOT / "scripts" / "generate_raw_csv.py")
    else:
        # regenerate to keep demo deterministic / fresh
        _run(ROOT / "scripts" / "generate_raw_csv.py")

    _run(ROOT / "scripts" / "ingest_qc.py")

    features = pd.read_csv(ROOT / "data" / "processed" / "features.csv")
    print(f"Features: {len(features)} rows")

    # --- severity prediction ---
    print("→ sklearn severity prediction")
    feat_pred, metrics = train_and_predict(features)
    feat_pred.to_csv(OUT / "features_with_prediction.csv", index=False)
    pd.DataFrame([metrics]).to_csv(OUT / "predict_metrics.csv", index=False)
    print("  metrics:", metrics)

    # --- three policies ---
    print("→ simulate do_nothing")
    sim_n = run_simulation(feat_pred, policy_do_nothing)
    sim_n.to_csv(OUT / "sim_do_nothing.csv", index=False)

    print("→ simulate rule_based")
    sim_r = run_simulation(feat_pred, make_rule_policy())
    sim_r.to_csv(OUT / "sim_rule_based.csv", index=False)

    print("→ simulate automated")
    sim_a = run_simulation(feat_pred, make_automated_policy())
    sim_a.to_csv(OUT / "sim_automated.csv", index=False)

    # workflow logs
    build_workflow_log(sim_r, "rule_based").to_csv(OUT / "pump_workflow_rule_based.csv", index=False)
    wf_a = build_workflow_log(sim_a, "automated")
    wf_a.to_csv(OUT / "pump_workflow_automated.csv", index=False)

    # damage summary
    summary = pd.DataFrame(
        [
            {"policy": "do_nothing", **summarize_damage(sim_n)},
            {"policy": "rule_based", **summarize_damage(sim_r)},
            {"policy": "automated", **summarize_damage(sim_a)},
        ]
    )
    summary.to_csv(OUT / "damage_summary.csv", index=False)
    print("\n=== Damage summary ===")
    print(summary.to_string(index=False))

    # figures
    print("\n→ save figures")
    from app.dashboard import save_figures

    data = {
        "features": features,
        "nothing": sim_n,
        "rule": sim_r,
        "auto": sim_a,
        "summary": summary,
        "workflow": wf_a,
        "metrics": pd.DataFrame([metrics]),
    }
    paths = save_figures(data)
    for p in paths:
        print(f"  {p}")

    # compact JSON for parent agent
    result = {
        "metrics": metrics,
        "summary": summary.to_dict(orient="records"),
        "figures": [str(p) for p in paths],
    }
    (OUT / "demo_result.json").write_text(json.dumps(result, indent=2))

    print("\n=== Demo complete ===")
    print(f"Outputs under: {OUT}")
    print("Open Streamlit UI (optional):")
    print(f"  cd {ROOT} && .venv/bin/streamlit run app/dashboard.py")
    print("Headless figures already saved — open the PNGs in outputs/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
