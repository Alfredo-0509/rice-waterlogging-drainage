"""End-to-end pipeline steps, shared by the CLI, the Streamlit app and the Tk GUI.

Every step is a plain function that takes a :class:`PipelineConfig` (where
relevant) and an optional ``log`` callable, writes its artefacts under
``data/`` or ``outputs/``, and returns paths / frames. No GUI code lives here,
so the same logic can be driven from a script, a notebook or Tkinter.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `from scripts...` when imported from anywhere
    sys.path.insert(0, str(ROOT))

from scripts.generate_raw_csv import (  # noqa: E402
    DEFAULT_RAIN_NOISE,
    DEFAULT_RAIN_NOISE_DRY,
    DEFAULT_RAIN_SENSOR_NOISE,
    DEFAULT_VWC_SENSOR_NOISE,
    generate as generate_raw_csv,
)
from scripts.ingest_qc import run_ingest  # noqa: E402
from src.drainage import (  # noqa: E402
    build_workflow_log,
    make_automated_policy,
    make_rule_policy,
    policy_do_nothing,
)
from src.hydrology import (  # noqa: E402
    DEFAULT_MODEL,
    HYDRO_MODELS,
    get_model,
    run_simulation,
    summarize_damage,
)
from src.predict import train_and_predict  # noqa: E402

# ---------------------------------------------------------------- paths
RAW = ROOT / "data" / "raw" / "pi_sensor_log.csv"
CLEAN = ROOT / "data" / "processed" / "sensor_clean.csv"
FEATURES = ROOT / "data" / "processed" / "features.csv"
OUT = ROOT / "outputs"
FEATURES_PRED = OUT / "features_with_prediction.csv"
METRICS = OUT / "predict_metrics.csv"
SUMMARY = OUT / "damage_summary.csv"
RESULT_JSON = OUT / "demo_result.json"
# Written by every simulation run; the GUI inspector uses it to show ONLY the
# CSVs that were actually simulated (older runs stay on disk but are hidden).
MANIFEST = OUT / "simulation_manifest.json"

POLICIES: dict[str, Callable[[dict], bool] | None] = {
    "do_nothing": None,
    "rule_based": None,
    "automated": None,
}
POLICY_LABELS = {
    "do_nothing": "Do nothing",
    "rule_based": "Rule based",
    "automated": "Automated (ML)",
}
POLICY_ORDER = ("do_nothing", "rule_based", "automated")

# The three categories used to group / sort CSVs in the GUI selector.
CSV_TYPE_ORDER = (
    "Raw sensor log",
    "Clean sensor",
    "Features",
    "Predictions",
    "Model metrics",
    "Damage summary",
    "Simulation",
    "Pump workflow",
)
CSV_SORT_CATEGORIES = ("Type of data", "Models", "Treatments")
SHARED_CATEGORY = "Shared"

_SIM_RE = re.compile(
    r"^sim_(?P<policy>do_nothing|rule_based|automated)(?:__(?P<model>[a-z_]+))?\.csv$"
)
_WF_RE = re.compile(
    r"^pump_workflow_(?P<policy>do_nothing|rule_based|automated)"
    r"(?:__(?P<model>[a-z_]+))?\.csv$"
)


def make_policy(policy_key: str) -> Callable[[dict], bool]:
    """Fresh policy instance (controllers are stateful, so never reuse one)."""
    if policy_key == "do_nothing":
        return policy_do_nothing
    if policy_key == "rule_based":
        return make_rule_policy()
    if policy_key == "automated":
        return make_automated_policy()
    raise ValueError(
        f"Unknown policy {policy_key!r}; choose one of: {', '.join(POLICY_ORDER)}"
    )


def sim_path(policy: str, model: str = DEFAULT_MODEL) -> Path:
    """Canonical per-model simulation CSV path.

    The default model keeps the original file names (``sim_rule_based.csv``),
    other models are suffixed (``sim_rule_based__two_layer.csv``).
    """
    if model == DEFAULT_MODEL:
        return OUT / f"sim_{policy}.csv"
    return OUT / f"sim_{policy}__{model}.csv"


def workflow_path(policy: str, model: str = DEFAULT_MODEL) -> Path:
    if model == DEFAULT_MODEL:
        return OUT / f"pump_workflow_{policy}.csv"
    return OUT / f"pump_workflow_{policy}__{model}.csv"


def parse_sim_name(name: str) -> Optional[tuple[str, str]]:
    """``sim_rule_based__two_layer.csv`` → (``rule_based``, ``two_layer``)."""
    m = _SIM_RE.match(name)
    if not m:
        return None
    return m.group("policy"), m.group("model") or DEFAULT_MODEL


def parse_workflow_name(name: str) -> Optional[tuple[str, str]]:
    """``pump_workflow_automated__plough_pan.csv`` → (policy, model)."""
    m = _WF_RE.match(name)
    if not m:
        return None
    return m.group("policy"), m.group("model") or DEFAULT_MODEL


# ---------------------------------------------------------------- config
@dataclass
class PipelineConfig:
    """Everything the pipeline needs; also mirrors the GUI / CLI controls."""

    seed: Optional[int] = None
    rain_noise: float = DEFAULT_RAIN_NOISE
    rain_noise_dry: float = DEFAULT_RAIN_NOISE_DRY
    rain_sensor_noise: float = DEFAULT_RAIN_SENSOR_NOISE
    vwc_sensor_noise: float = DEFAULT_VWC_SENSOR_NOISE
    models: tuple[str, ...] = (DEFAULT_MODEL,)
    policies: tuple[str, ...] = POLICY_ORDER
    figures: bool = True

    def validate(self) -> "PipelineConfig":
        for name in ("rain_noise", "rain_noise_dry", "rain_sensor_noise", "vwc_sensor_noise"):
            value = float(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be >= 0 (got {value})")
            setattr(self, name, value)
        if self.seed is not None:
            self.seed = int(self.seed)
        unknown = [m for m in self.models if m not in HYDRO_MODELS]
        if unknown:
            raise ValueError(
                f"Unknown model(s) {unknown}; choose from {list(HYDRO_MODELS)}"
            )
        bad = [p for p in self.policies if p not in POLICY_ORDER]
        if bad:
            raise ValueError(f"Unknown policy/policies {bad}; choose from {POLICY_ORDER}")
        if not self.models:
            raise ValueError("Select at least one hydrology model")
        if not self.policies:
            raise ValueError("Select at least one drainage policy")
        return self

    @property
    def primary_model(self) -> str:
        return self.models[0] if self.models else DEFAULT_MODEL


# ---------------------------------------------------------------- helpers
def _log(log: Optional[Callable[[str], None]], msg: str) -> None:
    if log is not None:
        log(msg)


def _fmt_seed(seed: Optional[int]) -> str:
    return "random" if seed is None else str(seed)


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------- steps
def generate_raw(cfg: PipelineConfig, log: Optional[Callable[[str], None]] = print) -> Path:
    """Step 1 — synthetic Pi sensor log (no fixed RNG seed by default)."""
    OUT.mkdir(parents=True, exist_ok=True)
    _log(
        log,
        "① Generating raw sensor log "
        f"(seed={_fmt_seed(cfg.seed)}, rain σ={cfg.rain_noise} mm/10min, "
        f"dry σ={cfg.rain_noise_dry}, rain-sensor σ={cfg.rain_sensor_noise}, "
        f"VWC-sensor σ={cfg.vwc_sensor_noise})",
    )
    df = generate_raw_csv(
        RAW,
        seed=cfg.seed,
        rain_noise=cfg.rain_noise,
        rain_noise_dry=cfg.rain_noise_dry,
        rain_sensor_noise=cfg.rain_sensor_noise,
        vwc_sensor_noise=cfg.vwc_sensor_noise,
        verbose=False,
    )
    _log(log, f"   → {RAW.relative_to(ROOT)} ({len(df)} rows, {RAW.stat().st_size:,} bytes)")
    return RAW


def run_qc(log: Optional[Callable[[str], None]] = print) -> tuple[Path, Path]:
    """Step 2 — QC + feature engineering."""
    _log(log, "② QC / feature engineering")
    if not RAW.exists():
        raise FileNotFoundError(f"{RAW} is missing — generate the raw log first")
    clean, features = run_ingest(RAW, CLEAN, FEATURES, verbose=False)
    clean_df = pd.read_csv(clean)
    counts = (
        clean_df.groupby(["sensor_type", "quality_flag"]).size().sort_values(ascending=False)
    )
    _log(log, f"   → {clean.relative_to(ROOT)} ({len(clean_df)} rows)")
    _log(log, f"   → {features.relative_to(ROOT)} ({len(pd.read_csv(features))} rows)")
    for (sensor, flag), n in counts.items():
        _log(log, f"     {sensor:<13} {flag:<10} {n:>5}")
    return clean, features


def run_prediction(
    log: Optional[Callable[[str], None]] = print,
) -> tuple[Path, dict]:
    """Step 3 — train the sklearn severity model and predict every slot."""
    _log(log, "③ Training severity model (GradientBoosting, time-ordered split)")
    features = _read_csv(FEATURES)
    if features is None:
        raise FileNotFoundError(f"{FEATURES} is missing — run QC first")
    feat_pred, metrics = train_and_predict(features)
    OUT.mkdir(parents=True, exist_ok=True)
    feat_pred.to_csv(FEATURES_PRED, index=False)
    pd.DataFrame([metrics]).to_csv(METRICS, index=False)
    for key in ("accuracy", "dummy_accuracy", "mae", "n_train", "n_test"):
        if key in metrics:
            _log(log, f"   {key:<15} {metrics[key]}")
    _log(log, f"   → {FEATURES_PRED.relative_to(ROOT)} ({len(feat_pred)} rows)")
    return FEATURES_PRED, metrics


def load_sim_inputs(
    log: Optional[Callable[[str], None]] = print,
) -> pd.DataFrame:
    """Features + predictions, falling back to physics-only features."""
    feat = _read_csv(FEATURES_PRED)
    if feat is not None:
        return feat
    feat = _read_csv(FEATURES)
    if feat is None:
        raise FileNotFoundError(f"{FEATURES} is missing — run QC first")
    _log(log, "   ! predictions missing — simulating with physics severity only")
    feat["predicted_severity"] = "none"
    feat["predicted_score"] = 0.0
    return feat


def run_simulations(
    cfg: PipelineConfig,
    log: Optional[Callable[[str], None]] = print,
) -> pd.DataFrame:
    """Step 4 — run every selected model × policy and rebuild the summary."""
    cfg.validate()
    OUT.mkdir(parents=True, exist_ok=True)
    feat = load_sim_inputs(log)
    _log(log, f"④ Water-balance simulation on {len(feat)} slots")

    for model_key in cfg.models:
        spec = get_model(model_key)
        _log(log, f"   model {spec.key}: {spec.label}")
        _log(log, f"     {spec.description}")
        _log(log, f"     params: {spec.summary()}")
        for policy_key in cfg.policies:
            sim = run_simulation(feat, make_policy(policy_key), model=model_key)
            path = sim_path(policy_key, model_key)
            sim.to_csv(path, index=False)
            build_workflow_log(sim, policy_key).to_csv(
                workflow_path(policy_key, model_key), index=False
            )
            summ = summarize_damage(sim)
            _log(
                log,
                "     {: <14} ponded={:>6.2f} h  peak={:>6.1f} mm  "
                "proxy={:.3f}  pump={:>5.2f} h".format(
                    POLICY_LABELS[policy_key],
                    summ["hours_ponded"],
                    summ["peak_pond_mm"],
                    summ["damage_proxy"],
                    summ["hours_pump_on"],
                ),
            )
    summary = build_summary(log)
    manifest = write_simulation_manifest(cfg.models, cfg.policies)
    _log(log, f"   → {manifest.relative_to(ROOT)} (latest simulated CSVs)")
    return summary


def build_summary(log: Optional[Callable[[str], None]] = None) -> pd.DataFrame:
    """Scan ``outputs/sim_*.csv`` and rebuild ``damage_summary.csv``.

    Scanning (instead of only using this run) keeps results from earlier model
    runs in the comparison, which is what the GUI summary tab displays.
    """
    rows = []
    model_rank = {m: i for i, m in enumerate(HYDRO_MODELS)}
    policy_rank = {p: i for i, p in enumerate(POLICY_ORDER)}
    for path in sorted(OUT.glob("sim_*.csv")):
        parsed = parse_sim_name(path.name)
        if parsed is None:
            continue
        policy, model = parsed
        df = _read_csv(path)
        if df is None or df.empty:
            continue
        row = {
            "model": model,
            "model_label": get_model(model).label,
            "policy": policy,
            "policy_label": POLICY_LABELS.get(policy, policy),
            **summarize_damage(df),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    summary = pd.DataFrame(rows)
    summary["_m"] = summary["model"].map(lambda m: model_rank.get(m, 99))
    summary["_p"] = summary["policy"].map(lambda p: policy_rank.get(p, 99))
    summary = summary.sort_values(["_m", "_p"]).drop(columns=["_m", "_p"])
    summary.to_csv(SUMMARY, index=False)
    if log is not None:
        best = summary.sort_values("damage_proxy").iloc[0]
        _log(
            log,
            "   damage summary: {} scenario(s), best = {} / {} (PROXY {:.3f})".format(
                len(summary),
                best["model"],
                best["policy"],
                best["damage_proxy"],
            ),
        )
    return summary


def load_outputs(primary_model: Optional[str] = None) -> dict[str, pd.DataFrame]:
    """Load artefacts for the figure builder (tolerant to missing pieces)."""
    data: dict[str, pd.DataFrame] = {}
    features = _read_csv(FEATURES)
    if features is not None:
        data["features"] = features

    summary = _read_csv(SUMMARY)
    if summary is not None and not summary.empty:
        data["summary"] = summary
        if primary_model is None:
            primary_model = str(summary["model"].iloc[0])

    def pick(policy: str) -> Optional[pd.DataFrame]:
        if primary_model is not None:
            df = _read_csv(sim_path(policy, primary_model))
            if df is not None:
                return df
        for path in sorted(OUT.glob(f"sim_{policy}*.csv")):
            df = _read_csv(path)
            if df is not None:
                return df
        return None

    mapping = {"nothing": "do_nothing", "rule": "rule_based", "auto": "automated"}
    for key, policy in mapping.items():
        df = pick(policy)
        if df is not None:
            data[key] = df

    wf = _read_csv(workflow_path("automated", primary_model or DEFAULT_MODEL))
    if wf is None:
        candidates = sorted(OUT.glob("pump_workflow_automated*.csv"))
        wf = _read_csv(candidates[0]) if candidates else None
    if wf is not None:
        data["workflow"] = wf

    metrics = _read_csv(METRICS)
    if metrics is not None:
        data["metrics"] = metrics
    return data


def build_figures(
    log: Optional[Callable[[str], None]] = print,
) -> list[Path]:
    """Step 5 — matplotlib figures from whatever outputs exist."""
    from app.dashboard import save_figures

    data = load_outputs()
    if not data:
        raise FileNotFoundError("No outputs to chart — run the simulation first")
    _log(log, "⑤ Building figures")
    paths = save_figures(data)
    for p in paths:
        _log(log, f"   → {p.relative_to(ROOT)}")
    return paths


def run_pipeline(
    cfg: PipelineConfig,
    log: Optional[Callable[[str], None]] = print,
) -> dict:
    """Full demo: raw → QC → predict → simulate → summary → figures."""
    cfg.validate()
    OUT.mkdir(parents=True, exist_ok=True)
    generate_raw(cfg, log)
    run_qc(log)
    FEATURES_PRED, metrics = run_prediction(log)
    summary = run_simulations(cfg, log)
    figure_paths: list[Path] = []
    if cfg.figures:
        figure_paths = build_figures(log)

    result = {
        "config": {
            "seed": cfg.seed,
            "rain_noise": cfg.rain_noise,
            "rain_noise_dry": cfg.rain_noise_dry,
            "rain_sensor_noise": cfg.rain_sensor_noise,
            "vwc_sensor_noise": cfg.vwc_sensor_noise,
            "models": list(cfg.models),
            "policies": list(cfg.policies),
        },
        "metrics": metrics,
        "summary": summary.to_dict(orient="records") if not summary.empty else [],
        "figures": [str(p) for p in figure_paths],
    }
    RESULT_JSON.write_text(json.dumps(result, indent=2, default=str))

    if not summary.empty:
        _log(log, "\n=== Damage summary (sorted by crop-loss PROXY) ===")
        ranked = summary.sort_values("damage_proxy")
        for _, r in ranked.iterrows():
            _log(
                log,
                "  {: <12} {: <12} proxy={:.3f}  ponded={:>6.2f} h  peak={:>6.1f} mm".format(
                    r["model"], r["policy"], r["damage_proxy"], r["hours_ponded"], r["peak_pond_mm"]
                ),
            )
    _log(log, f"\nOutputs under: {OUT}")
    return result


# ---------------------------------------------------------------- GUI helpers
def build_category_stats() -> pd.DataFrame:
    """Categorised, count-sorted statistics for the GUI summary tab.

    Columns: ``Group | Value | Count | Share %``. Groups are ordered in pipeline
    order and each group's values are sorted by count (high → low).
    """
    rows: list[dict] = []

    def add_group(group: str, counts: dict) -> None:
        total = int(sum(int(v) for v in counts.values()))
        if total <= 0:
            return
        items = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))
        for value, n in items:
            rows.append(
                {
                    "group": group,
                    "value": str(value),
                    "count": int(n),
                    "share_pct": round(100.0 * n / total, 1),
                }
            )

    raw = _read_csv(RAW)
    if raw is not None and "quality_flag" in raw:
        for sensor, sub in raw.groupby("sensor_type"):
            add_group(f"Raw log · {sensor} · quality_flag", sub["quality_flag"].value_counts().to_dict())

    clean = _read_csv(CLEAN)
    if clean is not None and "quality_flag" in clean:
        add_group("Clean sensor · quality_flag", clean["quality_flag"].value_counts().to_dict())

    features = _read_csv(FEATURES)
    if features is not None and "quality_summary" in features:
        add_group("Features · quality_summary", features["quality_summary"].value_counts().to_dict())

    pred = _read_csv(FEATURES_PRED)
    if pred is not None:
        if "predicted_severity" in pred:
            add_group(
                "Predictions · predicted_severity",
                pred["predicted_severity"].value_counts().to_dict(),
            )
        if "phys_severity" in pred:
            add_group(
                "Predictions · physics severity",
                pred["phys_severity"].value_counts().to_dict(),
            )

    for path in sorted(OUT.glob("sim_*.csv")):
        parsed = parse_sim_name(path.name)
        if parsed is None:
            continue
        policy, model = parsed
        df = _read_csv(path)
        if df is None or "severity_label" not in df:
            continue
        add_group(
            f"Sim · {policy} / {model} · severity_label",
            df["severity_label"].value_counts().to_dict(),
        )

    for path in sorted(OUT.glob("pump_workflow_*.csv")):
        df = _read_csv(path)
        if df is None or "action" not in df:
            continue
        label = path.stem.replace("pump_workflow_", "")
        add_group(f"Pump workflow · {label} · action", df["action"].value_counts().to_dict())

    return pd.DataFrame(rows, columns=["group", "value", "count", "share_pct"])


def build_metrics_table() -> pd.DataFrame:
    """Model metrics as a long ``metric | value`` table for display."""
    metrics = _read_csv(METRICS)
    if metrics is None or metrics.empty:
        return pd.DataFrame(columns=["metric", "value"])
    row = metrics.iloc[0].to_dict()
    return pd.DataFrame(
        [{"metric": k, "value": "" if pd.isna(v) else str(v)} for k, v in row.items()]
    )


def data_file_inventory() -> pd.DataFrame:
    """Which expected artefacts exist, how big, and when they were written."""
    expected = [
        ("Raw sensor log", RAW, "input"),
        ("Clean sensor log", CLEAN, "input"),
        ("Features", FEATURES, "input"),
        ("Predictions", FEATURES_PRED, "output"),
        ("Damage summary", SUMMARY, "output"),
        ("Model metrics", METRICS, "output"),
        ("Simulation manifest", MANIFEST, "metadata"),
    ]
    expected += [
        (f"Simulation · {p.name}", p, "output")
        for p in sorted(OUT.glob("sim_*.csv"))
    ]
    expected += [
        (f"Pump workflow · {p.name}", p, "output")
        for p in sorted(OUT.glob("pump_workflow_*.csv"))
    ]
    expected += [
        (f"Figure · {p.name}", p, "figure")
        for p in sorted(OUT.glob("fig_*.png"))
    ]
    rows = []
    for label, path, kind in expected:
        row = {
            "file": label,
            "kind": kind,
            "status": "ready" if path.exists() else "missing",
            "rows": "",
            "size_kb": "",
            "modified": "",
            "path": str(path.relative_to(ROOT)),
        }
        if path.exists():
            frame = _read_csv(path) if path.suffix == ".csv" else None
            row.update(
                {
                    "rows": len(frame) if frame is not None else "",
                    "size_kb": round(path.stat().st_size / 1024, 1),
                    "modified": datetime.fromtimestamp(path.stat().st_mtime).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def expected_csv_paths() -> list[Path]:
    """Every CSV currently on disk (used by the "all CSVs" mode / file inventory).

    Inputs first, then the damage summary, then simulations (policy-major,
    bucket first) and finally the pump workflow logs.
    """
    preferred = [
        RAW,
        CLEAN,
        FEATURES,
        FEATURES_PRED,
        SUMMARY,
        METRICS,
    ]
    model_rank = {m: i for i, m in enumerate(HYDRO_MODELS)}

    def sim_key(path: Path) -> tuple[int, int]:
        parsed = parse_sim_name(path.name)
        if parsed is None:
            return (99, 99)
        policy, model = parsed
        return (
            POLICY_ORDER.index(policy) if policy in POLICY_ORDER else 99,
            model_rank.get(model, 99),
        )

    def wf_key(path: Path) -> tuple[int, int]:
        stem = path.stem.replace("pump_workflow_", "")
        policy, _, model = stem.partition("__")
        return (
            POLICY_ORDER.index(policy) if policy in POLICY_ORDER else 99,
            model_rank.get(model or DEFAULT_MODEL, 99),
        )

    sims = sorted(OUT.glob("sim_*.csv"), key=sim_key)
    workflows = sorted(OUT.glob("pump_workflow_*.csv"), key=wf_key)
    seen: list[Path] = []
    for path in preferred + sims + workflows:
        if path.exists() and path not in seen:
            seen.append(path)
    return seen


# ------------------------------------------------------- CSV inspector helpers
def write_simulation_manifest(
    models: tuple[str, ...] | list[str],
    policies: tuple[str, ...] | list[str],
) -> Path:
    """Record exactly which simulation / workflow CSVs the latest run wrote.

    The GUI uses this to avoid listing stale CSVs left behind by earlier runs
    with different models or policies.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for model in models:
        for policy in policies:
            files.extend((sim_path(policy, model), workflow_path(policy, model)))
    payload = {
        "models": list(models),
        "policies": list(policies),
        "csv_files": [str(p.relative_to(ROOT)) for p in files],
        "updated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "note": (
            "CSVs produced by the most recent simulation run. The GUI CSV "
            "inspector lists only these by default; uncheck 'Only simulated "
            "runs' or use the Files tab to see every artefact on disk."
        ),
    }
    MANIFEST.write_text(json.dumps(payload, indent=2))
    return MANIFEST


def simulation_manifest() -> dict:
    """Latest-run manifest, falling back to ``demo_result.json`` for CLI runs."""
    if MANIFEST.exists():
        try:
            data = json.loads(MANIFEST.read_text())
            if data.get("csv_files"):
                return data
        except (OSError, ValueError):
            pass
    if RESULT_JSON.exists():
        try:
            cfg = json.loads(RESULT_JSON.read_text()).get("config", {})
            return {
                "models": cfg.get("models", []),
                "policies": cfg.get("policies", []),
                "csv_files": [],
                "source": "demo_result.json",
            }
        except (OSError, ValueError):
            pass
    return {"models": [], "policies": [], "csv_files": [], "source": "none"}


def simulated_csv_paths(include_inputs: bool = False) -> list[Path]:
    """CSVs from the latest simulated run (+ derived result CSVs).

    Input CSVs (raw / clean / features) are excluded unless requested; stale
    simulation CSVs from older runs are always excluded.
    """
    manifest = simulation_manifest()
    result_csvs = [FEATURES_PRED, SUMMARY, METRICS]
    input_csvs = [RAW, CLEAN, FEATURES]
    candidates = (input_csvs if include_inputs else []) + result_csvs
    files = manifest.get("csv_files") or []
    if files:
        candidates += [ROOT / rel for rel in files]
    else:
        for model in manifest.get("models", []):
            for policy in manifest.get("policies", []):
                candidates.extend(
                    (sim_path(policy, model), workflow_path(policy, model))
                )
    seen: list[Path] = []
    for path in candidates:
        path = Path(path)
        if path.exists() and path not in seen:
            seen.append(path)
    return seen


def csv_metadata(path: Path) -> dict:
    """Categorise a CSV along the GUI's three dimensions.

    Returns type of data, model, treatment (= drainage policy), sort ranks and
    display fields. Shared CSVs (predictions, metrics, summary, inputs) have no
    single model/treatment and are labelled ``SHARED_CATEGORY``.
    """
    path = Path(path)
    name = path.name
    shared_types = {
        RAW: ("Raw sensor log", True),
        CLEAN: ("Clean sensor", True),
        FEATURES: ("Features", True),
        FEATURES_PRED: ("Predictions", False),
        METRICS: ("Model metrics", False),
        SUMMARY: ("Damage summary", False),
    }
    if path in shared_types:
        data_type, is_input = shared_types[path]
        policy, model_key = "", ""
        is_simulated = False
    else:
        parsed = parse_sim_name(name) or parse_workflow_name(name)
        if parsed is None:
            data_type, is_input, is_simulated = "Other CSV", False, False
            policy, model_key = "", ""
        else:
            policy, model_key = parsed
            data_type = "Simulation" if name.startswith("sim_") else "Pump workflow"
            is_input, is_simulated = False, True

    model_rank = (
        list(HYDRO_MODELS).index(model_key) if model_key in HYDRO_MODELS else -1
    )
    treatment_rank = (
        POLICY_ORDER.index(policy) if policy in POLICY_ORDER else -1
    )
    type_rank = (
        CSV_TYPE_ORDER.index(data_type)
        if data_type in CSV_TYPE_ORDER
        else len(CSV_TYPE_ORDER)
    )
    return {
        "path": path,
        "relative": str(path.relative_to(ROOT)) if path.is_absolute() else str(path),
        "filename": name,
        "type": data_type,
        "type_rank": type_rank,
        "model": model_key or SHARED_CATEGORY,
        "model_label": (
            HYDRO_MODELS[model_key].label if model_key in HYDRO_MODELS else SHARED_CATEGORY
        ),
        "model_key": model_key,
        "model_rank": model_rank,
        "treatment": POLICY_LABELS.get(policy, SHARED_CATEGORY),
        "treatment_key": policy,
        "treatment_rank": treatment_rank,
        "is_input": is_input,
        "is_simulated": is_simulated,
    }


def sort_csv_paths(paths: list[Path], by: str = CSV_SORT_CATEGORIES[0]) -> list[Path]:
    """Sort CSVs by one of the three categories (then the other two)."""
    def key(path: Path) -> tuple:
        m = csv_metadata(path)
        tail = (m["filename"],)
        if by == "Models":
            return (m["model_rank"], m["treatment_rank"], m["type_rank"], *tail)
        if by == "Treatments":
            return (m["treatment_rank"], m["model_rank"], m["type_rank"], *tail)
        return (m["type_rank"], m["model_rank"], m["treatment_rank"], *tail)

    return sorted(paths, key=key)
