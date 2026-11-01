"""Streamlit (or matplotlib-fallback) dashboard for rice waterlogging demo.

Run:  streamlit run app/dashboard.py
Or rely on run_demo.py which also saves figures under outputs/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def _load() -> dict[str, pd.DataFrame]:
    paths = {
        "features": ROOT / "data" / "processed" / "features.csv",
        "nothing": OUT / "sim_do_nothing.csv",
        "rule": OUT / "sim_rule_based.csv",
        "auto": OUT / "sim_automated.csv",
        "summary": OUT / "damage_summary.csv",
        "workflow": OUT / "pump_workflow_automated.csv",
        "metrics": OUT / "predict_metrics.csv",
    }
    data = {}
    for k, p in paths.items():
        if p.exists():
            data[k] = pd.read_csv(p)
    return data


def save_figures(data: dict[str, pd.DataFrame] | None = None) -> list[Path]:
    """Write judge-friendly matplotlib figures to outputs/."""
    data = data or _load()
    OUT.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    feat = data["features"]
    nothing = data["nothing"]
    auto = data["auto"]
    rule = data.get("rule")

    # shared time axis as hours from start
    t = np.arange(len(feat)) * (10 / 60.0)

    # --- Fig 1: rain + VWC ---
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(t, feat["rain_mm_h"], width=0.12, color="steelblue", alpha=0.7, label="Rain (mm/h)")
    ax1.set_ylabel("Rain intensity (mm/h)", color="steelblue")
    ax1.set_xlabel("Hours since start (Asia/Shanghai)")
    ax2 = ax1.twinx()
    ax2.plot(t, feat["vwc_m3m3"], color="darkgreen", lw=2, label="VWC")
    ax2.axhline(0.32, color="green", ls="--", lw=1, label="FC=0.32")
    ax2.set_ylabel("VWC (m³/m³)", color="darkgreen")
    ax2.set_ylim(0.1, 0.6)
    ax1.set_title("Sensor series: typhoon rainfall hyetograph + soil VWC")
    fig.tight_layout()
    p = OUT / "fig_rain_vwc.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    saved.append(p)

    # --- Fig 2: ponding + pump ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, nothing["pond_mm"], color="crimson", label="Do-nothing ponding")
    axes[0].plot(t, auto["pond_mm"], color="seagreen", label="Automated ponding")
    if rule is not None:
        axes[0].plot(t, rule["pond_mm"], color="darkorange", ls="--", alpha=0.8, label="Rule-based")
    axes[0].axhline(20, color="gray", ls=":", label="Harm threshold 20 mm")
    axes[0].set_ylabel("Ponding / ditch (mm)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title("Ponding depth: do-nothing vs automated drainage")

    axes[1].fill_between(t, 0, auto["pump_on"], step="pre", color="seagreen", alpha=0.5, label="Auto pump ON")
    if rule is not None:
        axes[1].fill_between(t, 0, rule["pump_on"] * 0.5, step="pre", color="darkorange", alpha=0.4, label="Rule pump ON (scaled)")
    axes[1].set_ylabel("Pump state")
    axes[1].set_xlabel("Hours since start")
    axes[1].set_ylim(-0.05, 1.2)
    axes[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    p = OUT / "fig_ponding_pump.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    saved.append(p)

    # --- Fig 3: severity ---
    fig, ax = plt.subplots(figsize=(10, 3.5))
    if "predicted_score" in auto.columns:
        ax.plot(t, auto["predicted_score"], color="purple", label="Predicted severity score")
    ax.plot(t, nothing["severity_score"], color="crimson", alpha=0.7, label="Physics score (do-nothing)")
    ax.set_ylabel("Severity score (0–1)")
    ax.set_xlabel("Hours since start")
    ax.set_title("Predicted / physics waterlogging severity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = OUT / "fig_severity.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    saved.append(p)

    # --- Fig 4: damage comparison bars ---
    summary = data["summary"].set_index("policy")
    metrics = ["hours_ponded", "peak_pond_mm", "damage_proxy"]
    labels = ["Ponded hours", "Peak depth (mm)", "Crop-loss PROXY"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    colors = {"do_nothing": "crimson", "rule_based": "darkorange", "automated": "seagreen"}
    for ax, m, lab in zip(axes, metrics, labels):
        vals = [float(summary.loc[pol, m]) for pol in summary.index]
        c = [colors.get(pol, "gray") for pol in summary.index]
        ax.bar(list(summary.index), vals, color=c)
        ax.set_title(lab)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Damage comparison: automation reduces ponding & crop-loss PROXY")
    fig.tight_layout()
    p = OUT / "fig_damage_compare.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    saved.append(p)

    # --- Fig 5: simple field damage heatmap (filled rect by ponding) ---
    # Represent the single ~10 ha field as a 10x10 grid whose fill intensity
    # tracks mean ponding in late-storm window — NOT a geographic China map.
    late = nothing["pond_mm"].iloc[int(18 * 6) : int(36 * 6)]
    late_auto = auto["pond_mm"].iloc[int(18 * 6) : int(36 * 6)]
    depth_n = float(late.mean()) if len(late) else 0.0
    depth_a = float(late_auto.mean()) if len(late_auto) else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, depth, title in [
        (axes[0], depth_n, f"Do-nothing\nmean pond≈{depth_n:.0f} mm"),
        (axes[1], depth_a, f"Automated\nmean pond≈{depth_a:.0f} mm"),
    ]:
        grid = np.full((10, 10), depth)
        # slight spatial variation for visual interest (same mean)
        rng = np.random.default_rng(1)
        grid = np.clip(grid + rng.normal(0, depth * 0.05 + 1e-6, grid.shape), 0, None)
        im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=max(80, depth_n, depth_a))
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Field P-RICE-A (~10 ha schematic)")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, label="Ponding depth (mm)")
    fig.suptitle("Field damage view (ponding depth fill — schematic, not a map)")
    p = OUT / "fig_field_damage.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    return saved


def main_streamlit() -> None:
    import streamlit as st

    st.set_page_config(page_title="Rice Waterlogging Drainage Demo", layout="wide")
    st.title("Rice waterlogging drainage — Jianghuai prototype")
    st.caption("Asia/Shanghai · ~10 ha · rainfall-only typhoon hyetograph · crop-loss PROXY (not RMB)")

    data = _load()
    if "nothing" not in data:
        st.warning("Outputs missing. Run `python run_demo.py` first.")
        return

    paths = save_figures(data)
    for p in paths:
        st.subheader(p.stem.replace("fig_", "").replace("_", " ").title())
        st.image(str(p), use_container_width=True)

    st.subheader("Damage summary")
    st.dataframe(data["summary"], use_container_width=True)

    if "metrics" in data:
        st.subheader("Severity model metrics")
        st.dataframe(data["metrics"], use_container_width=True)

    if "workflow" in data:
        st.subheader("Automated pump workflow (state changes)")
        wf = data["workflow"]
        st.dataframe(wf[wf["state_change"] == 1].head(40), use_container_width=True)


if __name__ == "__main__":
    try:
        import streamlit  # noqa: F401

        # When executed via `streamlit run`, Streamlit injects runtime; when
        # plain python, just save figures.
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx() is not None:
            main_streamlit()
        else:
            paths = save_figures()
            print("Saved figures:", [str(p) for p in paths])
    except Exception:
        paths = save_figures()
        print("Saved figures:", [str(p) for p in paths])
