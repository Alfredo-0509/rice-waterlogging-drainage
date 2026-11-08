"""sklearn severity prediction from recent sensor features.

Target = near-future (next 3–6 h) physics severity label from the simulator.
Split by time (first 70% train / last 30% test) — NOT random rows.
Always report a dummy baseline (always 'watch').
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

from src.hydrology import WaterBalanceSim, DT_H

LABELS = ["none", "watch", "waterlogging", "severe"]
HORIZON_SLOTS = 24  # 4 hours ahead (within 3–6 h band)
LOOKBACK = 18  # last 3 hours of features


def _future_severity_labels(features: pd.DataFrame) -> pd.DataFrame:
    """Run do-nothing physics once to label each time with future severity."""
    sim = WaterBalanceSim(initial_theta=float(features["vwc_m3m3"].iloc[0]))
    scores = []
    labels = []
    ponds = []
    thetas = []
    pond_flags: list[float] = []
    for _, row in features.iterrows():
        recent = sum(pond_flags[-144:]) * DT_H if pond_flags else 0.0
        res = sim.step(
            rain_mm=float(row["rain_mm_10min"]),
            pump_on=False,
            rain_6h_mm=float(row["rain_6h_mm"]),
            recent_pond_hours=recent,
        )
        pond_flags.append(1.0 if res.pond_mm > 20 else 0.0)
        scores.append(res.severity_score)
        labels.append(res.severity_label)
        ponds.append(res.pond_mm)
        thetas.append(res.theta)

    phys = features.copy()
    phys["phys_severity"] = labels
    phys["phys_score"] = scores
    phys["phys_pond_mm"] = ponds
    phys["phys_theta"] = thetas

    # target = max severity score in next HORIZON_SLOTS, mapped to label
    future_score = (
        pd.Series(scores)
        .iloc[::-1]
        .rolling(HORIZON_SLOTS, min_periods=1)
        .max()
        .iloc[::-1]
        .values
    )
    phys["target_score"] = future_score

    def score_to_label(s: float) -> str:
        if s < 0.15:
            return "none"
        if s < 0.35:
            return "watch"
        if s < 0.65:
            return "waterlogging"
        return "severe"

    phys["target_label"] = [score_to_label(s) for s in future_score]
    return phys


def _feature_matrix(phys: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build ML features from lookback aggregates."""
    df = phys.copy()
    df["vwc_slope_3h"] = df["vwc_m3m3"].diff(LOOKBACK) / 3.0
    df["rain_3h_mm"] = df["rain_mm_10min"].rolling(LOOKBACK, min_periods=1).sum()
    df["rain_intensity_max_1h"] = df["rain_mm_h"].rolling(6, min_periods=1).max()
    df["vwc_mean_3h"] = df["vwc_m3m3"].rolling(LOOKBACK, min_periods=1).mean()

    cols = [
        "rain_mm_h",
        "rain_1h_mm",
        "rain_6h_mm",
        "rain_3h_mm",
        "rain_intensity_max_1h",
        "vwc_m3m3",
        "vwc_mean_3h",
        "vwc_slope_3h",
        "hours_above_field_capacity",
        "phys_pond_mm",
        "phys_score",
    ]
    X = df[cols].fillna(0.0)
    # drop warm-up + last horizon (no full future)
    valid = np.ones(len(df), dtype=bool)
    valid[:LOOKBACK] = False
    valid[-HORIZON_SLOTS:] = False
    return X.loc[valid], df.loc[valid, "target_label"], df.loc[valid, "target_score"]


def train_and_predict(features: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Time-ordered split; return features with predictions + metrics dict."""
    phys = _future_severity_labels(features)
    X, y_lab, y_score = _feature_matrix(phys)

    # Time-ordered split (NOT random rows): first 65% train, last 35% test.
    # Covers late band-1 / lull / main burst / early recession depending on length.
    n = len(X)
    split = max(40, min(int(n * 0.65), n - 30))
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y_lab.iloc[:split], y_lab.iloc[split:]
    ys_tr, ys_te = y_score.iloc[:split], y_score.iloc[split:]

    clf = GradientBoostingClassifier(random_state=0, max_depth=3, n_estimators=80)
    # If training set missing a class, sklearn still works with present classes
    present = sorted(set(y_tr.tolist()))
    if len(present) < 2:
        # degenerate: fall back to physics scores only
        preds_all = phys["phys_severity"].tolist()
        scores_all = phys["phys_score"].tolist()
        metrics = {
            "note": "thin_class_diversity_physics_only",
            "accuracy": None,
            "mae": None,
            "dummy_accuracy": None,
            "n_train": int(split),
            "n_test": int(n - split),
        }
    else:
        clf.fit(X_tr, y_tr)
        reg = GradientBoostingRegressor(random_state=0, max_depth=3, n_estimators=80)
        reg.fit(X_tr, ys_tr)

        pred_te = clf.predict(X_te)
        score_te = reg.predict(X_te)
        acc = accuracy_score(y_te, pred_te)
        mae = mean_absolute_error(ys_te, score_te)
        dummy_pred = np.array(["watch"] * len(y_te))
        dummy_acc = accuracy_score(y_te, dummy_pred)

        # predict on full valid index, then align to all rows
        pred_valid = clf.predict(X)
        score_valid = reg.predict(X)
        preds_all = phys["phys_severity"].tolist()
        scores_all = phys["phys_score"].astype(float).tolist()
        # overwrite valid region with ML preds
        valid_idx = X.index.tolist()
        for idx, p, s in zip(valid_idx, pred_valid, score_valid):
            preds_all[idx] = p
            scores_all[idx] = float(np.clip(s, 0.0, 1.0))

        # for early warm-up / tail keep physics; for train region use in-sample preds
        # Secondary check on the last 40 train rows (mixed classes near lull)
        early_te_slice = slice(max(0, split - 40), split)
        if early_te_slice.stop > early_te_slice.start:
            y_early = y_tr.iloc[early_te_slice]
            # refit-free: predict with model trained on data before this slice
            X_early_tr, y_early_tr = X_tr.iloc[: early_te_slice.start], y_tr.iloc[: early_te_slice.start]
            if len(set(y_early_tr)) >= 2 and len(y_early) >= 10:
                clf_e = GradientBoostingClassifier(random_state=0, max_depth=3, n_estimators=80)
                clf_e.fit(X_early_tr, y_early_tr)
                pred_e = clf_e.predict(X_tr.iloc[early_te_slice])
                early_acc = accuracy_score(y_early, pred_e)
                early_dummy = accuracy_score(y_early, ["watch"] * len(y_early))
            else:
                early_acc, early_dummy = None, None
        else:
            early_acc, early_dummy = None, None

        metrics = {
            "accuracy": round(float(acc), 3),
            "mae": round(float(mae), 3),
            "dummy_accuracy": round(float(dummy_acc), 3),
            "early_holdout_accuracy": None if early_acc is None else round(float(early_acc), 3),
            "early_holdout_dummy_accuracy": None if early_dummy is None else round(float(early_dummy), 3),
            "n_train": int(split),
            "n_test": int(n - split),
            "classes_train": present,
            "test_label_counts": y_te.value_counts().to_dict(),
            "note": "late test window is often all-severe under do-nothing ponding; see early_holdout_* and MAE",
        }

    out = features.copy()
    out["predicted_severity"] = preds_all
    out["predicted_score"] = scores_all
    out["phys_severity"] = phys["phys_severity"].values
    out["phys_score"] = phys["phys_score"].values
    out["target_label"] = phys["target_label"].values
    out["target_score"] = phys["target_score"].values
    return out, metrics
