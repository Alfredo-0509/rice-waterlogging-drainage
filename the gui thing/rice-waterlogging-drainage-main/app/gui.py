#!/usr/bin/env python3
"""Tkinter control room for the rice-waterlogging drainage demo.

Run:
    python3 app/gui.py            # opens the window
    python3 app/gui.py --check    # verifies imports + lists artefacts (no window)

The window gives one button per pipeline stage:

    ① Generate raw CSV   (scripts/generate_raw_csv.py — configurable noise, no fixed seed)
    ② QC + features      (scripts/ingest_qc.py)
    ③ Severity model     (src/predict.py)
    ④ Simulations        (src/hydrology.py — choose the simulation model + policies)
    ⑤ Summary + charts   (outputs/damage_summary.csv + PNGs)

A single CSV inspector tab uses a dropdown instead of one tab per file. CSVs
are grouped / sorted along three categories — type of data, model and treatment
(drainage policy) — and by default only CSVs from the latest simulated run are
listed, so results from older runs do not flood the selector. Each selected CSV
gets a searchable / filterable / sortable table, category counts (sorted high →
low) and a numeric summary. The Overview tab aggregates the damage summary,
model metrics and categorised statistics.

Tkinter ships with CPython on Windows/macOS; on Debian/Ubuntu install
``python3-tk``. This module imports pandas and the pipeline at import time but
only touches Tk inside ``main()`` / class methods, so ``--check`` still works on
a machine without Tk.
"""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src import pipeline
from src.hydrology import DEFAULT_MODEL, HYDRO_MODELS

try:  # pragma: no cover - depends on host OS packages
    import tkinter as tk
    from tkinter import messagebox, ttk

    TK_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover
    tk = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]
    messagebox = None  # type: ignore[assignment]
    TK_IMPORT_ERROR = exc


APP_TITLE = "Rice Waterlogging Control Room"
BG = "#f2f6f2"
ACCENT = "#14532d"
ROW_LIMIT = 4000
ALL_COLUMNS = "(any column)"

MODEL_BY_LABEL = {m.label: m.key for m in HYDRO_MODELS.values()}

ALL_FILTER = "(all)"
CSV_GROUP_OPTIONS = pipeline.CSV_SORT_CATEGORIES

PREFERRED_CATEGORY_COLUMNS = (
    "quality_flag",
    "quality_summary",
    "predicted_severity",
    "phys_severity",
    "severity_label",
    "action",
    "policy",
    "sensor_type",
    "model",
    "state_change",
    "status",
)


# --------------------------------------------------------------------- helpers
def _display_label(meta: dict, by: str) -> str:
    """One dropdown entry; always shows all three categories."""
    type_txt = meta["type"]
    model_txt = meta["model"]
    treatment_txt = meta["treatment"]
    name = meta["filename"]
    if by == "Models":
        if meta["model_key"]:
            return f"{model_txt} ▸ {treatment_txt} · {type_txt} — {name}"
        return f"Shared ▸ {type_txt} — {name}"
    if by == "Treatments":
        if meta["treatment_key"]:
            return f"{treatment_txt} ▸ {model_txt} · {type_txt} — {name}"
        return f"Shared ▸ {type_txt} — {name}"
    if meta["treatment_key"]:
        return f"{type_txt} · {model_txt} · {treatment_txt} — {name}"
    return f"{type_txt} (shared) — {name}"


def _format_cell(value) -> str:
    try:
        if value is None or pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _column_width(df: pd.DataFrame, column: str, sample: int = 200) -> int:
    try:
        lengths = df[column].head(sample).astype(str).str.len()
        content = int(lengths.max()) if len(lengths) else 0
    except Exception:
        content = 0
    return max(70, min(240, max(len(str(column)) + 2, content) * 7 + 14))


def _open_path(path: Path) -> None:
    try:
        if not path.exists():
            raise FileNotFoundError(str(path))
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as exc:  # pragma: no cover - desktop dependent
        if messagebox is not None:
            messagebox.showwarning("Could not open", f"{path}\n\n{exc}")


def _load_photo(path: Path, max_w: int, max_h: int):
    """PNG → Tk image, using Pillow when available and Tk subsampling otherwise."""
    try:
        from PIL import Image, ImageTk

        img = Image.open(str(path))
        img.thumbnail((max_w, max_h))
        return ImageTk.PhotoImage(img)
    except Exception:
        photo = tk.PhotoImage(file=str(path))
        factor = max(
            1,
            -(-photo.width() // max_w),
            -(-photo.height() // max_h),
        )
        return photo.subsample(factor, factor) if factor > 1 else photo


# ------------------------------------------------------------------ scrollable
class ScrollableFrame:
    """Vertical scrolling container (left control panel)."""

    def __init__(self, parent, width: int = 400):
        self.frame = ttk.Frame(parent)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(
            self.frame, borderwidth=0, highlightthickness=0, background=BG, width=width
        )
        self.scroll = ttk.Scrollbar(
            self.frame, orient="vertical", command=self.canvas.yview
        )
        self.canvas.configure(yscrollcommand=self.scroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scroll.grid(row=0, column=1, sticky="ns")
        self.inner = ttk.Frame(self.canvas)
        self._window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._on_inner)
        self.canvas.bind("<Configure>", self._on_canvas)
        # Scroll only while the pointer is over this panel (children included).
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.canvas.bind_all(sequence, self._on_wheel, add="+")

    def _on_inner(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas(self, event):
        self.canvas.itemconfigure(self._window, width=event.width)

    def _on_wheel(self, event):
        try:
            widget = self.canvas.winfo_containing(event.x_root, event.y_root)
        except Exception:
            return None
        while widget is not None:
            if widget is self.inner or widget is self.canvas:
                break
            widget = getattr(widget, "master", None)
        else:
            return None
        if getattr(event, "num", None) == 4:
            step = -1
        elif getattr(event, "num", None) == 5:
            step = 1
        else:
            step = -1 if getattr(event, "delta", 0) > 0 else 1
        self.canvas.yview_scroll(step, "units")
        return None


# --------------------------------------------------------------------- tables
class TreeTable:
    """Sortable Treeview over an in-memory DataFrame (click headers to sort)."""

    def __init__(self, parent, height: int = 8, max_rows: int = ROW_LIMIT):
        self.frame = ttk.Frame(parent)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(
            self.frame, show="headings", height=height, selectmode="browse"
        )
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb = ttk.Scrollbar(self.frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(self.frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        self.tree.tag_configure("odd", background="#f6faf6")
        self.tree.tag_configure("even", background="#ffffff")

        self.df = pd.DataFrame()
        self.view = pd.DataFrame()
        self.sort_col: str | None = None
        self.desc = False
        self.columns: list[str] = []
        self.max_rows = max_rows
        self._numeric: set[str] = set()
        self.displayed = 0

    # -- public ---------------------------------------------------------
    def set_data(
        self,
        df: pd.DataFrame,
        sort_col: str | None = None,
        desc: bool = False,
        reset_sort: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True) if df is not None else pd.DataFrame()
        self._numeric = {
            c for c in self.df.columns if pd.api.types.is_numeric_dtype(self.df[c])
        }
        if sort_col is not None:
            self.sort_col, self.desc = sort_col, bool(desc)
        elif reset_sort or self.sort_col not in self.df.columns:
            self.sort_col = self.df.columns[0] if len(self.df.columns) else None
            self.desc = False
        self._render()

    # -- internals ------------------------------------------------------
    def _on_heading(self, column: str) -> None:
        if self.sort_col == column:
            self.desc = not self.desc
        else:
            self.sort_col = column
            self.desc = column in self._numeric
        self._render()

    def _render(self) -> None:
        columns = list(self.df.columns)
        if columns != self.columns:
            self.columns = columns
            self.tree.delete(*self.tree.get_children())
            self.tree["columns"] = columns
            for col in columns:
                self.tree.heading(col, text=str(col), command=lambda c=col: self._on_heading(c))
                self.tree.column(
                    col,
                    width=_column_width(self.df, col),
                    minwidth=60,
                    anchor="e" if col in self._numeric else "w",
                    stretch=True,
                )
        for col in columns:
            arrow = ""
            if col == self.sort_col:
                arrow = " ▼" if self.desc else " ▲"
            self.tree.heading(col, text=f"{col}{arrow}")

        view = self.df
        if self.sort_col is not None and len(view):
            key = (
                pd.to_numeric(view[self.sort_col], errors="coerce")
                if self.sort_col in self._numeric
                else view[self.sort_col].astype(str).str.lower()
            )
            view = (
                view.assign(__sort_key=key)
                .sort_values(
                    "__sort_key",
                    ascending=not self.desc,
                    na_position="last",
                    kind="mergesort",
                )
                .drop(columns="__sort_key")
            )
        self.view = view

        self.tree.delete(*self.tree.get_children())
        shown = min(len(view), self.max_rows)
        for i in range(shown):
            row = view.iloc[i]
            values = [_format_cell(row[c]) for c in self.columns]
            self.tree.insert("", "end", values=values, tags=("odd" if i % 2 else "even",))
        self.displayed = shown


# ----------------------------------------------------------------- CSV tab
class CsvTable:
    """One CSV = one tab: filter toolbar + sortable table + category statistics."""

    def __init__(self, parent, path: Path):
        self.path = path
        self.df = pd.DataFrame()
        self._mtime: float | None = None
        self.frame = ttk.Frame(parent, padding=(8, 6))
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(2, weight=1)
        self._build()
        self.reload()

    # -- construction ---------------------------------------------------
    def set_path(self, path: Path) -> None:
        """Point this inspector at another CSV, resetting its own filters."""
        self.path = Path(path)
        self._mtime = None
        self.search_var.set("")
        self.reload(force=True)

    def _build(self) -> None:
        bar = ttk.Frame(self.frame)
        bar.grid(row=0, column=0, sticky="ew")

        ttk.Label(bar, text="Search all columns:").pack(side="left")
        self.search_var = tk.StringVar()
        ttk.Entry(bar, textvariable=self.search_var, width=22).pack(
            side="left", padx=(4, 10)
        )
        self.search_var.trace_add("write", lambda *_: self._apply_filters())

        ttk.Label(bar, text="Column:").pack(side="left")
        self.filter_col_var = tk.StringVar(value=ALL_COLUMNS)
        self.filter_col = ttk.Combobox(
            bar,
            textvariable=self.filter_col_var,
            state="readonly",
            width=16,
            values=[ALL_COLUMNS],
        )
        self.filter_col.pack(side="left", padx=4)
        self.filter_col.bind("<<ComboboxSelected>>", self._on_filter_column)

        ttk.Label(bar, text="value:").pack(side="left")
        self.filter_val_var = tk.StringVar()
        self.filter_val = ttk.Combobox(bar, textvariable=self.filter_val_var, width=14)
        self.filter_val.pack(side="left", padx=4)
        self.filter_val.bind("<<ComboboxSelected>>", lambda _e: self._apply_filters())
        self.filter_val.bind("<Return>", lambda _e: self._apply_filters())

        ttk.Button(bar, text="Apply", command=self._apply_filters).pack(
            side="left", padx=(4, 0)
        )
        ttk.Button(bar, text="Clear", command=self._clear_filters).pack(
            side="left", padx=4
        )
        ttk.Button(bar, text="Reload", command=lambda: self.reload(force=True)).pack(
            side="right"
        )
        ttk.Button(
            bar, text="Open file", command=lambda: _open_path(self.path)
        ).pack(side="right", padx=4)
        ttk.Button(bar, text="Copy path", command=self._copy_path).pack(
            side="right", padx=4
        )

        self.info_var = tk.StringVar(value="")
        ttk.Label(self.frame, textvariable=self.info_var, style="Hint.TLabel").grid(
            row=1, column=0, sticky="w", pady=(4, 4)
        )

        self.table = TreeTable(self.frame, height=14)
        self.table.frame.grid(row=2, column=0, sticky="nsew")

        stats = ttk.LabelFrame(
            self.frame,
            text="Categorised statistics (sorted by count) + numeric summary",
            padding=6,
        )
        stats.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        stats.columnconfigure(1, weight=1)
        stats.columnconfigure(3, weight=1)
        stats.rowconfigure(1, weight=1)

        ttk.Label(stats, text="Category column:").grid(row=0, column=0, sticky="w")
        self.cat_var = tk.StringVar()
        self.cat_combo = ttk.Combobox(
            stats, textvariable=self.cat_var, state="readonly", width=22
        )
        self.cat_combo.grid(row=0, column=1, sticky="w", padx=4)
        self.cat_combo.bind("<<ComboboxSelected>>", lambda _e: self._render_stats())

        ttk.Label(stats, text="Numeric column:").grid(
            row=0, column=2, sticky="e", padx=(16, 4)
        )
        self.num_var = tk.StringVar()
        self.num_combo = ttk.Combobox(
            stats, textvariable=self.num_var, state="readonly", width=22
        )
        self.num_combo.grid(row=0, column=3, sticky="w")
        self.num_combo.bind("<<ComboboxSelected>>", lambda _e: self._render_stats())

        self.cat_table = TreeTable(stats, height=5, max_rows=40)
        self.cat_table.frame.grid(
            row=1, column=0, columnspan=2, sticky="nsew", pady=(6, 0)
        )

        self.num_text = tk.Text(
            stats,
            height=5,
            width=40,
            wrap="word",
            relief="flat",
            background="#ffffff",
            font=("DejaVu Sans Mono", 9),
        )
        self.num_text.grid(
            row=1, column=2, columnspan=2, sticky="nsew", padx=(16, 0), pady=(6, 0)
        )
        self.num_text.configure(state="disabled")

    # -- data -----------------------------------------------------------
    def reload(self, force: bool = False) -> None:
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            mtime = None
        if not force and mtime is not None and mtime == self._mtime:
            return
        self._mtime = mtime
        error = ""
        try:
            self.df = pd.read_csv(self.path)
        except Exception as exc:  # unreadable CSV shouldn't kill the app
            self.df = pd.DataFrame()
            error = str(exc)

        columns = list(self.df.columns)
        self.filter_col["values"] = [ALL_COLUMNS] + columns
        self.filter_col_var.set(ALL_COLUMNS)
        self.filter_val_var.set("")
        self.filter_val["values"] = []

        categories = []
        for col in columns:
            try:
                nunique = self.df[col].nunique(dropna=False)
            except Exception:
                continue
            if 0 < nunique <= 40:
                categories.append(col)
        self.cat_combo["values"] = categories
        if categories:
            preferred = next(
                (c for c in PREFERRED_CATEGORY_COLUMNS if c in categories), categories[0]
            )
            self.cat_var.set(preferred)
        else:
            self.cat_var.set("")

        numeric = [c for c in columns if pd.api.types.is_numeric_dtype(self.df[c])]
        self.num_combo["values"] = numeric
        if numeric:
            preferred_num = next(
                (
                    c
                    for c in ("damage_proxy", "pond_mm", "severity_score", "value_si")
                    if c in numeric
                ),
                numeric[0],
            )
            self.num_var.set(preferred_num)
        else:
            self.num_var.set("")

        self._apply_filters(reset_sort=True)
        if error:
            self.info_var.set(f"Could not read {self.path.name}: {error}")

    def _clear_filters(self) -> None:
        self.search_var.set("")
        self.filter_col_var.set(ALL_COLUMNS)
        self.filter_val_var.set("")
        self.filter_val["values"] = []
        self._apply_filters(reset_sort=True)

    def _on_filter_column(self, _event=None) -> None:
        col = self.filter_col_var.get()
        if col == ALL_COLUMNS or col not in self.df.columns:
            self.filter_val["values"] = []
            return
        try:
            values = sorted(self.df[col].dropna().astype(str).unique())[:300]
        except Exception:
            values = []
        self.filter_val["values"] = values
        if values:
            self.filter_val_var.set(values[0])
        self._apply_filters()

    def _apply_filters(self, reset_sort: bool = False) -> None:
        if self.df.empty:
            self.table.set_data(
                pd.DataFrame(columns=self.df.columns), reset_sort=True
            )
            self.info_var.set(
                f"{self.path.name} · no rows"
                if not self.df.columns
                else f"{self.path.name} · 0 rows × {len(self.df.columns)} columns"
            )
            self._render_stats()
            return

        mask = pd.Series(True, index=self.df.index)
        search = self.search_var.get().strip().lower()
        if search:
            text = self.df.astype(str)
            matches = text.apply(
                lambda s: s.str.lower().str.contains(search, regex=False)
            )
            mask &= matches.any(axis=1)

        col = self.filter_col_var.get()
        value = self.filter_val_var.get().strip().lower()
        if col != ALL_COLUMNS and value and col in self.df.columns:
            mask &= (
                self.df[col].astype(str).str.lower().str.contains(value, regex=False)
            )

        view = self.df[mask]
        self.table.set_data(view, reset_sort=reset_sort)

        shown = self.table.displayed
        sort_txt = ""
        if self.table.sort_col:
            sort_txt = (
                f" · sorted by {self.table.sort_col} "
                f"{'▼' if self.table.desc else '▲'}"
            )
        capped = f" · showing first {shown:,}" if shown < len(view) else ""
        self.info_var.set(
            f"{self.path.name} · {len(self.df):,} rows × {len(self.df.columns)} "
            f"columns · matching {len(view):,}{capped}{sort_txt}"
        )
        self._render_stats()

    def _render_stats(self) -> None:
        view = self.table.view
        cat = self.cat_var.get()
        if cat and cat in view.columns and len(view):
            counts = view[cat].astype(str).value_counts()
            total = int(counts.sum())
            df = pd.DataFrame(
                {
                    "value": list(counts.index[:40]),
                    "count": [int(v) for v in counts.values[:40]],
                }
            )
            df["share_pct"] = (100.0 * df["count"] / max(total, 1)).round(1)
        else:
            df = pd.DataFrame(columns=["value", "count", "share_pct"])
        self.cat_table.set_data(df, sort_col="count", desc=True, reset_sort=True)

        num_col = self.num_var.get()
        if num_col and num_col in view.columns:
            series = pd.to_numeric(view[num_col], errors="coerce").dropna()
            if len(series):
                text = (
                    f"{num_col} (filtered rows)\n"
                    f"  count  {len(series):>12,}\n"
                    f"  min    {series.min():>12.4g}\n"
                    f"  mean   {series.mean():>12.4g}\n"
                    f"  median {series.median():>12.4g}\n"
                    f"  max    {series.max():>12.4g}\n"
                    f"  std    {series.std():>12.4g}"
                )
            else:
                text = f"{num_col}: no numeric values in the filtered rows"
        else:
            text = (
                "Select a numeric column to see count / min / mean / median / "
                "max of the filtered rows."
            )
        self.num_text.configure(state="normal")
        self.num_text.delete("1.0", "end")
        self.num_text.insert("1.0", text)
        self.num_text.configure(state="disabled")

    def _copy_path(self) -> None:
        try:
            self.frame.clipboard_clear()
            self.frame.clipboard_append(str(self.path))
            self.info_var.set(f"Copied path: {self.path}")
        except Exception:
            pass


# ------------------------------------------------------------- CSV inspector
class CsvBrowserTab:
    """Dropdown-driven CSV inspector (replaces one tab per CSV).

    Three category dimensions — type of data, model and treatment (drainage
    policy) — act as filters and as the grouping/sort order. By default only
    CSVs from the latest simulated run are listed.
    """

    def __init__(self, parent):
        self.frame = ttk.Frame(parent, padding=10)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(4, weight=1)

        self.sort_var = tk.StringVar(value=CSV_GROUP_OPTIONS[0])
        self.type_var = tk.StringVar(value=ALL_FILTER)
        self.model_var = tk.StringVar(value=ALL_FILTER)
        self.treatment_var = tk.StringVar(value=ALL_FILTER)
        self.only_simulated_var = tk.BooleanVar(value=True)
        self.csv_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="")
        self._labels: dict[str, Path] = {}
        self._meta: dict[Path, dict] = {}
        self._current: Path | None = None

        ttk.Label(
            self.frame,
            text=(
                "Pick one CSV from the dropdown. Group / sort it by type of "
                "data, model or treatment (= drainage policy), and filter each "
                "category. Only CSVs produced by the latest simulation run are "
                "listed; use the Files tab (or uncheck 'Only simulated runs') "
                "to see everything on disk."
            ),
            style="Hint.TLabel",
            wraplength=1100,
            justify="left",
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        filters = ttk.Frame(self.frame)
        filters.grid(row=1, column=0, sticky="ew")

        def combo(label: str, var, values, width: int):
            ttk.Label(filters, text=label).pack(side="left", padx=(0, 3))
            cb = ttk.Combobox(
                filters, textvariable=var, state="readonly", width=width, values=values
            )
            cb.pack(side="left", padx=(0, 10))
            cb.bind("<<ComboboxSelected>>", lambda _e: self._refresh_options())
            return cb

        self.sort_combo = combo(
            "Group / sort by:", self.sort_var, list(CSV_GROUP_OPTIONS), 16
        )
        self.type_combo = combo("Type of data:", self.type_var, [ALL_FILTER], 18)
        self.model_combo = combo("Model:", self.model_var, [ALL_FILTER], 16)
        self.treatment_combo = combo(
            "Treatment:", self.treatment_var, [ALL_FILTER], 18
        )
        ttk.Checkbutton(
            filters,
            text="Only simulated runs",
            variable=self.only_simulated_var,
            command=self.refresh,
        ).pack(side="left", padx=(0, 10))
        ttk.Button(filters, text="Refresh", command=self.refresh).pack(side="left")

        selection = ttk.Frame(self.frame)
        selection.grid(row=2, column=0, sticky="ew", pady=(10, 2))
        ttk.Label(selection, text="CSV:").pack(side="left")
        self.csv_combo = ttk.Combobox(
            selection, textvariable=self.csv_var, state="readonly", width=88
        )
        self.csv_combo.pack(side="left", padx=(6, 0), fill="x", expand=True)
        self.csv_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_select())

        ttk.Label(
            self.frame, textvariable=self.status_var, style="Hint.TLabel"
        ).grid(row=3, column=0, sticky="w", pady=(4, 6))

        self.body = ttk.Frame(self.frame)
        self.body.grid(row=4, column=0, sticky="nsew")
        self.body.columnconfigure(0, weight=1)
        self.body.rowconfigure(1, weight=1)
        self.empty_label = ttk.Label(
            self.body, text="", style="Hint.TLabel", wraplength=1000, justify="left"
        )
        self.empty_label.grid(row=0, column=0, sticky="w")
        self.csv_table = CsvTable(self.body, pipeline.SUMMARY)
        self.csv_table.frame.grid(row=1, column=0, sticky="nsew")

    # -- data -----------------------------------------------------------
    def refresh(self) -> None:
        only = bool(self.only_simulated_var.get())
        paths = (
            pipeline.simulated_csv_paths()
            if only
            else pipeline.expected_csv_paths()
        )
        self._meta = {p: pipeline.csv_metadata(p) for p in paths}
        for var, widget, field in (
            (self.type_var, self.type_combo, "type"),
            (self.model_var, self.model_combo, "model"),
            (self.treatment_var, self.treatment_combo, "treatment"),
        ):
            values = [ALL_FILTER] + sorted(
                {str(md[field]) for md in self._meta.values()}
            )
            widget["values"] = values
            if var.get() not in values:
                var.set(ALL_FILTER)
        self._refresh_options()

    def _refresh_options(self) -> None:
        by = self.sort_var.get()
        if by not in CSV_GROUP_OPTIONS:
            by = CSV_GROUP_OPTIONS[0]
            self.sort_var.set(by)
        type_sel = self.type_var.get()
        model_sel = self.model_var.get()
        treatment_sel = self.treatment_var.get()
        matching = [
            path
            for path, md in self._meta.items()
            if (type_sel == ALL_FILTER or md["type"] == type_sel)
            and (model_sel == ALL_FILTER or md["model"] == model_sel)
            and (treatment_sel == ALL_FILTER or md["treatment"] == treatment_sel)
        ]
        paths = pipeline.sort_csv_paths(matching, by=by)

        self._labels = {}
        labels = []
        for path in paths:
            label = _display_label(self._meta[path], by)
            labels.append(label)
            self._labels[label] = path
        self.csv_combo["values"] = labels
        if self._current in paths:
            selected = next(
                (label for label, p in self._labels.items() if p == self._current),
                "",
            )
        else:
            selected = labels[0] if labels else ""
        self.csv_var.set(selected)
        self._on_select()

        types = {md["type"] for md in self._meta.values()}
        models = {
            md["model"] for md in self._meta.values()
            if md["model"] != pipeline.SHARED_CATEGORY
        }
        treatments = {
            md["treatment"] for md in self._meta.values()
            if md["treatment"] != pipeline.SHARED_CATEGORY
        }
        manifest = pipeline.simulation_manifest()
        scope = (
            "latest run: {} models × {} treatments".format(
                len(manifest.get("models", [])), len(manifest.get("policies", []))
            )
            if self.only_simulated_var.get()
            else "all CSVs on disk"
        )
        self.status_var.set(
            f"{len(paths)} of {len(self._meta)} CSVs shown · {scope} · "
            f"grouped by {by} · {len(types)} data types / {len(models)} models / "
            f"{len(treatments)} treatments"
        )

    def _on_select(self, _event=None) -> None:
        path = self._labels.get(self.csv_var.get())
        if path is None:
            self._current = None
            self.csv_table.frame.grid_forget()
            self.empty_label.configure(
                text=(
                    "No CSVs match. Run ④ Run selected simulations (or the full "
                    "pipeline), relax the three category filters, or uncheck "
                    "'Only simulated runs'."
                )
            )
            return
        self._current = path
        self.empty_label.configure(text="")
        self.csv_table.frame.grid(row=1, column=0, sticky="nsew")
        self.csv_table.set_path(path)


# ----------------------------------------------------------------- overview
class OverviewTab:
    """Aggregated damage summary, model metrics and categorised statistics."""

    def __init__(self, parent):
        self.frame = ttk.Frame(parent, padding=10)
        self.frame.columnconfigure(0, weight=3)
        self.frame.columnconfigure(1, weight=2)
        self.frame.rowconfigure(1, weight=3)
        self.frame.rowconfigure(2, weight=4)

        ttk.Label(
            self.frame,
            text=(
                "Left panel: ① generate raw CSV → ② QC → ③ predict → ④ simulate "
                "(pick a hydrology model) → ⑤ charts. The CSV inspector tab "
                "selects one result CSV at a time from a dropdown grouped by "
                "type of data / model / treatment; click a column header to sort."
            ),
            style="Hint.TLabel",
            wraplength=1000,
            justify="left",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        damage_card = ttk.LabelFrame(
            self.frame,
            text="Damage summary — sorted by crop-loss PROXY (best first)",
            padding=6,
        )
        damage_card.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        damage_card.columnconfigure(0, weight=1)
        damage_card.rowconfigure(0, weight=1)
        self.damage = TreeTable(damage_card, height=8)
        self.damage.frame.grid(row=0, column=0, sticky="nsew")
        self.insight = ttk.Label(
            damage_card, text="", style="Hint.TLabel", wraplength=560, justify="left"
        )
        self.insight.grid(row=1, column=0, sticky="w", pady=(6, 0))

        metrics_card = ttk.LabelFrame(
            self.frame, text="Severity model metrics", padding=6
        )
        metrics_card.grid(row=1, column=1, sticky="nsew", padx=(6, 0))
        metrics_card.columnconfigure(0, weight=1)
        metrics_card.rowconfigure(0, weight=1)
        self.metrics = TreeTable(metrics_card, height=8)
        self.metrics.frame.grid(row=0, column=0, sticky="nsew")

        cat_card = ttk.LabelFrame(
            self.frame,
            text="Categorised statistics — counts sorted high → low",
            padding=6,
        )
        cat_card.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        cat_card.columnconfigure(0, weight=1)
        cat_card.rowconfigure(0, weight=1)
        self.cats = TreeTable(cat_card, height=10)
        self.cats.frame.grid(row=0, column=0, sticky="nsew")

    def refresh(self) -> None:
        summary = pipeline.SUMMARY
        if summary.exists():
            try:
                df = pd.read_csv(summary).sort_values("damage_proxy")
            except Exception:
                df = pd.DataFrame()
        else:
            df = pd.DataFrame()

        if df.empty:
            self.damage.set_data(
                pd.DataFrame(columns=["model", "policy", "damage_proxy"]),
                reset_sort=True,
            )
            self.insight.configure(
                text="No simulations yet — choose a model in ④ and press "
                "'Run selected simulations'."
            )
        else:
            self.damage.set_data(df, sort_col="damage_proxy", desc=False, reset_sort=True)
            self.insight.configure(text=self._insight_text(df))

        metrics = pipeline.build_metrics_table()
        self.metrics.set_data(
            metrics, sort_col="metric", desc=False, reset_sort=True
        )

        cats = pipeline.build_category_stats()
        if cats.empty:
            cats = pd.DataFrame(columns=["group", "value", "count", "share_pct"])
        self.cats.set_data(cats, sort_col="count", desc=True, reset_sort=True)

    @staticmethod
    def _insight_text(df: pd.DataFrame) -> str:
        best = df.sort_values("damage_proxy").iloc[0]
        text = (
            f"Best: {best.get('model_label', best['model'])} / "
            f"{best.get('policy_label', best['policy'])} → "
            f"PROXY {float(best['damage_proxy']):.3f}"
        )
        baseline = df[df["policy"] == "do_nothing"] if "policy" in df else df.iloc[0:0]
        if not baseline.empty:
            dn = baseline[baseline["model"] == best["model"]]
            ref = float((dn if not dn.empty else baseline).damage_proxy.iloc[0])
            if ref > 0:
                text += (
                    f" · {100.0 * (ref - float(best['damage_proxy'])) / ref:.0f}% lower "
                    f"crop-loss PROXY than do-nothing ({ref:.3f})"
                )
        return text


# --------------------------------------------------------------------- files
class FilesTab:
    def __init__(self, parent):
        self.frame = ttk.Frame(parent, padding=10)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(1, weight=1)
        ttk.Label(
            self.frame,
            text=(
                "Artefacts produced by each stage. 'missing' means that step has "
                "not run yet — use the buttons on the left. "
                "Double-click a row to open the file."
            ),
            style="Hint.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.table = TreeTable(self.frame, height=16)
        self.table.frame.grid(row=1, column=0, sticky="nsew")
        self.table.tree.bind("<Double-1>", self._open_selected)

    def _open_selected(self, _event=None) -> None:
        tree = self.table.tree
        selected = tree.selection()
        if not selected:
            return
        values = tree.item(selected[0], "values")
        try:
            column = list(self.table.columns).index("path")
            relative = str(values[column])
        except (ValueError, IndexError):
            return
        if relative and relative not in ("", "nan"):
            _open_path(ROOT / relative)

    def refresh(self) -> None:
        self.table.set_data(
            pipeline.data_file_inventory(), sort_col="file", desc=False, reset_sort=True
        )


# ------------------------------------------------------------------- figures
class FigureTab:
    def __init__(self, parent):
        self.frame = ttk.Frame(parent, padding=10)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(1, weight=1)
        bar = ttk.Frame(self.frame)
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(bar, text="Figure:").pack(side="left")
        self.figure_var = tk.StringVar()
        self.combo = ttk.Combobox(
            bar, textvariable=self.figure_var, state="readonly", width=38
        )
        self.combo.pack(side="left", padx=(6, 6))
        self.combo.bind("<<ComboboxSelected>>", lambda _e: self._show())
        ttk.Button(bar, text="Refresh", command=self.refresh).pack(side="left")
        ttk.Button(bar, text="Open externally", command=self._open_current).pack(
            side="left", padx=6
        )
        self.picture = ttk.Label(self.frame, anchor="center")
        self.picture.grid(row=1, column=0, sticky="nsew")
        self.photo = None

    def refresh(self) -> None:
        paths = sorted(pipeline.OUT.glob("fig_*.png"))
        names = [p.name for p in paths]
        self.combo["values"] = names
        if not names:
            self.figure_var.set("")
            self.photo = None
            self.picture.configure(
                image="", text="No figures yet — run ⑤ Build summary & figures."
            )
            return
        if self.figure_var.get() not in names:
            self.figure_var.set(names[0])
        self._show()

    def _show(self) -> None:
        name = self.figure_var.get()
        if not name:
            return
        path = pipeline.OUT / name
        if not path.exists():
            return
        try:
            self.photo = _load_photo(path, max_w=1120, max_h=620)
            self.picture.configure(image=self.photo, text="")
        except Exception as exc:  # pragma: no cover
            self.picture.configure(
                image="", text=f"Could not render {name}:\n{exc}"
            )

    def _open_current(self) -> None:
        name = self.figure_var.get()
        if name:
            _open_path(pipeline.OUT / name)


# ----------------------------------------------------------------------- app
class App:
    """Main window (composition over Tk widgets keeps imports Tk-free)."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(APP_TITLE)
        self.root.geometry("1500x950")
        self.root.minsize(1180, 760)
        self.root.configure(background=BG)

        self.busy = False
        self._events: "queue.Queue[tuple]" = queue.Queue()
        self.step_buttons: list[ttk.Button] = []

        self._setup_style()
        self._build_header()
        self._build_footer()
        self._build_body()

        self.refresh_all()
        self._append_log(
            "Ready. ① Generate raw CSV (noise is configurable, seed blank = new "
            "random log each run) → ② QC → ③ predict → ④ simulate → ⑤ charts.\n"
        )
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- styling --------------------------------------------------------
    def _setup_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=BG, font=("DejaVu Sans", 9))
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG)
        style.configure("TLabelframe", background=BG)
        style.configure("TLabelframe.Label", background=BG, foreground=ACCENT,
                        font=("DejaVu Sans", 10, "bold"))
        style.configure("Hint.TLabel", foreground="#4b5563", font=("DejaVu Sans", 8))
        style.configure("Accent.TButton", font=("DejaVu Sans", 11, "bold"))
        style.configure("Header.TFrame", background=ACCENT)
        style.configure("Header.TLabel", background=ACCENT, foreground="#ffffff",
                        font=("DejaVu Sans", 17, "bold"))
        style.configure("SubHeader.TLabel", background=ACCENT, foreground="#d1fae5",
                        font=("DejaVu Sans", 9))
        style.configure("Treeview", rowheight=21, font=("DejaVu Sans", 9))
        style.configure("Treeview.Heading", font=("DejaVu Sans", 9, "bold"))
        style.configure("TNotebook.Tab", padding=(10, 5))

    # -- header / footer ------------------------------------------------
    def _build_header(self) -> None:
        header = ttk.Frame(self.root, style="Header.TFrame", padding=(16, 12))
        header.pack(side="top", fill="x")
        ttk.Label(
            header, text=APP_TITLE, style="Header.TLabel"
        ).pack(anchor="w")
        ttk.Label(
            header,
            text=(
                "Pi sensor log → QC → severity model → water balance "
                "(selectable simulation models) → drainage policies · "
                "Jianghuai polder · crop-loss PROXY, not RMB"
            ),
            style="SubHeader.TLabel",
        ).pack(anchor="w")

    def _build_footer(self) -> None:
        footer = ttk.Frame(self.root, padding=(10, 4, 10, 10))
        footer.pack(side="bottom", fill="x")
        top = ttk.Frame(footer)
        top.pack(fill="x")
        self.progress = ttk.Progressbar(top, mode="indeterminate", length=180)
        self.progress.pack(side="left")
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(top, textvariable=self.status_var).pack(side="left", padx=10)
        ttk.Button(top, text="Clear log", command=self._clear_log).pack(side="right")

        log_wrap = ttk.Frame(footer)
        log_wrap.pack(fill="both", expand=True, pady=(6, 0))
        log_wrap.columnconfigure(0, weight=1)
        log_wrap.rowconfigure(0, weight=1)
        self.log = tk.Text(
            log_wrap,
            height=8,
            wrap="none",
            background="#0f172a",
            foreground="#e2e8f0",
            insertbackground="#e2e8f0",
            font=("DejaVu Sans Mono", 9),
            relief="flat",
        )
        self.log.grid(row=0, column=0, sticky="nsew")
        vsb = ttk.Scrollbar(log_wrap, orient="vertical", command=self.log.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=vsb.set, state="disabled")

    # -- body -----------------------------------------------------------
    def _build_body(self) -> None:
        paned = ttk.PanedWindow(self.root, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=10, pady=8)

        self.control_scroll = ScrollableFrame(paned, width=400)
        try:
            paned.add(self.control_scroll.frame, weight=0)
        except tk.TclError:  # very old Tk
            paned.add(self.control_scroll.frame)

        right = ttk.Frame(paned)
        try:
            paned.add(right, weight=1)
        except tk.TclError:
            paned.add(right)
        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill="both", expand=True)

        self.overview_tab = OverviewTab(self.notebook)
        self.csv_browser = CsvBrowserTab(self.notebook)
        self.files_tab = FilesTab(self.notebook)
        self.figure_tab = FigureTab(self.notebook)

        self.notebook.add(self.overview_tab.frame, text="Overview")
        self.notebook.add(self.csv_browser.frame, text="CSV inspector")
        self.notebook.add(self.files_tab.frame, text="Files")
        self.notebook.add(self.figure_tab.frame, text="Figures")

        self._build_controls(self.control_scroll.inner)

    # -- control panel --------------------------------------------------
    def _section(self, parent, title: str):
        frame = ttk.LabelFrame(parent, text=title, padding=8)
        frame.pack(fill="x", padx=2, pady=(0, 8))
        frame.columnconfigure(1, weight=1)
        return frame

    def _row_entry(self, parent, row: int, label: str, var, hint: str = "") -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=var, width=10).grid(
            row=row, column=1, sticky="ew", padx=(6, 0), pady=2
        )
        if hint:
            ttk.Label(parent, text=hint, style="Hint.TLabel").grid(
                row=row + 1, column=0, columnspan=2, sticky="w"
            )
            return row + 2
        return row + 1

    def _button(
        self,
        parent,
        text: str,
        command,
        style: str = "TButton",
        row: int | None = None,
    ):
        """Add a step button.

        ``row`` places it with ``grid`` (for sections whose other children are
        gridded); omit it for pack-only parents. Tk does not allow pack and grid
        slaves to share a master.
        """
        button = ttk.Button(parent, text=text, command=command, style=style)
        if row is None:
            button.pack(fill="x", pady=(6, 0))
        else:
            button.grid(
                row=row, column=0, columnspan=2, sticky="ew", pady=(8, 0)
            )
        self.step_buttons.append(button)
        return button

    def _build_controls(self, parent) -> None:
        ttk.Label(
            parent,
            text="Run one stage at a time, or the whole pipeline. "
            "Inputs below are used by ① and ④.",
            style="Hint.TLabel",
            wraplength=350,
            justify="left",
        ).pack(fill="x", padx=2, pady=(0, 8))

        # ① sensor generation -------------------------------------------
        sec = self._section(parent, "① Sensor data — generate_raw_csv.py")
        self.rain_noise_var = tk.StringVar(value=str(pipeline.DEFAULT_RAIN_NOISE))
        self.rain_noise_dry_var = tk.StringVar(value=str(pipeline.DEFAULT_RAIN_NOISE_DRY))
        self.rain_sensor_noise_var = tk.StringVar(
            value=str(pipeline.DEFAULT_RAIN_SENSOR_NOISE)
        )
        self.vwc_sensor_noise_var = tk.StringVar(
            value=str(pipeline.DEFAULT_VWC_SENSOR_NOISE)
        )
        self.seed_var = tk.StringVar(value="")
        row = 0
        row = self._row_entry(
            sec, row, "Rain noise σ (mm/10min)", self.rain_noise_var,
            "hyetograph jitter while raining",
        )
        row = self._row_entry(
            sec, row, "Dry-slot rain σ", self.rain_noise_dry_var,
            "hyetograph jitter in dry slots",
        )
        row = self._row_entry(
            sec, row, "Rain sensor σ", self.rain_sensor_noise_var,
            "measurement noise added to the logged rain value",
        )
        row = self._row_entry(
            sec, row, "Soil sensor σ (m³/m³)", self.vwc_sensor_noise_var,
            "measurement noise added to the logged VWC value",
        )
        row = self._row_entry(
            sec, row, "Seed (blank = random)", self.seed_var,
            "leave blank for a different log every run",
        )
        self._button(
            sec, "① Generate raw CSV",
            lambda: self._start("Generate raw CSV", self._task_generate),
            row=row,
        )

        # ② QC -----------------------------------------------------------
        sec = self._section(parent, "② Quality control — ingest_qc.py")
        ttk.Label(
            sec,
            text="Flags stuck / suspect / saturated / missing readings, then "
            "builds features.csv (rain 1h/6h, VWC, hours above FC).",
            style="Hint.TLabel",
            wraplength=350,
            justify="left",
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        self._button(
            sec, "② Run QC + features",
            lambda: self._start("QC + features", self._task_qc),
            row=1,
        )

        # ③ predict ------------------------------------------------------
        sec = self._section(parent, "③ Severity model — predict.py")
        ttk.Label(
            sec,
            text="Gradient-boosting classifier trained on physics severity "
            "(time-ordered split, dummy baseline reported).",
            style="Hint.TLabel",
            wraplength=350,
            justify="left",
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        self._button(
            sec, "③ Train & predict severity",
            lambda: self._start("Train & predict severity", self._task_predict),
            row=1,
        )

        # ④ simulate -----------------------------------------------------
        sec = self._section(parent, "④ Water balance — hydrology.py")
        ttk.Label(sec, text="Simulation model:").grid(row=0, column=0, sticky="w")
        self.model_label_var = tk.StringVar(
            value=HYDRO_MODELS[DEFAULT_MODEL].label
        )
        self.model_combo = ttk.Combobox(
            sec,
            textvariable=self.model_label_var,
            state="readonly",
            values=[m.label for m in HYDRO_MODELS.values()],
        )
        self.model_combo.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        self.model_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_model_change())

        self.model_desc = ttk.Label(
            sec, text="", style="Hint.TLabel", wraplength=350, justify="left"
        )
        self.model_desc.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 2))
        self.model_params = ttk.Label(
            sec, text="", style="Hint.TLabel", wraplength=350, justify="left"
        )
        self.model_params.grid(row=2, column=0, columnspan=2, sticky="w")

        self.all_models_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            sec,
            text="Compare all models (runs every model above)",
            variable=self.all_models_var,
            command=self._on_model_change,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 2))

        ttk.Label(sec, text="Drainage policies:").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )
        self.policy_vars: dict[str, tk.BooleanVar] = {}
        for offset, key in enumerate(pipeline.POLICY_ORDER):
            var = tk.BooleanVar(value=True)
            self.policy_vars[key] = var
            ttk.Checkbutton(
                sec, text=pipeline.POLICY_LABELS[key], variable=var
            ).grid(row=5 + offset, column=0, columnspan=2, sticky="w")

        self._button(
            sec, "④ Run selected simulations",
            lambda: self._start("Water-balance simulation", self._task_simulate),
            row=8,
        )

        # ⑤ summary ------------------------------------------------------
        sec = self._section(parent, "⑤ Summary, charts & CSV list")
        ttk.Label(
            sec,
            text="Rebuilds damage_summary.csv from all simulation CSVs, "
            "redraws the PNG figures and refreshes the CSV dropdown with the "
            "files simulated in this run.",
            style="Hint.TLabel",
            wraplength=350,
            justify="left",
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        self._button(
            sec, "⑤ Build summary & figures",
            lambda: self._start("Summary + figures", self._task_figures),
            row=1,
        )

        # full run -------------------------------------------------------
        run_all = ttk.Frame(parent)
        run_all.pack(fill="x", padx=2, pady=(4, 8))
        self._button(
            run_all,
            "▶  Run full pipeline",
            lambda: self._start("Full pipeline", self._task_full),
            style="Accent.TButton",
        )
        ttk.Button(
            run_all, text="Open outputs folder", command=lambda: _open_path(pipeline.OUT)
        ).pack(fill="x", pady=(6, 0))

        self._on_model_change()

    def _on_model_change(self) -> None:
        compare_all = self.all_models_var.get()
        self.model_combo.configure(state="disabled" if compare_all else "readonly")
        key = MODEL_BY_LABEL.get(self.model_label_var.get(), DEFAULT_MODEL)
        spec = HYDRO_MODELS[key]
        self.model_desc.configure(text=spec.description)
        self.model_params.configure(
            text=f"{'All models' if compare_all else spec.label} · {spec.summary()}"
        )

    def _model_key(self) -> str:
        return MODEL_BY_LABEL.get(self.model_label_var.get(), DEFAULT_MODEL)

    # -- config ---------------------------------------------------------
    def _cfg_from_ui(self) -> pipeline.PipelineConfig:
        def number(var, name):
            raw = var.get().strip()
            if raw == "":
                raise ValueError(f"{name} is empty")
            try:
                return float(raw)
            except ValueError as exc:
                raise ValueError(f"{name} must be a number (got {raw!r})") from exc

        seed_text = self.seed_var.get().strip()
        seed = None
        if seed_text:
            try:
                seed = int(seed_text)
            except ValueError as exc:
                raise ValueError("Seed must be a whole number, or blank for random") from exc

        models = (
            tuple(HYDRO_MODELS)
            if self.all_models_var.get()
            else (self._model_key(),)
        )
        policies = tuple(
            key for key in pipeline.POLICY_ORDER if self.policy_vars[key].get()
        )
        if not policies:
            raise ValueError("Select at least one drainage policy")

        cfg = pipeline.PipelineConfig(
            seed=seed,
            rain_noise=number(self.rain_noise_var, "Rain noise σ"),
            rain_noise_dry=number(self.rain_noise_dry_var, "Dry-slot rain σ"),
            rain_sensor_noise=number(self.rain_sensor_noise_var, "Rain sensor σ"),
            vwc_sensor_noise=number(self.vwc_sensor_noise_var, "Soil sensor σ"),
            models=models,
            policies=policies,
            figures=True,
        )
        return cfg.validate()

    # -- tasks (run in a worker thread) ---------------------------------
    def _task_generate(self, cfg, log):
        return pipeline.generate_raw(cfg, log)

    def _task_qc(self, cfg, log):
        return pipeline.run_qc(log)

    def _task_predict(self, cfg, log):
        return pipeline.run_prediction(log)

    def _task_simulate(self, cfg, log):
        return pipeline.run_simulations(cfg, log)

    def _task_figures(self, cfg, log):
        # Rescan every simulation CSV (including models from earlier runs), then
        # redraw the charts from the refreshed summary.
        summary = pipeline.build_summary(log)
        figures = pipeline.build_figures(log)
        return {
            "summary": summary.to_dict(orient="records"),
            "figures": [str(path) for path in figures],
        }

    def _task_full(self, cfg, log):
        return pipeline.run_pipeline(cfg, log)

    def _start(self, label: str, fn) -> None:
        if self.busy:
            return
        try:
            cfg = self._cfg_from_ui()
        except ValueError as exc:
            messagebox.showerror("Check the inputs", str(exc), parent=self.root)
            return

        self.busy = True
        self._set_step_state("disabled")
        self.status_var.set(f"{label} — running …")
        self.progress.start(14)
        self._append_log(f"\n=== {label} ===")
        threading.Thread(
            target=self._worker, args=(label, fn, cfg), daemon=True
        ).start()
        self.root.after(80, self._poll)

    def _worker(self, label: str, fn, cfg) -> None:
        try:
            result = fn(cfg, self._queue_log)
            self._events.put(("done", label, result))
        except Exception:
            self._events.put(("error", label, traceback.format_exc()))

    def _queue_log(self, message) -> None:
        self._events.put(("log", str(message)))

    def _poll(self) -> None:
        try:
            while True:
                item = self._events.get_nowait()
                kind = item[0]
                if kind == "log":
                    self._append_log(item[1])
                elif kind == "done":
                    self._finish(item[1], item[2], None)
                elif kind == "error":
                    self._finish(item[1], None, item[2])
        except queue.Empty:
            pass
        if self.busy:
            self.root.after(80, self._poll)

    def _finish(self, label: str, result, error) -> None:
        self.busy = False
        self.progress.stop()
        self._set_step_state("normal")
        if error is not None:
            self.status_var.set(f"{label} — failed")
            self._append_log(error.rstrip() + "\n")
            last_line = error.strip().splitlines()[-1] if error.strip() else "Unknown error"
            messagebox.showerror(f"{label} failed", last_line, parent=self.root)
        else:
            self.status_var.set(f"{label} — done")
            self._append_log(self._format_result(result))
        self.refresh_all()

    @staticmethod
    def _format_result(result) -> str:
        if isinstance(result, dict):
            bits = []
            summary = result.get("summary")
            if summary:
                bits.append(f"{len(summary)} simulation rows")
            figures = result.get("figures")
            if figures:
                bits.append(f"{len(figures)} figures")
            metrics = result.get("metrics")
            if isinstance(metrics, dict) and metrics.get("accuracy") is not None:
                bits.append(f"model accuracy {metrics['accuracy']}")
            return "Completed — " + (", ".join(bits) if bits else "ok") + "\n"
        if isinstance(result, pd.DataFrame):
            return f"Completed — {len(result)} rows\n"
        if isinstance(result, (list, tuple)):
            if not result:
                return "Completed — nothing to do\n"
            return "Completed — " + ", ".join(str(x) for x in result) + "\n"
        return f"Completed — {result}\n"

    def _set_step_state(self, state: str) -> None:
        for button in self.step_buttons:
            try:
                button.configure(state=state)
            except tk.TclError:
                pass

    # -- log ------------------------------------------------------------
    def _append_log(self, text: str) -> None:
        if not text:
            return
        self.log.configure(state="normal")
        self.log.insert("end", text if text.endswith("\n") else text + "\n")
        lines = int(self.log.index("end-1c").split(".")[0])
        if lines > 2600:
            self.log.delete("1.0", f"{lines - 2000}.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    # -- refresh --------------------------------------------------------
    def refresh_all(self) -> None:
        try:
            self.csv_browser.refresh()
            self.overview_tab.refresh()
            self.files_tab.refresh()
            self.figure_tab.refresh()
        except Exception:
            self._append_log(
                "Could not refresh the result views:\n" + traceback.format_exc()
            )

    # -- shutdown -------------------------------------------------------
    def _on_close(self) -> None:
        if self.busy:
            if not messagebox.askyesno(
                "A step is still running",
                "A pipeline step is still running. Close anyway?",
                parent=self.root,
            ):
                return
        self.root.destroy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tkinter control room for the rice-waterlogging demo."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify imports + list the categorized CSVs without opening a window",
    )
    args = parser.parse_args(argv)

    if args.check:
        print("Project root:", ROOT)
        print("Hydrology models:")
        for spec in HYDRO_MODELS.values():
            print(f"  - {spec.key:<10} {spec.label}  ({spec.summary()})")
        print("CSV inspector (latest simulated run only):")
        paths = pipeline.simulated_csv_paths()
        all_paths = pipeline.expected_csv_paths()
        if not paths:
            print("  (none yet — run ④ or the full pipeline)")
        metas = [pipeline.csv_metadata(p) for p in paths]
        type_names = sorted(
            {m["type"] for m in metas},
            key=lambda t: (
                pipeline.CSV_TYPE_ORDER.index(t)
                if t in pipeline.CSV_TYPE_ORDER
                else len(pipeline.CSV_TYPE_ORDER)
            ),
        )
        print("  types:     ", ", ".join(type_names) or "-")
        print("  models:    ", ", ".join(sorted({m["model"] for m in metas})) or "-")
        print("  treatments:", ", ".join(sorted({m["treatment"] for m in metas})) or "-")
        for by in pipeline.CSV_SORT_CATEGORIES:
            names = [
                m["filename"]
                for m in (
                    pipeline.csv_metadata(p)
                    for p in pipeline.sort_csv_paths(paths, by=by)
                )
            ]
            preview = ", ".join(names[:5]) + (" …" if len(names) > 5 else "")
            print(f"  grouped by {by}: {preview or '(none)'}")
        print(
            f"  {len(paths)} CSVs listed; {len(all_paths)} on disk "
            f"({len(all_paths) - len(paths)} hidden)"
        )

        # Exercise the GUI data layer (no Tk required) so --check is a real
        # smoke test on headless machines.
        print("GUI data helpers:")
        try:
            cfg = pipeline.PipelineConfig().validate()
            print(
                "  defaults: seed={}, noise=({}, {}, {}, {}), models={}, policies={}".format(
                    cfg.seed,
                    cfg.rain_noise,
                    cfg.rain_noise_dry,
                    cfg.rain_sensor_noise,
                    cfg.vwc_sensor_noise,
                    ",".join(cfg.models),
                    ",".join(cfg.policies),
                )
            )
            cats = pipeline.build_category_stats()
            metrics = pipeline.build_metrics_table()
            inventory = pipeline.data_file_inventory()
            print(f"  category stats: {len(cats)} rows (sorted by count)")
            print(f"  metrics table:  {len(metrics)} rows")
            print(f"  file inventory: {len(inventory)} rows")
            readable = 0
            for path in paths:
                pd.read_csv(path, nrows=1)
                readable += 1
            print(f"  CSV inspector:  {readable}/{len(paths)} readable")
        except Exception as exc:
            print("  data-layer check FAILED:", exc)
            return 2

        if TK_IMPORT_ERROR is not None:
            print("Tkinter: NOT available —", TK_IMPORT_ERROR)
            return 1
        print("Tkinter:", tk.TkVersion, "(window support OK)")
        return 0

    if TK_IMPORT_ERROR is not None:
        print("Tkinter could not be imported:", TK_IMPORT_ERROR)
        print("Install it first:")
        print("  Debian/Ubuntu: sudo apt-get install python3-tk")
        print("  Fedora:        sudo dnf install python3-tkinter")
        print("  macOS/Windows: use the python.org installer (Tk is bundled)")
        print("\nThe CLI still works: python3 run_demo.py")
        return 1

    app = App()
    app.root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
