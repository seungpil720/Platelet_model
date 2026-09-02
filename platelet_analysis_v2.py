#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
platelet_sparse_hurdle_analysis_v2.py

Sparse-aware platelet demand forecasting, inventory-policy simulation, and
manuscript figure/table generation.

This is the consolidated, GitHub-ready v2 pipeline for the platelet-demand
forecasting revision. It merges and cleans up two Colab notebook exports:

  - platelet_colab_sparse_hurdle_plus_models.py
      Sparse/intermittent diagnostics, hurdle/two-stage models, LightGBM,
      Prophet, LSTM, validation-based model selection, and inventory-policy
      simulation (Part 1), plus the manuscript Figure 1/2 and Table 1/2/4/5
      generation code that consumes Part 1's outputs (Part 2).
  - platelet_figures_20260702.py
      An earlier, self-contained draft pipeline (simpler LASSO/Ridge/SARIMA
      models with its own figures/tables). Its methodology was superseded by
      the sparse-hurdle pipeline above, so it is not reproduced here; only
      the current (Part 1 + Part 2) methodology is kept.

What changed relative to the raw Colab exports
------------------------------------------------
  - All Google Colab-only code (drive.mount, "!pip install ...", display())
    was removed. The script now runs as a plain, local/CLI Python program.
  - Duplicate function definitions introduced by re-pasted notebook cells
    (e.g. two identical copies of safe_filename(), two near-identical copies
    of ensure_dir()) were de-duplicated, keeping the version that actually
    executed last in the notebook.
  - Two different constants were both named META_COLS in the original
    notebook (one for the modeling data loader, one for the figure data
    loader). The figure-side constant was renamed to FIGURE_META_COLS to
    avoid a silent collision; no logic was changed.
  - The many repeated "for each model, plot ... and save" notebook cells
    (Figure 2 per model, ABO-trend per model, product-ABO trend per model)
    were consolidated into one function, generate_supplementary_figures(),
    kept as an optional step rather than always executed.
  - A single argparse-driven main() now drives the whole pipeline: run the
    forecasting/inventory analysis, then (optionally) regenerate Figure 1,
    Figure 2, and Tables 1/2/4/5 from its outputs.

Usage
-----
    python platelet_sparse_hurdle_analysis_v2.py --data /path/to/platelet_data_english_260529.xlsx

Optional flags (see `python platelet_sparse_hurdle_analysis_v2.py --help`):
    --output-dir DIR        Directory for the analysis CSV/XLSX outputs.
    --figures-dir DIR       Directory for Figure 1 / Figure 2 PNGs.
    --tables-dir DIR        Directory for the manuscript Excel/Word tables.
    --skip-analysis         Skip Part 1 and reuse existing CSVs in --output-dir.
    --skip-figures          Skip Figure 1 / Figure 2 generation.
    --skip-tables           Skip Table 1/2/4/5 generation.
    --extra-figures         Also generate the supplementary per-model figures
                             (Figure 2, ABO-trend, and product-ABO-trend for
                             every model found in the predictions file).
    --no-lightgbm / --no-prophet / --no-lstm
                             Skip the corresponding optional model family.

Requirements
------------
    numpy, pandas, scipy, scikit-learn, matplotlib, openpyxl
Optional (a model family or export is skipped with a warning if missing):
    lightgbm, prophet, torch, python-docx
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe default for headless/CLI runs; ignored if a GUI backend is already active
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Optional models (LightGBM, Prophet, LSTM/PyTorch) and the optional Word
# export (python-docx) are imported lazily inside the functions that need
# them, so the core analysis still runs in a minimal environment.

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# ============================================================
# PART 1. Sparse-aware forecasting and inventory-policy pipeline
# ============================================================

# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------
@dataclass


class SparseHurdleConfig:
    data_path: str = "platelet_data_english_260529.xlsx"
    output_dir: str = "results_sparse_hurdle_plus_models"

    train_start: str = "2019-03-01"
    train_end: str = "2024-02-29"
    val_start: str = "2024-03-01"
    val_end: str = "2024-08-31"
    test_start: str = "2024-09-01"
    test_end: str = "2025-02-28"

    target_prefixes: Tuple[str, str] = ("plt", "aph")
    abo_types: Tuple[str, str, str, str] = ("a", "b", "o", "ab")

    # Sparse/intermittent diagnostics.
    # A target is forced to include hurdle/two-stage candidates when either condition is met.
    sparse_zero_rate_threshold: float = 0.20
    intermittent_adi_threshold: float = 1.32
    intermittent_cv2_threshold: float = 0.49
    min_positive_days_for_hurdle: int = 30

    # Forecasting setup.
    max_horizon: int = 3
    moving_windows: Tuple[int, int] = (7, 14)
    seasonal_period: int = 7
    high_demand_quantile: float = 0.90
    underprediction_weight: float = 3.0
    random_state: int = 42

    # Optional extended models requested for manuscript comparison.
    # Set these to False to skip a model family and reduce run time.
    run_lightgbm: bool = True
    run_prophet: bool = True
    run_lstm: bool = True

    # LightGBM settings.
    lightgbm_n_estimators: int = 200
    lightgbm_learning_rate: float = 0.03
    lightgbm_num_leaves: int = 15
    lightgbm_min_child_samples: int = 20

    # Prophet settings.  Prophet is used as an interpretable time-series model with
    # selected known/lagged regressors. To keep the model stable, only a small number
    # of leakage-safe regressors are used.
    prophet_max_regressors: int = 12
    prophet_include_yearly: bool = True
    prophet_include_weekly: bool = True

    # LSTM settings. The LSTM is treated as a high-capacity benchmark; results should
    # be interpreted cautiously when the demand distribution shifts or sparse targets dominate.
    lstm_sequence_length: int = 14
    lstm_hidden_size: int = 32
    lstm_num_layers: int = 1
    lstm_dropout: float = 0.10
    lstm_max_epochs: int = 30
    lstm_batch_size: int = 32
    lstm_learning_rate: float = 1e-3
    lstm_patience: int = 5

    # Hurdle threshold tuning.
    occurrence_threshold_grid: Tuple[float, ...] = (0.20, 0.30, 0.40, 0.50, 0.60)
    threshold_selection_metric: str = "weighted_mae"  # or "cost_sensitive"

    # Inventory objective.
    shelf_life_days: int = 4
    lead_time_days: int = 1
    initial_days_supply: float = 2.0
    target_days_supply_grid: Tuple[float, ...] = (2.0, 2.5, 3.0, 3.5)
    safety_factor_grid: Tuple[float, ...] = (0.90, 1.00, 1.10, 1.20)
    shortage_cost: float = 10.0
    wastage_cost: float = 1.0
    procurement_cost: float = 0.0


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def in_window(index: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    return (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))


# ============================================================
# 2. Data loading
# ============================================================

META_COLS = [
    "Category",
    "Revised variable name",
    "Condition 1 (inpatient/outpatient)",
    "Condition 2 (date)",
    "Description",
    "Condition 3 (detail)",
]


def clean_variable_name(x: object) -> str:
    s = str(x).strip().replace(" ", "_").replace("-", "_")
    s = re.sub(r"[^0-9A-Za-z_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if s.startswith("PLT_"):
        s = "plt_" + s[len("PLT_"):]
    if s.endswith("_O"):
        s = s[:-2] + "_o"
    return s


def load_platelet_excel(xlsx_path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"File not found: {xlsx_path}")
    raw = pd.read_excel(xlsx_path, sheet_name=0)
    if raw.shape[1] <= len(META_COLS):
        raise ValueError("Expected metadata columns followed by date columns.")

    raw = raw.rename(columns=dict(zip(raw.columns[: len(META_COLS)], META_COLS)))
    metadata = raw[META_COLS].copy()
    metadata["variable"] = metadata["Revised variable name"].map(clean_variable_name)

    date_cols = list(raw.columns[len(META_COLS):])
    dates = pd.to_datetime(date_cols, errors="coerce")
    if dates.isna().any():
        bad = [str(c) for c, d in zip(date_cols, dates) if pd.isna(d)][:10]
        raise ValueError(f"Could not parse some date columns: {bad}")

    values = raw[date_cols].copy()
    values.index = metadata["variable"].values
    panel = values.T
    panel.index = pd.DatetimeIndex(dates, name="date")
    panel.columns = metadata["variable"].values
    panel = panel.sort_index().apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if panel.columns.duplicated().any():
        panel = panel.T.groupby(level=0).sum().T
        metadata = metadata.drop_duplicates(subset=["variable"], keep="first").reset_index(drop=True)

    return panel, metadata


def get_target_columns(config: SparseHurdleConfig, panel: pd.DataFrame) -> List[str]:
    targets: List[str] = []
    for product in config.target_prefixes:
        for abo in config.abo_types:
            c = f"{product}_transf_{abo}"
            if c in panel.columns:
                targets.append(c)
    if not targets:
        raise ValueError("No target columns found. Expected columns such as plt_transf_a and aph_transf_a.")
    return targets


# ============================================================
# 3. Sparse / intermittent diagnostics
# ============================================================

def average_demand_interval(y: pd.Series) -> float:
    y = pd.Series(y).astype(float)
    positive_positions = np.flatnonzero(y.values > 0)
    if len(positive_positions) <= 1:
        return float("inf")
    intervals = np.diff(positive_positions)
    return float(np.mean(intervals))


def squared_cv_nonzero(y: pd.Series) -> float:
    nz = pd.Series(y).astype(float)
    nz = nz[nz > 0]
    if len(nz) <= 1 or np.isclose(nz.mean(), 0.0):
        return float("inf")
    return float((nz.std(ddof=1) / nz.mean()) ** 2)


def classify_intermittent_demand(adi: float, cv2: float, adi_thr: float = 1.32, cv2_thr: float = 0.49) -> str:
    # Syntetos-Boylan style classification.
    if adi < adi_thr and cv2 < cv2_thr:
        return "smooth"
    if adi >= adi_thr and cv2 < cv2_thr:
        return "intermittent"
    if adi < adi_thr and cv2 >= cv2_thr:
        return "erratic"
    return "lumpy"


def sparse_diagnostics(panel: pd.DataFrame, targets: Sequence[str], config: SparseHurdleConfig) -> pd.DataFrame:
    rows = []
    for t in targets:
        y = panel[t].astype(float)
        zero_rate = float((y == 0).mean())
        adi = average_demand_interval(y)
        cv2 = squared_cv_nonzero(y)
        cls = classify_intermittent_demand(
            adi,
            cv2,
            config.intermittent_adi_threshold,
            config.intermittent_cv2_threshold,
        )
        forced_hurdle = bool(
            zero_rate >= config.sparse_zero_rate_threshold
            or adi >= config.intermittent_adi_threshold
            or cls in {"intermittent", "lumpy"}
        )
        rows.append({
            "target": t,
            "product": "APC" if t.startswith("aph_") else "PC",
            "mean_demand": float(y.mean()),
            "median_demand": float(y.median()),
            "zero_rate": zero_rate,
            "nonzero_days": int((y > 0).sum()),
            "ADI_average_demand_interval": adi,
            "CV2_nonzero_demand": cv2,
            "sbc_classification": cls,
            "force_hurdle_candidates": forced_hurdle,
            "method_recommendation": (
                "Use hurdle/two-stage candidates; general regression alone is unstable"
                if forced_hurdle else
                "General regression acceptable; still compare against adaptive baselines"
            ),
        })
    return pd.DataFrame(rows)


# ============================================================
# 4. Feature engineering
# ============================================================

def make_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    cal = pd.DataFrame(index=index)
    cal["dow"] = index.dayofweek
    cal["month"] = index.month
    cal["is_weekend"] = (index.dayofweek >= 5).astype(int)
    cal["dow_mon"] = (index.dayofweek == 0).astype(int)
    cal["dow_tue"] = (index.dayofweek == 1).astype(int)
    cal["dow_wed"] = (index.dayofweek == 2).astype(int)
    cal["dow_thu"] = (index.dayofweek == 3).astype(int)
    cal["dow_fri"] = (index.dayofweek == 4).astype(int)
    cal["dow_sat"] = (index.dayofweek == 5).astype(int)
    cal["dow_sun"] = (index.dayofweek == 6).astype(int)
    return cal


def build_feature_frame(panel: pd.DataFrame, targets: Sequence[str], horizon: int, windows: Sequence[int]) -> pd.DataFrame:
    """Create leakage-safe features for direct horizon forecasting.

    Target y(t+h) is predicted at time t.  Calendar is known at target date t+h.
    All observed clinical/product variables are lagged so they are available at time t.
    """
    features = pd.DataFrame(index=panel.index)

    # Calendar features for target date.
    future_index = panel.index + pd.Timedelta(days=horizon)
    cal = make_calendar_features(pd.DatetimeIndex(future_index))
    cal.index = panel.index
    features = pd.concat([features, cal], axis=1)

    # Lagged target history and rolling history.
    for t in targets:
        features[f"{t}_lag1"] = panel[t].shift(1)
        features[f"{t}_lag7"] = panel[t].shift(7)
        for w in windows:
            features[f"{t}_roll{w}_mean"] = panel[t].shift(1).rolling(w, min_periods=max(2, min(w, 3))).mean()
            features[f"{t}_roll{w}_max"] = panel[t].shift(1).rolling(w, min_periods=max(2, min(w, 3))).max()
            features[f"{t}_roll{w}_nonzero_rate"] = (panel[t].shift(1) > 0).rolling(w, min_periods=max(2, min(w, 3))).mean()

    # Lagged non-target variables.  To keep Colab execution fast and prevent noisy high-dimensional
    # fitting, retain clinically and operationally relevant predictors only.  Target history above
    # already captures all 8 target series.
    relevant_patterns = (
        "income", "expire", "transfused", "past_1wk",
        "dept_pt_IMH", "ward_pt_MICU", "ward_pt_NICU",
        "surg_general", "surg_",
        "plt_1_", "plt_2_", "plt_3_", "plt_4_", "plt_5_", "plt_6_", "plt_7_", "plt_8_",
        "wbc_", "hb_", "inr_", "aptt_",
    )
    non_targets = [c for c in panel.columns if c not in targets and any(p in c for p in relevant_patterns)]
    lagged = panel[non_targets].shift(1)
    lagged.columns = [f"{c}_lag1" for c in non_targets]
    features = pd.concat([features, lagged], axis=1)

    features = features.replace([np.inf, -np.inf], np.nan)
    return features


def target_series_for_horizon(panel: pd.DataFrame, target: str, horizon: int) -> pd.Series:
    return panel[target].shift(-horizon).rename(target)


# ============================================================
# 5. Models
# ============================================================

def make_lasso(alpha: float = 0.05, random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=alpha, max_iter=1000, tol=1e-3, selection="random", random_state=random_state)),
    ])


def make_ridge(alpha: float = 5.0) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])


def make_logistic(random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=0.5,
            penalty="l2",
            class_weight="balanced",
            solver="liblinear",
            max_iter=3000,
            random_state=random_state,
        )),
    ])



def make_lightgbm_regressor(config: SparseHurdleConfig):
    """Return a LightGBM regressor if lightgbm is installed; otherwise return None.

    LightGBM is useful as a nonlinear tabular benchmark. It is evaluated with the
    same leakage-safe features and validation/test splits as LASSO and Ridge.
    """
    try:
        from lightgbm import LGBMRegressor
    except Exception as e:
        print("[skip] LightGBM is not installed. In Colab, run: !pip -q install lightgbm")
        return None
    return LGBMRegressor(
        n_estimators=config.lightgbm_n_estimators,
        learning_rate=config.lightgbm_learning_rate,
        num_leaves=config.lightgbm_num_leaves,
        min_child_samples=config.lightgbm_min_child_samples,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="regression_l1",
        random_state=config.random_state,
        n_jobs=1,
        verbosity=-1,
    )


def fit_predict_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, X_test: pd.DataFrame, config: SparseHurdleConfig) -> Optional[Tuple[pd.Series, pd.Series]]:
    model = make_lightgbm_regressor(config)
    if model is None:
        return None
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])
    pipe.fit(X_train, y_train)
    val_pred = pd.Series(np.maximum(pipe.predict(X_val), 0.0), index=X_val.index)
    test_pred = pd.Series(np.maximum(pipe.predict(X_test), 0.0), index=X_test.index)
    return val_pred, test_pred


def _select_prophet_regressors(X_train: pd.DataFrame, target: str, max_regressors: int) -> List[str]:
    """Select a small set of stable, leakage-safe regressors for Prophet."""
    priority_patterns = [
        f"{target}_lag1", f"{target}_lag7", f"{target}_roll7_mean", f"{target}_roll14_mean",
        "dow", "month", "is_weekend", "dow_fri",
        "plt_3_inpt", "plt_4_inpt", "plt_5_inpt", "dept_pt_IMH", "surg_general",
    ]
    cols = []
    for pat in priority_patterns:
        matches = [c for c in X_train.columns if c == pat or pat in c]
        for c in matches:
            if c not in cols:
                cols.append(c)
    # Fill remaining slots with features most correlated with y will be handled outside if needed;
    # here we keep deterministic selection to avoid target leakage through validation/test.
    numeric_cols = [c for c in X_train.columns if c not in cols]
    for c in numeric_cols:
        if len(cols) >= max_regressors:
            break
        if any(k in c for k in ["lag1", "lag7", "roll7_mean", "roll14_mean", "dow", "month"]):
            cols.append(c)
    return cols[:max_regressors]


def fit_predict_prophet(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    target: str,
    horizon: int,
    config: SparseHurdleConfig,
) -> Optional[Tuple[pd.Series, pd.Series]]:
    """Fit Prophet with a small set of known/lagged regressors.

    The ds column is the target date (origin date + horizon). Regressors are values
    available at the forecast origin, so this remains leakage-safe.
    """
    try:
        from prophet import Prophet
    except Exception:
        print("[skip] Prophet is not installed. In Colab, run: !pip -q install prophet")
        return None

    regs = _select_prophet_regressors(X_train, target, config.prophet_max_regressors)
    imputer = SimpleImputer(strategy="median")
    Xtr = pd.DataFrame(imputer.fit_transform(X_train[regs]), index=X_train.index, columns=regs) if regs else pd.DataFrame(index=X_train.index)
    Xv = pd.DataFrame(imputer.transform(X_val[regs]), index=X_val.index, columns=regs) if regs else pd.DataFrame(index=X_val.index)
    Xte = pd.DataFrame(imputer.transform(X_test[regs]), index=X_test.index, columns=regs) if regs else pd.DataFrame(index=X_test.index)

    train_df = pd.DataFrame({
        "ds": X_train.index + pd.Timedelta(days=horizon),
        "y": y_train.values,
    })
    for c in regs:
        train_df[c] = Xtr[c].values

    try:
        m = Prophet(
            weekly_seasonality=config.prophet_include_weekly,
            yearly_seasonality=config.prophet_include_yearly,
            daily_seasonality=False,
            seasonality_mode="additive",
            changepoint_prior_scale=0.05,
        )
        for c in regs:
            m.add_regressor(c, standardize=True)
        m.fit(train_df)

        def _predict(X_eval: pd.DataFrame) -> pd.Series:
            future = pd.DataFrame({"ds": X_eval.index + pd.Timedelta(days=horizon)})
            X_eval_imp = Xv if X_eval is X_val else Xte
            for c in regs:
                future[c] = X_eval_imp[c].values
            fc = m.predict(future)
            return pd.Series(np.maximum(fc["yhat"].values, 0.0), index=X_eval.index)

        return _predict(X_val), _predict(X_test)
    except Exception as e:
        print(f"[skip] Prophet failed for {target}, horizon={horizon}: {e}")
        return None


class _TorchLSTMRegressor:
    """Small PyTorch LSTM benchmark for tabular time-series features."""

    def __init__(self, config: SparseHurdleConfig):
        self.config = config
        self.fitted_ = False

    def _make_sequences(self, X: pd.DataFrame, y: pd.Series, fit: bool = False):
        import torch
        if fit:
            self.imputer_ = SimpleImputer(strategy="median")
            self.scaler_ = StandardScaler()
            X_imp = self.imputer_.fit_transform(X)
            X_scaled = self.scaler_.fit_transform(X_imp)
        else:
            X_imp = self.imputer_.transform(X)
            X_scaled = self.scaler_.transform(X_imp)
        L = self.config.lstm_sequence_length
        xs, ys, idx = [], [], []
        for i in range(L - 1, len(X_scaled)):
            xs.append(X_scaled[i - L + 1:i + 1, :])
            ys.append(float(y.iloc[i]))
            idx.append(y.index[i])
        if not xs:
            return None, None, []
        return torch.tensor(np.asarray(xs), dtype=torch.float32), torch.tensor(np.asarray(ys), dtype=torch.float32).view(-1, 1), idx

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
        try:
            import torch
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset
        except Exception:
            print("[skip] PyTorch is not installed; LSTM skipped.")
            return False

        torch.manual_seed(self.config.random_state)
        Xtr, ytr, _ = self._make_sequences(X_train, y_train, fit=True)
        Xv, yv, _ = self._make_sequences(X_val, y_val, fit=False)
        if Xtr is None or Xv is None:
            return False
        n_features = Xtr.shape[-1]
        cfg = self.config

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                dropout = cfg.lstm_dropout if cfg.lstm_num_layers > 1 else 0.0
                self.lstm = nn.LSTM(n_features, cfg.lstm_hidden_size, cfg.lstm_num_layers, batch_first=True, dropout=dropout)
                self.head = nn.Linear(cfg.lstm_hidden_size, 1)
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :])

        self.model_ = Net()
        opt = torch.optim.Adam(self.model_.parameters(), lr=cfg.lstm_learning_rate)
        loss_fn = nn.L1Loss()
        loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=cfg.lstm_batch_size, shuffle=False)
        best_val = float("inf")
        best_state = None
        patience_left = cfg.lstm_patience
        for epoch in range(cfg.lstm_max_epochs):
            self.model_.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(self.model_(xb), yb)
                loss.backward()
                opt.step()
            self.model_.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.model_(Xv), yv).item()
            if val_loss + 1e-6 < best_val:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
                patience_left = cfg.lstm_patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.fitted_ = True
        return True

    def predict(self, X: pd.DataFrame, y_placeholder: pd.Series) -> pd.Series:
        import torch
        Xseq, _, idx = self._make_sequences(X, y_placeholder, fit=False)
        if Xseq is None:
            return pd.Series(np.nan, index=X.index)
        self.model_.eval()
        with torch.no_grad():
            pred = self.model_(Xseq).cpu().numpy().ravel()
        out = pd.Series(np.nan, index=X.index, dtype=float)
        out.loc[idx] = np.maximum(pred, 0.0)
        # Fill the first sequence_length-1 dates with a simple median of predictions to keep metrics defined.
        fill_value = float(np.nanmedian(out.values)) if np.isfinite(np.nanmedian(out.values)) else 0.0
        return out.fillna(fill_value)


def fit_predict_lstm(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, config: SparseHurdleConfig) -> Optional[Tuple[pd.Series, pd.Series]]:
    try:
        model = _TorchLSTMRegressor(config)
        ok = model.fit(X_train, y_train, X_val, y_val)
        if not ok:
            return None
        return model.predict(X_val, y_val), model.predict(X_test, y_test)
    except Exception as e:
        print(f"[skip] LSTM failed: {e}")
        return None


class HurdleModel:
    """Two-stage occurrence × positive-quantity model.

    The model can return a soft expected-demand forecast: P(y>0) × E(y | y>0),
    and thresholded forecasts where low occurrence probability maps to zero.
    Thresholds are tuned on validation, not test.
    """

    def __init__(
        self,
        occurrence_model: Optional[Pipeline] = None,
        quantity_model: Optional[Pipeline] = None,
        threshold: float = 0.50,
        min_positive_days: int = 30,
        random_state: int = 42,
    ):
        self.occurrence_model = occurrence_model or make_logistic(random_state=random_state)
        self.quantity_model = quantity_model or make_lasso(alpha=0.05, random_state=random_state)
        self.threshold = threshold
        self.min_positive_days = min_positive_days
        self.random_state = random_state
        self.fitted_ = False
        self.fallback_positive_mean_: float = 0.0
        self.has_two_classes_: bool = True
        self.has_enough_positives_: bool = True

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HurdleModel":
        y = pd.Series(y).astype(float)
        occ = (y > 0).astype(int)
        self.has_two_classes_ = occ.nunique() == 2
        self.fallback_positive_mean_ = float(y[y > 0].mean()) if (y > 0).any() else 0.0
        self.has_enough_positives_ = int((y > 0).sum()) >= self.min_positive_days

        if self.has_two_classes_:
            self.occurrence_model.fit(X, occ)
        else:
            self.constant_occurrence_ = int(occ.iloc[0]) if len(occ) else 0

        if self.has_enough_positives_:
            self.quantity_model.fit(X.loc[y > 0], y.loc[y > 0])
        self.fitted_ = True
        return self

    def occurrence_probability(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("Model is not fitted.")
        if self.has_two_classes_:
            try:
                return self.occurrence_model.predict_proba(X)[:, 1]
            except Exception:
                pred = self.occurrence_model.predict(X)
                return np.asarray(pred, dtype=float)
        return np.full(len(X), float(getattr(self, "constant_occurrence_", 0)))

    def positive_quantity(self, X: pd.DataFrame) -> np.ndarray:
        if self.has_enough_positives_:
            q = self.quantity_model.predict(X)
        else:
            q = np.full(len(X), self.fallback_positive_mean_)
        return np.maximum(np.asarray(q, dtype=float), 0.0)

    def predict_soft(self, X: pd.DataFrame) -> np.ndarray:
        p = self.occurrence_probability(X)
        q = self.positive_quantity(X)
        return np.maximum(p * q, 0.0)

    def predict_thresholded(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        threshold = self.threshold if threshold is None else threshold
        p = self.occurrence_probability(X)
        q = self.positive_quantity(X)
        return np.where(p >= threshold, q, 0.0).astype(float)


def rolling_forecast_series(y_full: pd.Series, eval_index: pd.DatetimeIndex, horizon: int, window: int, kind: str = "mean") -> pd.Series:
    """Forecast y(t+h) using information up to t = target_date - h."""
    preds = []
    for target_date in eval_index:
        cutoff = target_date - pd.Timedelta(days=horizon)
        hist = y_full.loc[y_full.index <= cutoff].tail(window)
        if len(hist) == 0:
            val = 0.0
        elif kind == "nonzero_rate":
            val = float((hist > 0).mean())
        elif kind == "positive_mean":
            pos = hist[hist > 0]
            val = float(pos.mean()) if len(pos) else 0.0
        else:
            val = float(hist.mean())
        preds.append(max(val, 0.0))
    return pd.Series(preds, index=eval_index)


def seasonal_naive_forecast(y_full: pd.Series, eval_index: pd.DatetimeIndex, horizon: int, seasonal_period: int = 7) -> pd.Series:
    preds = []
    for target_date in eval_index:
        source_date = target_date - pd.Timedelta(days=seasonal_period)
        if source_date in y_full.index:
            val = float(y_full.loc[source_date])
        else:
            cutoff = target_date - pd.Timedelta(days=horizon)
            hist = y_full.loc[y_full.index <= cutoff]
            val = float(hist.tail(seasonal_period).mean()) if len(hist) else 0.0
        preds.append(max(val, 0.0))
    return pd.Series(preds, index=eval_index)


# ============================================================
# 6. Metrics and threshold selection
# ============================================================

def weighted_mae(y_true: np.ndarray, y_pred: np.ndarray, under_weight: float = 3.0) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = np.abs(y_true - y_pred)
    weights = np.where(y_pred < y_true, under_weight, 1.0)
    return float(np.mean(err * weights))


def safe_mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, seasonal_period: int = 7) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) <= seasonal_period:
        denom = np.mean(np.abs(np.diff(y_train))) if len(y_train) > 1 else np.nan
    else:
        denom = np.mean(np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period]))
    if not np.isfinite(denom) or denom <= 1e-12:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)) / denom)


def occurrence_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.50) -> Dict[str, float]:
    y_bin = (np.asarray(y_true) > 0).astype(int)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    out = {
        "occurrence_accuracy": float(accuracy_score(y_bin, y_pred)),
        "occurrence_precision": float(precision_score(y_bin, y_pred, zero_division=0)),
        "occurrence_recall": float(recall_score(y_bin, y_pred, zero_division=0)),
        "occurrence_f1": float(f1_score(y_bin, y_pred, zero_division=0)),
    }
    if len(np.unique(y_bin)) == 2 and len(np.unique(y_score)) > 1:
        try:
            out["occurrence_auc"] = float(roc_auc_score(y_bin, y_score))
        except Exception:
            out["occurrence_auc"] = np.nan
    else:
        out["occurrence_auc"] = np.nan
    return out


def forecast_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_train: pd.Series,
    under_weight: float,
    high_threshold: float,
    occurrence_score: Optional[pd.Series] = None,
    occurrence_threshold: float = 0.50,
) -> Dict[str, float]:
    y_true = y_true.astype(float)
    y_pred = pd.Series(y_pred, index=y_true.index).astype(float).clip(lower=0)
    out = {
        "n": int(len(y_true)),
        "mean_actual": float(y_true.mean()),
        "mean_pred": float(y_pred.mean()),
        "zero_rate_actual": float((y_true == 0).mean()),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "weighted_mae": weighted_mae(y_true.values, y_pred.values, under_weight),
        "mase": safe_mase(y_true.values, y_pred.values, y_train.values),
        "bias": float(np.mean(y_pred.values - y_true.values)),
        "underprediction_rate": float((y_pred.values < y_true.values).mean()),
        "overprediction_rate": float((y_pred.values > y_true.values).mean()),
    }
    # High-demand event detection: score can be forecast quantity.
    true_high = (y_true.values >= high_threshold).astype(int)
    pred_high = (y_pred.values >= high_threshold).astype(int)
    out.update({
        "high_precision": float(precision_score(true_high, pred_high, zero_division=0)),
        "high_recall": float(recall_score(true_high, pred_high, zero_division=0)),
        "high_f1": float(f1_score(true_high, pred_high, zero_division=0)),
    })
    # Occurrence detection for hurdle-like interpretation.
    occ_score = occurrence_score if occurrence_score is not None else y_pred
    out.update(occurrence_metrics(y_true.values, np.asarray(occ_score), occurrence_threshold))
    return out


def tune_hurdle_threshold(
    model: HurdleModel,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    threshold_grid: Sequence[float],
    under_weight: float,
) -> Tuple[float, pd.DataFrame]:
    rows = []
    for th in threshold_grid:
        pred = pd.Series(model.predict_thresholded(X_val, threshold=th), index=y_val.index)
        p = pd.Series(model.occurrence_probability(X_val), index=y_val.index)
        row = {"threshold": th}
        row.update(forecast_metrics(
            y_true=y_val,
            y_pred=pred,
            y_train=y_val,  # unused for threshold choice except MASE placeholder
            under_weight=under_weight,
            high_threshold=float(y_val.quantile(0.90)),
            occurrence_score=p,
            occurrence_threshold=th,
        ))
        rows.append(row)
    df = pd.DataFrame(rows)
    best = df.sort_values(["weighted_mae", "mae", "occurrence_f1"], ascending=[True, True, False]).iloc[0]
    return float(best["threshold"]), df


# ============================================================
# 7. Prediction pipeline
# ============================================================

_FEATURE_CACHE: Dict[int, pd.DataFrame] = {}


def split_X_y_from_all(
    X_all: pd.DataFrame,
    y_all: pd.Series,
    start: str,
    end: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    idx_mask = in_window(X_all.index, start, end)
    valid_mask = idx_mask & y_all.notna().values
    X = X_all.loc[valid_mask]
    y = y_all.loc[valid_mask].astype(float)
    return X, y


def fit_predict_models_for_target_horizon(
    panel: pd.DataFrame,
    targets: Sequence[str],
    target: str,
    horizon: int,
    config: SparseHurdleConfig,
    diag_row: Dict[str, object],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit general and sparse-aware candidates, return predictions and metrics for val/test."""
    # Build leakage-safe feature matrix once per horizon, then reuse for all targets.
    if horizon not in _FEATURE_CACHE:
        _FEATURE_CACHE[horizon] = build_feature_frame(panel, targets, horizon=horizon, windows=config.moving_windows)
    X_all = _FEATURE_CACHE[horizon]
    y_all = target_series_for_horizon(panel, target, horizon=horizon)
    X_train, y_train = split_X_y_from_all(X_all, y_all, config.train_start, config.train_end)
    X_val, y_val = split_X_y_from_all(X_all, y_all, config.val_start, config.val_end)
    X_test, y_test = split_X_y_from_all(X_all, y_all, config.test_start, config.test_end)

    if len(y_train) == 0 or len(y_val) == 0 or len(y_test) == 0:
        raise ValueError(f"Empty split for {target}, horizon={horizon}")

    high_threshold = float(y_train.quantile(config.high_demand_quantile))
    if high_threshold <= 0:
        high_threshold = float(max(1.0, y_train[y_train > 0].quantile(config.high_demand_quantile) if (y_train > 0).any() else 1.0))

    pred_rows: List[pd.DataFrame] = []
    metric_rows: List[Dict[str, object]] = []
    threshold_rows: List[pd.DataFrame] = []

    def add_predictions(split: str, model_name: str, y: pd.Series, pred: pd.Series, occ_score: Optional[pd.Series] = None, occ_threshold: float = 0.50):
        nonlocal pred_rows, metric_rows
        pred = pd.Series(pred, index=y.index).astype(float).clip(lower=0)
        rowdf = pd.DataFrame({
            "date": y.index,
            "split": split,
            "target": target,
            "horizon": horizon,
            "model": model_name,
            "actual": y.values,
            "pred": pred.values,
            "is_sparse_target": bool(diag_row.get("force_hurdle_candidates", False)),
        })
        if occ_score is not None:
            rowdf["occurrence_score"] = pd.Series(occ_score, index=y.index).values
            rowdf["occurrence_threshold"] = occ_threshold
        pred_rows.append(rowdf)
        m = forecast_metrics(
            y_true=y,
            y_pred=pred,
            y_train=y_train,
            under_weight=config.underprediction_weight,
            high_threshold=high_threshold,
            occurrence_score=occ_score,
            occurrence_threshold=occ_threshold,
        )
        metric_rows.append({
            "split": split,
            "target": target,
            "horizon": horizon,
            "model": model_name,
            "is_sparse_target": bool(diag_row.get("force_hurdle_candidates", False)),
            **m,
        })

    # Baselines and ordinary regression models.
    y_full = panel[target].astype(float)
    for split, y_eval in [("validation", y_val), ("test", y_test)]:
        # historical mean uses training mean only.
        add_predictions(split, "historical_mean", y_eval, pd.Series(y_train.mean(), index=y_eval.index))
        add_predictions(split, "seasonal_naive", y_eval, seasonal_naive_forecast(y_full, y_eval.index, horizon, config.seasonal_period))
        for w in config.moving_windows:
            add_predictions(split, f"moving_average_{w}d", y_eval, rolling_forecast_series(y_full, y_eval.index, horizon, w, kind="mean"))

    regression_models = {
        "lasso": make_lasso(alpha=0.05, random_state=config.random_state),
        "ridge": make_ridge(alpha=5.0),
    }
    reg_val_preds = {}
    reg_test_preds = {}
    for name, model in regression_models.items():
        fitted = clone(model).fit(X_train, y_train)
        val_pred = pd.Series(np.maximum(fitted.predict(X_val), 0.0), index=y_val.index)
        test_pred = pd.Series(np.maximum(fitted.predict(X_test), 0.0), index=y_test.index)
        reg_val_preds[name] = val_pred
        reg_test_preds[name] = test_pred
        add_predictions("validation", name, y_val, val_pred)
        add_predictions("test", name, y_test, test_pred)

    # Optional extended manuscript-comparison models: LightGBM, Prophet, and LSTM.
    # These are evaluated under exactly the same splits and metrics. If a package is
    # unavailable or a model fails for a sparse target, it is skipped and documented by the console log.
    if config.run_lightgbm:
        out = fit_predict_lightgbm(X_train, y_train, X_val, X_test, config)
        if out is not None:
            val_pred, test_pred = out
            add_predictions("validation", "lightgbm", y_val, val_pred)
            add_predictions("test", "lightgbm", y_test, test_pred)

    if config.run_prophet:
        out = fit_predict_prophet(X_train, y_train, X_val, X_test, target, horizon, config)
        if out is not None:
            val_pred, test_pred = out
            add_predictions("validation", "prophet_augmented", y_val, val_pred)
            add_predictions("test", "prophet_augmented", y_test, test_pred)

    if config.run_lstm:
        out = fit_predict_lstm(X_train, y_train, X_val, y_val, X_test, y_test, config)
        if out is not None:
            val_pred, test_pred = out
            add_predictions("validation", "lstm", y_val, val_pred)
            add_predictions("test", "lstm", y_test, test_pred)

    # Blends combine feature-based model with adaptive rolling behavior.
    for base_name in ["lasso"]:
        for w in [7, 14]:
            val_ma = rolling_forecast_series(y_full, y_val.index, horizon, w, kind="mean")
            test_ma = rolling_forecast_series(y_full, y_test.index, horizon, w, kind="mean")
            val_pred = 0.5 * reg_val_preds[base_name] + 0.5 * val_ma
            test_pred = 0.5 * reg_test_preds[base_name] + 0.5 * test_ma
            model_name = f"blend_{base_name}_ma{w}_50_50"
            add_predictions("validation", model_name, y_val, val_pred)
            add_predictions("test", model_name, y_test, test_pred)

    # Hurdle candidates are required for sparse/intermittent targets, but also evaluated for all targets
    # so that the validation process can decide whether they help.
    positive_days = int((y_train > 0).sum())
    force_hurdle = bool(diag_row.get("force_hurdle_candidates", False))
    if force_hurdle:
        # Hurdle LASSO: logistic occurrence + LASSO positive quantity.
        hurdle_lasso = HurdleModel(
            occurrence_model=make_logistic(random_state=config.random_state),
            quantity_model=make_lasso(alpha=0.05, random_state=config.random_state),
            min_positive_days=config.min_positive_days_for_hurdle,
            random_state=config.random_state,
        ).fit(X_train, y_train)
        best_th, th_df = tune_hurdle_threshold(
            hurdle_lasso, X_val, y_val,
            config.occurrence_threshold_grid,
            config.underprediction_weight,
        )
        hurdle_lasso.threshold = best_th
        th_df.insert(0, "target", target)
        th_df.insert(1, "horizon", horizon)
        th_df.insert(2, "model", "hurdle_lasso")
        threshold_rows.append(th_df)

        for mode in ["soft", "thresholded"]:
            model_name = f"hurdle_lasso_{mode}"
            if mode == "soft":
                val_pred = pd.Series(hurdle_lasso.predict_soft(X_val), index=y_val.index)
                test_pred = pd.Series(hurdle_lasso.predict_soft(X_test), index=y_test.index)
            else:
                val_pred = pd.Series(hurdle_lasso.predict_thresholded(X_val), index=y_val.index)
                test_pred = pd.Series(hurdle_lasso.predict_thresholded(X_test), index=y_test.index)
            val_prob = pd.Series(hurdle_lasso.occurrence_probability(X_val), index=y_val.index)
            test_prob = pd.Series(hurdle_lasso.occurrence_probability(X_test), index=y_test.index)
            add_predictions("validation", model_name, y_val, val_pred, occ_score=val_prob, occ_threshold=best_th)
            add_predictions("test", model_name, y_test, test_pred, occ_score=test_prob, occ_threshold=best_th)

        # Hurdle rolling: rolling nonzero probability × rolling positive mean.
        for w in [14]:
            for split, y_eval in [("validation", y_val), ("test", y_test)]:
                p = rolling_forecast_series(y_full, y_eval.index, horizon, w, kind="nonzero_rate")
                q = rolling_forecast_series(y_full, y_eval.index, horizon, w, kind="positive_mean")
                # Soft expected demand and thresholded version.  Use validation-tuned threshold based on grid.
                soft_pred = p * q
                add_predictions(split, f"hurdle_rolling{w}_soft", y_eval, soft_pred, occ_score=p, occ_threshold=0.50)
            # Tune rolling threshold on validation.
            p_val = rolling_forecast_series(y_full, y_val.index, horizon, w, kind="nonzero_rate")
            q_val = rolling_forecast_series(y_full, y_val.index, horizon, w, kind="positive_mean")
            rows = []
            for th in config.occurrence_threshold_grid:
                pred = pd.Series(np.where(p_val >= th, q_val, 0.0), index=y_val.index)
                m = forecast_metrics(y_val, pred, y_train, config.underprediction_weight, high_threshold, p_val, th)
                rows.append({"threshold": th, **m})
            roll_th_df = pd.DataFrame(rows)
            best_roll_th = float(roll_th_df.sort_values(["weighted_mae", "mae", "occurrence_f1"], ascending=[True, True, False]).iloc[0]["threshold"])
            roll_th_df.insert(0, "target", target)
            roll_th_df.insert(1, "horizon", horizon)
            roll_th_df.insert(2, "model", f"hurdle_rolling{w}_thresholded")
            threshold_rows.append(roll_th_df)
            for split, y_eval in [("validation", y_val), ("test", y_test)]:
                p = rolling_forecast_series(y_full, y_eval.index, horizon, w, kind="nonzero_rate")
                q = rolling_forecast_series(y_full, y_eval.index, horizon, w, kind="positive_mean")
                pred = pd.Series(np.where(p >= best_roll_th, q, 0.0), index=y_eval.index)
                add_predictions(split, f"hurdle_rolling{w}_thresholded", y_eval, pred, occ_score=p, occ_threshold=best_roll_th)

    preds = pd.concat(pred_rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    thresholds = pd.concat(threshold_rows, ignore_index=True) if threshold_rows else pd.DataFrame()
    return preds, metrics, thresholds


def run_forecasting(panel: pd.DataFrame, targets: Sequence[str], diag: pd.DataFrame, config: SparseHurdleConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_preds = []
    all_metrics = []
    all_thresholds = []
    diag_map = diag.set_index("target").to_dict(orient="index")
    for h in range(1, config.max_horizon + 1):
        print(f"[forecast] horizon={h}")
        for target in targets:
            preds, metrics, thresholds = fit_predict_models_for_target_horizon(
                panel=panel,
                targets=targets,
                target=target,
                horizon=h,
                config=config,
                diag_row=diag_map[target],
            )
            all_preds.append(preds)
            all_metrics.append(metrics)
            if len(thresholds):
                all_thresholds.append(thresholds)
    pred_df = pd.concat(all_preds, ignore_index=True)
    metric_df = pd.concat(all_metrics, ignore_index=True)
    threshold_df = pd.concat(all_thresholds, ignore_index=True) if all_thresholds else pd.DataFrame()
    return pred_df, metric_df, threshold_df


# ============================================================
# 8. Validation model selection and inventory simulation
# ============================================================

def select_models_by_validation(metrics: pd.DataFrame, config: SparseHurdleConfig) -> pd.DataFrame:
    """Select forecasting model per target/horizon using validation weighted MAE.

    For sparse targets, hurdle models are not automatically forced to win; they must improve validation.
    But the selection table keeps a flag showing whether hurdle/two-stage candidates were required.
    """
    val = metrics[metrics["split"] == "validation"].copy()
    rows = []
    for (target, horizon), g in val.groupby(["target", "horizon"]):
        g2 = g.sort_values(["weighted_mae", "mae", "occurrence_f1"], ascending=[True, True, False])
        best = g2.iloc[0].to_dict()
        best["selection_metric"] = "validation_weighted_mae"
        best["ranked_candidate_count"] = len(g2)
        rows.append(best)
    return pd.DataFrame(rows)


def build_selected_prediction_table(preds: pd.DataFrame, selected: pd.DataFrame, split: str) -> pd.DataFrame:
    sel = selected[["target", "horizon", "model"]].rename(columns={"model": "selected_model"})
    df = preds[preds["split"] == split].merge(sel, on=["target", "horizon"], how="inner")
    df = df[df["model"] == df["selected_model"]].copy()
    return df.drop(columns=["selected_model"])


def prediction_lookup(selected_preds: pd.DataFrame) -> Dict[Tuple[str, int], pd.Series]:
    out: Dict[Tuple[str, int], pd.Series] = {}
    for (target, h), g in selected_preds.groupby(["target", "horizon"]):
        s = pd.Series(g["pred"].values, index=pd.to_datetime(g["date"])).sort_index()
        out[(target, int(h))] = s
    return out


def expected_demand_window(preds_by_horizon: Dict[Tuple[str, int], pd.Series], target: str, date: pd.Timestamp, target_days_supply: float) -> float:
    full_days = int(math.floor(target_days_supply))
    frac = target_days_supply - full_days
    total = 0.0
    for h in range(1, full_days + 1):
        s = preds_by_horizon.get((target, h))
        if s is not None and date in s.index:
            total += float(s.loc[date])
    if frac > 1e-9:
        h = full_days + 1
        s = preds_by_horizon.get((target, h))
        if s is not None and date in s.index:
            total += frac * float(s.loc[date])
    return max(total, 0.0)


def simulate_inventory_for_target(
    actual: pd.Series,
    preds_by_horizon: Dict[Tuple[str, int], pd.Series],
    target: str,
    target_days_supply: float,
    safety_factor: float,
    config: SparseHurdleConfig,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    actual = actual.astype(float).sort_index()
    inv_by_age = [0.0 for _ in range(config.shelf_life_days)]
    first_forecast = expected_demand_window(preds_by_horizon, target, actual.index[0], target_days_supply)
    init_units = max(config.initial_days_supply * (first_forecast / max(target_days_supply, 1e-6)), actual.mean() * config.initial_days_supply)
    inv_by_age[0] = float(init_units)
    pending_orders: List[Tuple[pd.Timestamp, float]] = []
    rows = []

    for date, demand in actual.items():
        # Receive orders arriving today.
        arrivals = 0.0
        remaining_pending = []
        for arrival_date, qty in pending_orders:
            if arrival_date <= date:
                arrivals += qty
            else:
                remaining_pending.append((arrival_date, qty))
        pending_orders = remaining_pending

        # Age inventory and add arrivals as age 0.
        expired = inv_by_age[-1]
        inv_by_age = [arrivals] + inv_by_age[:-1]

        # Fulfill oldest first.
        need = float(demand)
        fulfilled = 0.0
        for age in reversed(range(len(inv_by_age))):
            take = min(inv_by_age[age], need)
            inv_by_age[age] -= take
            need -= take
            fulfilled += take
            if need <= 1e-9:
                break
        unmet = max(need, 0.0)

        on_hand = sum(inv_by_age)
        already_ordered = sum(qty for _, qty in pending_orders)
        desired_position = safety_factor * expected_demand_window(preds_by_horizon, target, date, target_days_supply)
        order_qty = max(0.0, round(desired_position - (on_hand + already_ordered)))
        if order_qty > 0:
            pending_orders.append((date + pd.Timedelta(days=config.lead_time_days), order_qty))

        rows.append({
            "date": date,
            "target": target,
            "actual_demand": float(demand),
            "fulfilled": float(fulfilled),
            "unmet": float(unmet),
            "wasted": float(expired),
            "procured_ordered": float(order_qty),
            "on_hand_end": float(sum(inv_by_age)),
            "target_days_supply": target_days_supply,
            "safety_factor": safety_factor,
        })

    trace = pd.DataFrame(rows)
    demand_sum = trace["actual_demand"].sum()
    procured = trace["procured_ordered"].sum()
    fulfilled = trace["fulfilled"].sum()
    unmet = trace["unmet"].sum()
    wasted = trace["wasted"].sum()
    summary = {
        "target": target,
        "demand": float(demand_sum),
        "procured": float(procured),
        "fulfilled": float(fulfilled),
        "unmet": float(unmet),
        "wasted": float(wasted),
        "service_level": float(fulfilled / demand_sum) if demand_sum > 0 else np.nan,
        "wastage_rate": float(wasted / procured) if procured > 0 else 0.0,
        "cost": float(config.shortage_cost * unmet + config.wastage_cost * wasted + config.procurement_cost * procured),
        "target_days_supply": target_days_supply,
        "safety_factor": safety_factor,
    }
    return summary, trace


def run_inventory_grid(
    panel: pd.DataFrame,
    targets: Sequence[str],
    selected_preds: pd.DataFrame,
    config: SparseHurdleConfig,
    split: str,
    start: str,
    end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    preds_by_h = prediction_lookup(selected_preds[selected_preds["split"] == split])
    actual_period = panel.loc[in_window(panel.index, start, end), list(targets)]
    summaries = []
    traces = []
    for t in targets:
        actual = actual_period[t]
        for days in config.target_days_supply_grid:
            for sf in config.safety_factor_grid:
                summary, trace = simulate_inventory_for_target(actual, preds_by_h, t, days, sf, config)
                summaries.append(summary)
                trace = trace.copy()
                trace["split"] = split
                traces.append(trace)
    return pd.DataFrame(summaries), pd.concat(traces, ignore_index=True)


def select_inventory_policy(validation_grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, g in validation_grid.groupby("target"):
        best = g.sort_values(["cost", "unmet", "wasted"], ascending=[True, True, True]).iloc[0].to_dict()
        best["selection_basis"] = "validation_min_cost"
        rows.append(best)
    return pd.DataFrame(rows)


def evaluate_selected_inventory(
    panel: pd.DataFrame,
    targets: Sequence[str],
    selected_preds: pd.DataFrame,
    selected_policy: pd.DataFrame,
    config: SparseHurdleConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    preds_by_h = prediction_lookup(selected_preds[selected_preds["split"] == "test"])
    actual_test = panel.loc[in_window(panel.index, config.test_start, config.test_end), list(targets)]
    policy_map = selected_policy.set_index("target").to_dict(orient="index")
    summaries = []
    traces = []
    for t in targets:
        pol = policy_map[t]
        summary, trace = simulate_inventory_for_target(
            actual=actual_test[t],
            preds_by_horizon=preds_by_h,
            target=t,
            target_days_supply=float(pol["target_days_supply"]),
            safety_factor=float(pol["safety_factor"]),
            config=config,
        )
        summary["selected_on_validation"] = True
        summaries.append(summary)
        trace["split"] = "test"
        traces.append(trace)
    target_summary = pd.DataFrame(summaries)
    target_summary["product"] = np.where(target_summary["target"].str.startswith("aph_"), "APC", "PC")
    product_summary = target_summary.groupby("product", as_index=False).agg({
        "demand": "sum",
        "procured": "sum",
        "fulfilled": "sum",
        "unmet": "sum",
        "wasted": "sum",
        "cost": "sum",
    })
    product_summary["service_level"] = product_summary["fulfilled"] / product_summary["demand"]
    product_summary["wastage_rate"] = np.where(product_summary["procured"] > 0, product_summary["wasted"] / product_summary["procured"], 0.0)
    return target_summary, product_summary, pd.concat(traces, ignore_index=True)


# ============================================================
# 9. Reporting
# ============================================================

def summarize_models(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.groupby(["split", "horizon", "model"], as_index=False).agg(
        mean_mae=("mae", "mean"),
        mean_weighted_mae=("weighted_mae", "mean"),
        mean_mase=("mase", "mean"),
        mean_bias=("bias", "mean"),
        mean_occurrence_f1=("occurrence_f1", "mean"),
        mean_high_f1=("high_f1", "mean"),
        n_targets=("target", "nunique"),
    ).sort_values(["split", "horizon", "mean_weighted_mae", "mean_mae"])


def paired_tests_vs_lasso(metrics: pd.DataFrame, split: str = "test", horizon: int = 1) -> pd.DataFrame:
    g = metrics[(metrics["split"] == split) & (metrics["horizon"] == horizon)].copy()
    rows = []
    lasso = g[g["model"] == "lasso"]
    if lasso.empty:
        return pd.DataFrame()
    for metric in ["mae", "weighted_mae"]:
        lasso_map = lasso.set_index("target")[metric]
        for model, gm in g.groupby("model"):
            if model == "lasso":
                continue
            comp_map = gm.set_index("target")[metric]
            common = sorted(set(lasso_map.index) & set(comp_map.index))
            if len(common) < 3:
                pval = np.nan
                diff = np.nan
            else:
                d = lasso_map.loc[common].values - comp_map.loc[common].values
                diff = float(np.mean(d))
                try:
                    pval = float(wilcoxon(lasso_map.loc[common], comp_map.loc[common]).pvalue)
                except Exception:
                    pval = np.nan
            rows.append({
                "split": split,
                "horizon": horizon,
                "metric": metric,
                "comparison": f"lasso vs {model}",
                "n_common_targets": len(common),
                "lasso_mean": float(lasso_map.loc[common].mean()) if common else np.nan,
                "comparator_mean": float(comp_map.loc[common].mean()) if common else np.nan,
                "mean_difference_lasso_minus_comparator": diff,
                "p_value": pval,
                "interpretation": "LASSO significantly better" if np.isfinite(pval) and pval < 0.05 and diff < 0 else "No significant LASSO advantage",
            })
    return pd.DataFrame(rows)


def make_decision_table(diag: pd.DataFrame, selected_models: pd.DataFrame, selected_inventory: pd.DataFrame, test_metrics: pd.DataFrame) -> pd.DataFrame:
    h1_sel = selected_models[selected_models["horizon"] == 1][["target", "model", "weighted_mae", "mae", "occurrence_f1"]].rename(
        columns={"model": "selected_h1_model", "weighted_mae": "val_weighted_mae", "mae": "val_mae", "occurrence_f1": "val_occurrence_f1"}
    )
    test_best = test_metrics[(test_metrics["split"] == "test") & (test_metrics["horizon"] == 1)].copy()
    test_best = test_best.sort_values(["target", "weighted_mae", "mae"]).groupby("target", as_index=False).first()
    test_best = test_best[["target", "model", "weighted_mae", "mae", "occurrence_f1"]].rename(
        columns={"model": "best_test_h1_model", "weighted_mae": "best_test_weighted_mae", "mae": "best_test_mae", "occurrence_f1": "best_test_occurrence_f1"}
    )
    inv = selected_inventory[["target", "target_days_supply", "safety_factor", "cost", "service_level", "wastage_rate"]].rename(
        columns={"cost": "val_inventory_cost", "service_level": "val_service_level", "wastage_rate": "val_wastage_rate"}
    )
    out = diag.merge(h1_sel, on="target", how="left").merge(test_best, on="target", how="left").merge(inv, on="target", how="left")
    out["final_method_comment"] = np.where(
        out["force_hurdle_candidates"],
        "Sparse/intermittent: interpret general regression cautiously; prefer validated hurdle/blend/adaptive policy.",
        "Non-sparse: general regression acceptable, but adaptive baselines remain required comparators.",
    )
    return out


def write_caveats(path: Path, config: SparseHurdleConfig) -> None:
    txt = f"""Sparse-aware platelet forecasting methodological notes
=========================================================

1. Demand definition
   The outcome is observed issued-unit demand. It is not unobserved true clinical demand.
   Service level therefore means fulfillment of observed operational demand.

2. Why sparse targets require hurdle/two-stage candidates
   Targets with high zero-demand rate or intermittent/lumpy demand patterns are unstable under
   ordinary regression alone. A single regression model mixes occurrence and quantity. This script
   therefore compares general regression with hurdle/two-stage candidates:
      occurrence model: P(y > 0)
      quantity model: E(y | y > 0)
      final forecast: soft P × quantity or thresholded occurrence × quantity

3. Sparse classification
   Sparse/intermittent targets are flagged by zero-rate >= {config.sparse_zero_rate_threshold},
   ADI >= {config.intermittent_adi_threshold}, or Syntetos-Boylan class of intermittent/lumpy.

4. Validation-based selection
   Hurdle models are not automatically declared superior. They must improve validation performance.
   The final test period is used only for evaluation.

5. Optional extended models
   LightGBM, Prophet, and LSTM are included as candidate comparators when enabled. They are evaluated
   under the same train/validation/test split, weighted MAE, occurrence metrics, and inventory-policy
   selection rules as the sparse-aware baseline models.

6. Cost-sensitive objective
   Underprediction receives {config.underprediction_weight} times the error weight of overprediction.
   Inventory cost is {config.shortage_cost} × unmet + {config.wastage_cost} × wasted + {config.procurement_cost} × procured.

7. Manuscript interpretation
   If hurdle/two-stage models improve sparse targets, report this as evidence that product-ABO-specific
   modeling is required. If they do not improve a sparse target, still state that general regression alone
   is unstable and must be benchmarked against sparse-aware alternatives.
"""
    path.write_text(txt, encoding="utf-8")


# ============================================================
# 10. Main runner
# ============================================================

def run_analysis(config: SparseHurdleConfig) -> Dict[str, pd.DataFrame]:
    outdir = ensure_dir(config.output_dir)
    print("=" * 80)
    print("Sparse-aware platelet forecasting analysis with LightGBM, Prophet, and LSTM")
    print("Data:", config.data_path)
    print("Output:", outdir)
    print("=" * 80)

    panel, metadata = load_platelet_excel(config.data_path)
    targets = get_target_columns(config, panel)
    print(f"Panel shape: {panel.shape}; targets: {targets}")

    audit = []
    for c in panel.columns:
        s = panel[c]
        audit.append({
            "variable": c,
            "is_target": c in targets,
            "n_days": len(s),
            "missing_n": int(s.isna().sum()),
            "zero_rate": float((s == 0).mean()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
        })
    audit_df = pd.DataFrame(audit)
    diag = sparse_diagnostics(panel, targets, config)

    audit_df.to_csv(outdir / "00_data_audit.csv", index=False)
    diag.to_csv(outdir / "01_sparse_intermittent_diagnostics.csv", index=False)

    preds, metrics, thresholds = run_forecasting(panel, targets, diag, config)
    preds.to_csv(outdir / "02_all_predictions_long.csv", index=False)
    metrics.to_csv(outdir / "03_all_metrics_by_target.csv", index=False)
    thresholds.to_csv(outdir / "04_hurdle_threshold_tuning.csv", index=False)

    summary = summarize_models(metrics)
    summary.to_csv(outdir / "05_model_summary_by_split_horizon.csv", index=False)

    selected_models = select_models_by_validation(metrics, config)
    selected_models.to_csv(outdir / "06_selected_forecast_model_by_validation.csv", index=False)

    selected_preds = build_selected_prediction_table(preds, selected_models, split="validation")
    selected_preds_test = build_selected_prediction_table(preds, selected_models, split="test")
    selected_all = pd.concat([selected_preds, selected_preds_test], ignore_index=True)
    selected_all.to_csv(outdir / "07_selected_predictions_long.csv", index=False)

    paired = paired_tests_vs_lasso(metrics, split="test", horizon=1)
    paired.to_csv(outdir / "08_paired_tests_vs_lasso_h1.csv", index=False)

    # Inventory grid based on selected forecasting model per target/horizon.
    val_grid, val_trace = run_inventory_grid(panel, targets, selected_all, config, split="validation", start=config.val_start, end=config.val_end)
    val_grid.to_csv(outdir / "09_validation_inventory_grid_selected_forecasts.csv", index=False)
    selected_inventory = select_inventory_policy(val_grid)
    selected_inventory.to_csv(outdir / "10_selected_inventory_policy_by_validation.csv", index=False)

    test_target_inv, test_product_inv, test_trace = evaluate_selected_inventory(panel, targets, selected_all, selected_inventory, config)
    test_target_inv.to_csv(outdir / "11_test_inventory_target_level.csv", index=False)
    test_product_inv.to_csv(outdir / "12_test_inventory_product_level.csv", index=False)
    test_trace.to_csv(outdir / "13_test_inventory_daily_trace.csv", index=False)

    decision = make_decision_table(diag, selected_models, selected_inventory, metrics)
    decision.to_csv(outdir / "14_sparse_aware_target_decision_table.csv", index=False)

    # Excel workbook for convenient review.
    with pd.ExcelWriter(outdir / "sparse_hurdle_analysis_summary.xlsx", engine="openpyxl") as writer:
        audit_df.to_excel(writer, sheet_name="data_audit", index=False)
        diag.to_excel(writer, sheet_name="sparse_diagnostics", index=False)
        summary.to_excel(writer, sheet_name="model_summary", index=False)
        selected_models.to_excel(writer, sheet_name="selected_models", index=False)
        paired.to_excel(writer, sheet_name="paired_tests", index=False)
        thresholds.to_excel(writer, sheet_name="hurdle_thresholds", index=False)
        selected_inventory.to_excel(writer, sheet_name="selected_inventory", index=False)
        test_target_inv.to_excel(writer, sheet_name="test_inventory_target", index=False)
        test_product_inv.to_excel(writer, sheet_name="test_inventory_product", index=False)
        decision.to_excel(writer, sheet_name="decision_table", index=False)

    write_caveats(outdir / "methodological_caveats_sparse_hurdle.txt", config)
    (outdir / "analysis_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    print("\nSaved outputs to:", outdir)
    print("Key result files:")
    for name in [
        "01_sparse_intermittent_diagnostics.csv",
        "05_model_summary_by_split_horizon.csv",
        "06_selected_forecast_model_by_validation.csv",
        "10_selected_inventory_policy_by_validation.csv",
        "11_test_inventory_target_level.csv",
        "12_test_inventory_product_level.csv",
        "14_sparse_aware_target_decision_table.csv",
        "sparse_hurdle_analysis_summary.xlsx",
    ]:
        print(" -", name)

    return {
        "panel": panel,
        "metadata": metadata,
        "audit": audit_df,
        "sparse_diagnostics": diag,
        "predictions": preds,
        "metrics": metrics,
        "thresholds": thresholds,
        "summary": summary,
        "selected_models": selected_models,
        "paired_tests": paired,
        "selected_inventory": selected_inventory,
        "test_inventory_target": test_target_inv,
        "test_inventory_product": test_product_inv,
        "decision_table": decision,
    }



# ============================================================
# PART 2. Manuscript Figure 1 and Figure 2 generation
#
# These functions read the raw panel directly (via
# load_daily_panel_and_metadata, independent from Part 1's
# load_platelet_excel) and the Part 1 forecasting outputs, then
# reproduce the manuscript figures.
# ============================================================


# =========================================================
# 1. Config
# =========================================================
@dataclass
class FigureConfig:
    data_path: str
    output_dir: str

    # 최신 sparse/hurdle-aware(+extended models) 결과를 반영하려면 selected_predictions 사용
    selected_predictions_csv: str | None = None

    # 특정 모델(LASSO, Prophet 등)을 직접 Figure 2에 반영하고 싶을 때 사용
    all_predictions_csv: str | None = None
    figure2_model_name: str | None = None   # 예: "lasso", "prophet_augmented"

    # split dates
    train_start: str = "2019-03-01"
    train_end: str = "2024-02-29"
    val_start: str = "2024-03-01"
    val_end: str = "2024-08-31"
    test_start: str = "2024-09-01"
    test_end: str = "2025-02-28"

    # Figure 1 baseline for panel (C)
    baseline_start: str = "2019-03-01"
    baseline_end: str = "2019-06-30"

    # Figure 2 visible window
    recent_start: str = "2023-01-01"

    rolling_window: int = 30
    forecast_label: str = "Validation-selected forecast"

    # 자동 탐색이 실패하면 직접 넣을 수 있음
    hem_onc_cols_manual: list | None = None
    ga_cols_manual: list | None = None


# Note: renamed from META_COLS to avoid colliding with the differently-shaped
# META_COLS used by load_platelet_excel() in the forecasting pipeline above.
FIGURE_META_COLS = ["category", "variable", "cond1", "cond2", "description", "cond3"]

ABO_COLORS = {
    "A": "#1f77b4",   # blue
    "B": "#2ca02c",   # green
    "O": "#d62728",   # red
    "AB": "#9467bd",  # purple
}

PC_TARGETS = ["plt_transf_a", "plt_transf_b", "plt_transf_o", "plt_transf_ab"]
APC_TARGETS = ["aph_transf_a", "aph_transf_b", "aph_transf_o", "aph_transf_ab"]


# =========================================================
# 2. Data loader
# =========================================================
def _normalize_variable_name(name: str) -> str:
    name = str(name)
    if name.startswith("PLT_"):
        name = "plt_" + name[len("PLT_"):]
    if name.endswith("_O"):
        name = name[:-2] + "_o"
    return name


def _merge_aph_products(panel: pd.DataFrame) -> pd.DataFrame:
    """
    aph22_* + aph23_* -> aph_* 통합 컬럼 생성
    (원본은 유지하고 aph_*만 추가)
    """
    panel = panel.copy()
    flows = ["income", "transf", "transfused", "return", "expire"]
    abos = ["a", "b", "o", "ab"]

    for flow in flows:
        for abo in abos:
            c22 = f"aph22_{flow}_{abo}"
            c23 = f"aph23_{flow}_{abo}"
            new = f"aph_{flow}_{abo}"
            if new not in panel.columns and c22 in panel.columns and c23 in panel.columns:
                panel[new] = panel[c22].fillna(0) + panel[c23].fillna(0)

    for abo in abos:
        c22 = f"past_1wk_trans_aph22_{abo}"
        c23 = f"past_1wk_trans_aph23_{abo}"
        new = f"past_1wk_trans_aph_{abo}"
        if new not in panel.columns and c22 in panel.columns and c23 in panel.columns:
            panel[new] = panel[c22].fillna(0) + panel[c23].fillna(0)

    return panel


def load_daily_panel_and_metadata(xlsx_path: str):
    xlsx_path = Path(xlsx_path)
    df = pd.read_excel(xlsx_path, sheet_name=0)

    if len(df.columns) < 6:
        raise ValueError("Expected at least 6 metadata columns in the Excel file.")

    rename_map = dict(zip(df.columns[:6], FIGURE_META_COLS))
    df = df.rename(columns=rename_map)

    meta = df[FIGURE_META_COLS].copy()
    meta["variable"] = meta["variable"].astype(str).map(_normalize_variable_name)

    date_cols = [c for c in df.columns if c not in FIGURE_META_COLS]
    panel = df[date_cols].T
    panel.columns = meta["variable"].tolist()
    panel.index = pd.to_datetime(panel.index)
    panel.index.name = "date"
    panel = panel.sort_index()

    for c in panel.columns:
        panel[c] = pd.to_numeric(panel[c], errors="coerce")

    panel = _merge_aph_products(panel)
    return panel, meta


# =========================================================
# 3. Metadata search helpers
# =========================================================
def _meta_text(meta: pd.DataFrame) -> pd.Series:
    txt = (
        meta[["category", "variable", "cond1", "cond2", "description", "cond3"]]
        .fillna("")
        .astype(str)
        .agg(" | ".join, axis=1)
        .str.lower()
    )
    return txt


def find_columns_by_keyword_groups(meta: pd.DataFrame, keyword_groups: list[list[str]]) -> list[str]:
    """
    keyword_groups = [
        ["hematology", "oncology", "inpatient"],
        ["hemato", "oncology", "ward"],
    ]
    각 group 안의 키워드는 모두 포함되어야 함.
    여러 group에서 찾은 결과를 union.
    """
    txt = _meta_text(meta)
    selected = set()

    for group in keyword_groups:
        mask = pd.Series(True, index=meta.index)
        for kw in group:
            mask &= txt.str.contains(str(kw).lower(), regex=False)
        selected.update(meta.loc[mask, "variable"].tolist())

    return sorted(selected)


def auto_detect_hem_onc_cols(meta: pd.DataFrame) -> list[str]:
    patterns = [
        ["hematology", "oncology", "inpatient"],
        ["hematology-oncology", "inpatient"],
        ["hemato", "oncology", "inpatient"],
        ["hematology", "oncology", "ward"],
        ["hemato", "oncology", "ward"],
    ]
    return find_columns_by_keyword_groups(meta, patterns)


def auto_detect_general_anesthesia_cols(meta: pd.DataFrame) -> list[str]:
    patterns = [
        ["general", "anesthesia"],
        ["general anaesthesia"],
        ["ga", "surgery"],
    ]
    return find_columns_by_keyword_groups(meta, patterns)


def pick_existing(panel: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in panel.columns]


def print_candidate_columns(meta: pd.DataFrame, keyword: str, top_n: int = 30):
    txt = _meta_text(meta)
    cand = meta.loc[txt.str.contains(keyword.lower(), regex=False), ["variable", "description", "category"]]
    print(f"\n[Candidates for keyword='{keyword}']")
    print(cand.head(top_n).to_string(index=False))


# =========================================================
# 4. Prediction loader
# =========================================================
def standardize_prediction_columns(pred: pd.DataFrame) -> pd.DataFrame:
    pred = pred.copy()
    pred.columns = [str(c).strip().lower() for c in pred.columns]

    alias_map = {
        "date": ["date", "ds"],
        "target": ["target", "series", "variable"],
        "actual": ["actual", "y_true", "truth", "y"],
        "pred": ["pred", "prediction", "y_pred", "forecast"],
        "model": ["model", "forecast_model", "selected_model"],
        "horizon": ["horizon", "fh"],
        "split": ["split"],
    }

    rename = {}
    for std_col, aliases in alias_map.items():
        found = [c for c in aliases if c in pred.columns]
        if found:
            rename[found[0]] = std_col

    pred = pred.rename(columns=rename)

    required = ["date", "target", "pred"]
    missing = [c for c in required if c not in pred.columns]
    if missing:
        raise ValueError(f"Prediction file is missing required columns: {missing}")

    pred["date"] = pd.to_datetime(pred["date"])
    if "target" in pred.columns:
        pred["target"] = pred["target"].astype(str)
    if "model" in pred.columns:
        pred["model"] = pred["model"].astype(str)
    return pred


def load_predictions(
    selected_predictions_csv: str | None = None,
    all_predictions_csv: str | None = None,
    model_name: str | None = None,
    horizon: int = 1,
) -> pd.DataFrame | None:
    """
    우선순위:
      1) selected_predictions_csv 사용
      2) all_predictions_csv + model_name 사용
    """
    if selected_predictions_csv:
        pred = pd.read_csv(selected_predictions_csv)
        pred = standardize_prediction_columns(pred)
        if "horizon" in pred.columns:
            pred = pred.loc[pred["horizon"] == horizon].copy()
        return pred

    if all_predictions_csv and model_name:
        pred = pd.read_csv(all_predictions_csv)
        pred = standardize_prediction_columns(pred)
        if "horizon" in pred.columns:
            pred = pred.loc[pred["horizon"] == horizon].copy()
        if "model" not in pred.columns:
            raise ValueError("all_predictions_csv exists, but 'model' column not found.")
        pred = pred.loc[pred["model"].str.lower() == model_name.lower()].copy()
        return pred

    return None


def aggregate_predictions_by_product(pred: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    """
    target별 예측을 product 단위로 합산
    반환 columns: [date, actual, pred]
    """
    sub = pred.loc[pred["target"].isin(targets)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["date", "actual", "pred"])

    agg_dict = {"pred": "sum"}
    if "actual" in sub.columns:
        agg_dict["actual"] = "sum"

    out = sub.groupby("date", as_index=False).agg(agg_dict)
    return out


# =========================================================
# 5. Figure 1
# =========================================================
def _plot_abo_panel(ax, panel, target_cols, panel_letter, ymax=None):
    for blood_type, col in zip(["A", "B", "O", "AB"], target_cols):
        if col not in panel.columns:
            warnings.warn(f"{col} not found. Skipping.")
            continue

        s = panel[col].fillna(0)
        rm = s.rolling(30, min_periods=1).mean()

        # faint daily series
        ax.vlines(s.index, 0, s.values, color=ABO_COLORS[blood_type], alpha=0.08, linewidth=0.6)
        # rolling mean
        ax.plot(rm.index, rm.values, color=ABO_COLORS[blood_type], lw=1.8, label=f"Type {blood_type}")

    ax.set_ylabel("Units/day")
    if ymax is not None:
        ax.set_ylim(0, ymax)
    ax.legend(loc="upper left", ncol=2, frameon=True, fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.text(-0.06, 0.98, f"({panel_letter})", transform=ax.transAxes,
            fontsize=22, va="top", ha="right")


def _relative_series(series: pd.Series, baseline_start: str, baseline_end: str, rolling_window: int = 30):
    rm = series.rolling(rolling_window, min_periods=1).mean()
    baseline = rm.loc[baseline_start:baseline_end].mean()
    if pd.isna(baseline) or baseline == 0:
        baseline = 1.0
    return rm / baseline


def plot_figure1_temporal_patterns(
    panel: pd.DataFrame,
    meta: pd.DataFrame,
    cfg: FigureConfig,
):
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hem_onc_cols = cfg.hem_onc_cols_manual or auto_detect_hem_onc_cols(meta)
    ga_cols = cfg.ga_cols_manual or auto_detect_general_anesthesia_cols(meta)

    hem_onc_cols = pick_existing(panel, hem_onc_cols)
    ga_cols = pick_existing(panel, ga_cols)

    if len(hem_onc_cols) == 0:
        print("[WARN] Hematology-oncology inpatient columns were not auto-detected.")
        print_candidate_columns(meta, "hematology")
        print_candidate_columns(meta, "oncology")
        raise ValueError(
            "No hematology-oncology inpatient columns found automatically. "
            "Please set cfg.hem_onc_cols_manual."
        )

    if len(ga_cols) == 0:
        print("[WARN] General anesthesia columns were not auto-detected.")
        print_candidate_columns(meta, "general")
        print_candidate_columns(meta, "anesthesia")
        raise ValueError(
            "No general anesthesia columns found automatically. "
            "Please set cfg.ga_cols_manual."
        )

    total_plt = panel[pick_existing(panel, PC_TARGETS + APC_TARGETS)].fillna(0).sum(axis=1)
    hem_onc = panel[hem_onc_cols].fillna(0).sum(axis=1)
    general_anesthesia = panel[ga_cols].fillna(0).sum(axis=1)

    rel_total = _relative_series(total_plt, cfg.baseline_start, cfg.baseline_end, cfg.rolling_window)
    rel_hem_onc = _relative_series(hem_onc, cfg.baseline_start, cfg.baseline_end, cfg.rolling_window)
    rel_ga = _relative_series(general_anesthesia, cfg.baseline_start, cfg.baseline_end, cfg.rolling_window)

    fig, axes = plt.subplots(
        3, 1, figsize=(11.5, 8.8), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 0.9], "hspace": 0.15}
    )

    _plot_abo_panel(axes[0], panel, PC_TARGETS, panel_letter="A")
    _plot_abo_panel(axes[1], panel, APC_TARGETS, panel_letter="B")

    axes[2].plot(rel_total.index, rel_total.values, color="#1f4aa8", lw=1.8, label="Total platelets issued")
    axes[2].plot(rel_hem_onc.index, rel_hem_onc.values, color="#8b1d1d", lw=1.8, label="Hematology-oncology inpatient")
    axes[2].plot(rel_ga.index, rel_ga.values, color="#1f7a1f", lw=1.8, label="General anesthesia cases")
    axes[2].axhline(1.0, color="gray", lw=1.0, ls=":")
    axes[2].set_ylabel("Relative")
    axes[2].set_xlabel("Date")
    axes[2].legend(loc="upper left", frameon=True, fontsize=9)
    axes[2].grid(True, alpha=0.25)
    axes[2].text(-0.06, 0.98, "(C)", transform=axes[2].transAxes,
                 fontsize=22, va="top", ha="right")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    outpath = output_dir / "Figure1_temporal_patterns.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()

    return outpath


# =========================================================
# 6. Figure 2
# =========================================================
def _plot_surge_panel(
    ax,
    product_name: str,
    actual_series: pd.Series,
    forecast_series: pd.Series | None,
    cfg: FigureConfig,
    panel_letter: str,
    color_bar: str,
    color_roll: str,
    forecast_color: str,
):
    recent = actual_series.loc[cfg.recent_start:cfg.test_end].copy()
    roll = recent.rolling(cfg.rolling_window, min_periods=1).mean()

    train_mean = actual_series.loc[cfg.train_start:cfg.train_end].mean()
    test_mean = actual_series.loc[cfg.test_start:cfg.test_end].mean()
    surge_ratio = (test_mean / train_mean) if train_mean > 0 else np.nan

    # background shading
    ax.axvspan(pd.to_datetime(cfg.recent_start), pd.to_datetime(cfg.test_start), color="#dbe6f3", alpha=0.35)
    ax.axvspan(pd.to_datetime(cfg.test_start), pd.to_datetime(cfg.test_end), color="#edd7d2", alpha=0.45)

    # actual daily demand
    ax.bar(recent.index, recent.values, width=1.0, color=color_bar, alpha=0.18, label="Actual daily demand", edgecolor="none")

    # 30-day rolling mean
    ax.plot(roll.index, roll.values, color=color_roll, lw=1.8, label=f"{cfg.rolling_window}-day rolling mean")

    # historical mean
    ax.axhline(train_mean, color="gray", lw=1.5, ls="--", label=f"Historical-mean ({train_mean:.1f} units/day)")

    # forecast (test period only)
    if forecast_series is not None and len(forecast_series) > 0:
        fc = forecast_series.loc[cfg.test_start:cfg.test_end].copy()
        ax.plot(fc.index, fc.values, color=forecast_color, lw=2.0, label=cfg.forecast_label)

    # test start marker
    ax.axvline(pd.to_datetime(cfg.test_start), color="black", lw=1.1, ls=":")

    # text
    ax.text(-0.06, 0.98, f"({panel_letter})", transform=ax.transAxes,
            fontsize=20, va="top", ha="right")

    if pd.notna(surge_ratio):
        y_min, y_max = ax.get_ylim()
        ax.text(
            pd.to_datetime(cfg.test_start) + pd.Timedelta(days=12),
            y_min + (y_max - y_min) * 0.08,
            f"×{surge_ratio:.1f} demand surge",
            fontsize=12,
            color="black",
        )

    ax.set_ylabel(f"Daily {product_name} issued")
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_figure2_demand_surge(
    panel: pd.DataFrame,
    cfg: FigureConfig,
    pred_df: pd.DataFrame | None = None,
):
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    actual_pc = panel[pick_existing(panel, PC_TARGETS)].fillna(0).sum(axis=1)
    actual_apc = panel[pick_existing(panel, APC_TARGETS)].fillna(0).sum(axis=1)

    pred_pc = None
    pred_apc = None

    if pred_df is not None:
        agg_pc = aggregate_predictions_by_product(pred_df, PC_TARGETS)
        agg_apc = aggregate_predictions_by_product(pred_df, APC_TARGETS)

        if not agg_pc.empty:
            pred_pc = agg_pc.set_index("date")["pred"].sort_index()
        if not agg_apc.empty:
            pred_apc = agg_apc.set_index("date")["pred"].sort_index()

    # 수정된 부분: hspace를 직접 넣지 않고 gridspec_kw 사용
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.5, 8.5),
        sharex=True,
        gridspec_kw={"hspace": 0.17}
    )

    _plot_surge_panel(
        ax=axes[0],
        product_name="PC",
        actual_series=actual_pc,
        forecast_series=pred_pc,
        cfg=cfg,
        panel_letter="A",
        color_bar="#86a9d7",
        color_roll="#2f6fab",
        forecast_color="#d62728",
    )

    _plot_surge_panel(
        ax=axes[1],
        product_name="APC",
        actual_series=actual_apc,
        forecast_series=pred_apc,
        cfg=cfg,
        panel_letter="B",
        color_bar="#e6a3a3",
        color_roll="#8b1d1d",
        forecast_color="#e83e8c",
    )

    # 각 panel별 legend를 따로 생성
    axes[0].legend(loc="upper left", fontsize=9, frameon=True)
    axes[1].legend(loc="upper left", fontsize=9, frameon=True)

    axes[1].set_xlabel("Date")

    outpath = output_dir / "Figure2_demand_surge.png"

    # tight_layout은 gridspec hspace와 충돌할 수 있으므로 rect만 가볍게 적용
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()

    return outpath


def safe_filename(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^a-zA-Z0-9가-힣_+-]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


ABO_INFO = {
    "a":  {"label": "Type A",  "color": "#1f77b4"},
    "b":  {"label": "Type B",  "color": "#2ca02c"},
    "o":  {"label": "Type O",  "color": "#d62728"},
    "ab": {"label": "Type AB", "color": "#9467bd"},
}


def target_to_product_abo(target: str):
    """
    Example:
      plt_transf_a  -> PC, a
      aph_transf_ab -> APC, ab
    """
    target = str(target).lower()

    if target.startswith("plt_"):
        product = "PC"
    elif target.startswith("aph_"):
        product = "APC"
    else:
        product = "Unknown"

    abo = target.split("_")[-1]
    return product, abo


def plot_model_trends_by_blood_type(
    panel: pd.DataFrame,
    pred_df_model: pd.DataFrame,
    model_name: str,
    output_dir: str,
    recent_start: str = "2023-01-01",
    train_start: str = "2019-03-01",
    train_end: str = "2024-02-29",
    test_start: str = "2024-09-01",
    test_end: str = "2025-02-28",
    rolling_window: int = 30,
    save_dpi: int = 300,
):
    """
    Create one 2 x 4 figure for one model.
    Top row: PC-A, PC-B, PC-O, PC-AB
    Bottom row: APC-A, APC-B, APC-O, APC-AB
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_df_model = pred_df_model.copy()
    pred_df_model["date"] = pd.to_datetime(pred_df_model["date"])
    pred_df_model["target"] = pred_df_model["target"].astype(str)

    targets_order = [
        "plt_transf_a", "plt_transf_b", "plt_transf_o", "plt_transf_ab",
        "aph_transf_a", "aph_transf_b", "aph_transf_o", "aph_transf_ab",
    ]

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(18, 7.8),
        sharex=True,
        gridspec_kw={"hspace": 0.28, "wspace": 0.22}
    )

    axes = axes.flatten()

    for ax, target in zip(axes, targets_order):
        product, abo = target_to_product_abo(target)
        abo_label = ABO_INFO.get(abo, {}).get("label", abo.upper())
        color = ABO_INFO.get(abo, {}).get("color", "tab:blue")

        if target not in panel.columns:
            ax.set_title(f"{product}-{abo_label}\nTarget missing", fontsize=10)
            ax.axis("off")
            continue

        actual = panel[target].fillna(0).copy()
        actual_recent = actual.loc[recent_start:test_end]
        actual_roll = actual_recent.rolling(rolling_window, min_periods=1).mean()
        train_mean = actual.loc[train_start:train_end].mean()

        pred_target = pred_df_model.loc[pred_df_model["target"] == target].copy()

        if pred_target.empty:
            forecast_series = None
        else:
            forecast_series = (
                pred_target
                .set_index("date")["pred"]
                .sort_index()
                .loc[test_start:test_end]
            )

        # background shading
        ax.axvspan(
            pd.to_datetime(recent_start),
            pd.to_datetime(test_start),
            color="#dbe6f3",
            alpha=0.35
        )
        ax.axvspan(
            pd.to_datetime(test_start),
            pd.to_datetime(test_end),
            color="#edd7d2",
            alpha=0.45
        )

        # actual daily demand
        ax.bar(
            actual_recent.index,
            actual_recent.values,
            width=1.0,
            color=color,
            alpha=0.16,
            edgecolor="none",
            label="Actual daily demand"
        )

        # actual rolling mean
        ax.plot(
            actual_roll.index,
            actual_roll.values,
            color=color,
            lw=1.8,
            label=f"Actual {rolling_window}-day rolling mean"
        )

        # train historical mean
        ax.axhline(
            train_mean,
            color="gray",
            lw=1.1,
            ls="--",
            label="Training historical mean"
        )

        # forecast
        if forecast_series is not None and len(forecast_series) > 0:
            ax.plot(
                forecast_series.index,
                forecast_series.values,
                color="black",
                lw=1.7,
                label=f"{model_name} forecast"
            )

        # test start marker
        ax.axvline(
            pd.to_datetime(test_start),
            color="black",
            lw=0.9,
            ls=":"
        )

        ax.set_title(f"{product}-{abo_label}", fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if ax in axes[0:4]:
            ax.tick_params(labelbottom=False)

    axes[0].set_ylabel("Daily PC issued\n(units/day)")
    axes[4].set_ylabel("Daily APC issued\n(units/day)")

    # Common legend from first axis
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=True,
        fontsize=9,
        bbox_to_anchor=(0.5, 1.02)
    )

    fig.suptitle(
        f"ABO-specific platelet demand and {model_name} forecast",
        fontsize=15,
        y=1.08
    )

    outpath = output_dir / f"ABO_trends_{safe_filename(model_name)}.png"
    plt.savefig(outpath, dpi=save_dpi, bbox_inches="tight")
    plt.show()

    return outpath


# =========================================================
# Function: Product-specific ABO trend figure
# =========================================================

def plot_product_abo_trends_for_model(
    panel: pd.DataFrame,
    pred_df_model: pd.DataFrame,
    model_name: str,
    product: str,
    output_dir: str,
    recent_start: str = "2023-01-01",
    train_start: str = "2019-03-01",
    train_end: str = "2024-02-29",
    test_start: str = "2024-09-01",
    test_end: str = "2025-02-28",
    rolling_window: int = 30,
    save_dpi: int = 300,
):
    """
    product = "PC" or "APC"
    Creates one 1 x 4 figure by ABO type.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    product = product.upper()

    if product == "PC":
        targets_order = ["plt_transf_a", "plt_transf_b", "plt_transf_o", "plt_transf_ab"]
        y_label = "Daily PC issued\n(units/day)"
    elif product == "APC":
        targets_order = ["aph_transf_a", "aph_transf_b", "aph_transf_o", "aph_transf_ab"]
        y_label = "Daily APC issued\n(units/day)"
    else:
        raise ValueError("product must be either 'PC' or 'APC'.")

    pred_df_model = pred_df_model.copy()
    pred_df_model["date"] = pd.to_datetime(pred_df_model["date"])
    pred_df_model["target"] = pred_df_model["target"].astype(str)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(18, 3.8),
        sharex=True,
        gridspec_kw={"wspace": 0.22}
    )

    for ax, target in zip(axes, targets_order):
        _, abo = target_to_product_abo(target)
        abo_label = ABO_INFO.get(abo, {}).get("label", abo.upper())
        color = ABO_INFO.get(abo, {}).get("color", "tab:blue")

        if target not in panel.columns:
            ax.set_title(f"{product}-{abo_label}\nTarget missing", fontsize=10)
            ax.axis("off")
            continue

        actual = panel[target].fillna(0)
        actual_recent = actual.loc[recent_start:test_end]
        actual_roll = actual_recent.rolling(rolling_window, min_periods=1).mean()
        train_mean = actual.loc[train_start:train_end].mean()

        pred_target = pred_df_model.loc[pred_df_model["target"] == target].copy()

        if pred_target.empty:
            forecast_series = None
        else:
            forecast_series = (
                pred_target
                .set_index("date")["pred"]
                .sort_index()
                .loc[test_start:test_end]
            )

        ax.axvspan(pd.to_datetime(recent_start), pd.to_datetime(test_start),
                   color="#dbe6f3", alpha=0.35)
        ax.axvspan(pd.to_datetime(test_start), pd.to_datetime(test_end),
                   color="#edd7d2", alpha=0.45)

        ax.bar(
            actual_recent.index,
            actual_recent.values,
            width=1.0,
            color=color,
            alpha=0.16,
            edgecolor="none",
            label="Actual daily demand"
        )

        ax.plot(
            actual_roll.index,
            actual_roll.values,
            color=color,
            lw=1.8,
            label=f"Actual {rolling_window}-day rolling mean"
        )

        ax.axhline(
            train_mean,
            color="gray",
            lw=1.1,
            ls="--",
            label="Training historical mean"
        )

        if forecast_series is not None and len(forecast_series) > 0:
            ax.plot(
                forecast_series.index,
                forecast_series.values,
                color="black",
                lw=1.7,
                label=f"{model_name} forecast"
            )

        ax.axvline(pd.to_datetime(test_start), color="black", lw=0.9, ls=":")
        ax.set_title(f"{product}-{abo_label}", fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(y_label)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=True,
        fontsize=9,
        bbox_to_anchor=(0.5, 1.08)
    )

    fig.suptitle(
        f"{product} ABO-specific demand and {model_name} forecast",
        fontsize=14,
        y=1.18
    )

    outpath = output_dir / f"{product}_ABO_trends_{safe_filename(model_name)}.png"
    plt.savefig(outpath, dpi=save_dpi, bbox_inches="tight")
    plt.show()

    return outpath



# ============================================================
# PART 3. Manuscript Table 1, 2, 4, and 5 generation
# ============================================================


try:
    from docx import Document
    from docx.shared import Pt, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
except ImportError:
    Document = None


def read_csv_if_exists(path):
    path = Path(path)
    if not path.exists():
        print(f"[WARNING] File not found: {path}")
        return None
    return pd.read_csv(path)


def clean_model_name(model):
    """
    Convert internal model names into manuscript-friendly names.
    """
    mapping = {
        "historical_mean": "Historical mean",
        "seasonal_naive": "Seasonal naïve",
        "moving_average_7d": "7-day moving average",
        "moving_average_14d": "14-day moving average",
        "lasso": "LASSO",
        "ridge": "Ridge",
        "lightgbm": "LightGBM",
        "prophet_augmented": "Augmented Prophet",
        "lstm": "LSTM",
        "blend_lasso_ma7_50_50": "LASSO–MA7 blend",
        "blend_lasso_ma14_50_50": "LASSO–MA14 blend",
        "hurdle_lasso_soft": "Hurdle LASSO, soft",
        "hurdle_lasso_thresholded": "Hurdle LASSO, thresholded",
        "hurdle_rolling14_soft": "Hurdle rolling-14, soft",
        "hurdle_rolling14_thresholded": "Hurdle rolling-14, thresholded",
    }
    return mapping.get(str(model), str(model))


def clean_target_name(target):
    """
    Convert internal target names into manuscript-friendly target names.
    """
    target = str(target).lower()

    product = "APC" if target.startswith("aph_") else "PC"
    abo = target.split("_")[-1].upper()

    return f"{product}-{abo}"


def model_family(model):
    """
    Assign model family for manuscript table.
    """
    model = str(model).lower()

    if model in ["historical_mean", "seasonal_naive", "moving_average_7d", "moving_average_14d"]:
        return "Baseline / time-series"
    elif model in ["lasso", "ridge", "lightgbm", "prophet_augmented", "lstm"]:
        return "Regression / ML / DL"
    elif model.startswith("blend_"):
        return "Adaptive blend"
    elif model.startswith("hurdle_"):
        return "Sparse-aware hurdle"
    else:
        return "Other"


def fmt_num(x, digits=2):
    """
    Format numeric values for manuscript tables.
    """
    if pd.isna(x):
        return "-"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def fmt_pct(x, digits=1):
    """
    Format proportion as percentage.
    """
    if pd.isna(x):
        return "-"
    try:
        return f"{100 * float(x):.{digits}f}"
    except Exception:
        return str(x)


def reorder_target_columns(df):
    """
    Reorder product–ABO target columns if present.
    """
    target_order = [
        "PC-A", "PC-B", "PC-O", "PC-AB",
        "APC-A", "APC-B", "APC-O", "APC-AB"
    ]

    existing_targets = [c for c in target_order if c in df.columns]
    other_cols = [c for c in df.columns if c not in target_order]

    return df[other_cols[:2] + existing_targets + other_cols[2:]]


# ============================================================
# 2. Table 1: Expanded model performance table
# ============================================================

def create_table1_model_performance(
    metrics_csv,
    split="test",
    horizon=1,
    metric_for_target="mae",
    rank_metric="weighted_mae",
):
    """
    Create Table 1.
    Expanded performance comparison of forecasting models across
    eight product–ABO target series.

    Required input:
        03_all_metrics_by_target.csv

    Expected columns:
        split, target, horizon, model, mae, weighted_mae, mase

    Returns:
        pd.DataFrame
    """

    metrics = pd.read_csv(metrics_csv)
    metrics.columns = [str(c).strip().lower() for c in metrics.columns]

    required = {"split", "target", "horizon", "model", metric_for_target}
    missing = required - set(metrics.columns)

    if missing:
        raise ValueError(f"Missing required columns in metrics file: {missing}")

    df = metrics[
        (metrics["split"].astype(str).str.lower() == split.lower()) &
        (metrics["horizon"].astype(int) == int(horizon))
    ].copy()

    if df.empty:
        raise ValueError("No rows found for the selected split and horizon.")

    df["Target"] = df["target"].map(clean_target_name)
    df["Model"] = df["model"].map(clean_model_name)
    df["Model family"] = df["model"].map(model_family)

    # Target-level metric pivot
    pivot = (
        df.pivot_table(
            index=["Model family", "Model"],
            columns="Target",
            values=metric_for_target,
            aggfunc="mean"
        )
        .reset_index()
    )

    # Summary metrics
    summary_cols = {}

    if "mae" in df.columns:
        summary_cols["Mean MAE"] = ("mae", "mean")

    if "weighted_mae" in df.columns:
        summary_cols["Mean weighted MAE"] = ("weighted_mae", "mean")

    if "mase" in df.columns:
        summary_cols["Mean MASE"] = ("mase", "mean")

    if "occurrence_f1" in df.columns:
        summary_cols["Mean occurrence F1"] = ("occurrence_f1", "mean")

    if "high_f1" in df.columns:
        summary_cols["Mean high-demand F1"] = ("high_f1", "mean")

    summary = (
        df.groupby(["Model family", "Model"], as_index=False)
        .agg(**summary_cols)
    )

    # Mean rank across targets
    if rank_metric in df.columns:
        df["_rank"] = df.groupby("Target")[rank_metric].rank(method="average", ascending=True)
        rank_df = (
            df.groupby(["Model family", "Model"], as_index=False)
            .agg(**{"Mean rank": ("_rank", "mean")})
        )
        summary = summary.merge(rank_df, on=["Model family", "Model"], how="left")

    table = pivot.merge(summary, on=["Model family", "Model"], how="left")

    # Reorder target columns
    target_order = [
        "PC-A", "PC-B", "PC-O", "PC-AB",
        "APC-A", "APC-B", "APC-O", "APC-AB"
    ]

    ordered_cols = ["Model family", "Model"]
    ordered_cols += [c for c in target_order if c in table.columns]
    ordered_cols += [c for c in table.columns if c not in ordered_cols]

    table = table[ordered_cols]

    # Sort models by family and mean weighted MAE or mean MAE
    sort_col = "Mean weighted MAE" if "Mean weighted MAE" in table.columns else "Mean MAE"

    if sort_col in table.columns:
        table = table.sort_values(["Model family", sort_col], ascending=[True, True])

    # Format numeric columns
    formatted = table.copy()

    for col in formatted.columns:
        if col not in ["Model family", "Model"]:
            formatted[col] = formatted[col].map(lambda x: fmt_num(x, 2))

    return formatted


# ============================================================
# 3. Table 2: Sparse and intermittent demand diagnostics
# ============================================================

def create_table2_sparse_diagnostics(diagnostics_csv):
    """
    Create Table 2.
    Sparse and intermittent demand diagnostics for product–ABO targets.

    Required input:
        01_sparse_intermittent_diagnostics.csv

    Expected columns:
        target, product, mean_demand, median_demand, zero_rate,
        nonzero_days, ADI_average_demand_interval,
        CV2_nonzero_demand, sbc_classification,
        force_hurdle_candidates, method_recommendation

    Returns:
        pd.DataFrame
    """

    diag = pd.read_csv(diagnostics_csv)
    diag.columns = [str(c).strip() for c in diag.columns]

    lower_map = {c.lower(): c for c in diag.columns}

    def get_col(possible_names):
        for name in possible_names:
            if name.lower() in lower_map:
                return lower_map[name.lower()]
        return None

    target_col = get_col(["target"])
    mean_col = get_col(["mean_demand"])
    median_col = get_col(["median_demand"])
    zero_col = get_col(["zero_rate"])
    nonzero_col = get_col(["nonzero_days"])
    adi_col = get_col(["ADI_average_demand_interval", "adi_average_demand_interval", "ADI"])
    cv2_col = get_col(["CV2_nonzero_demand", "cv2_nonzero_demand", "CV2"])
    class_col = get_col(["sbc_classification", "classification", "demand_class"])
    hurdle_col = get_col(["force_hurdle_candidates", "sparse_aware_candidate"])
    rec_col = get_col(["method_recommendation", "recommendation"])

    if target_col is None:
        raise ValueError("Diagnostics file must contain a target column.")

    out = pd.DataFrame()
    out["Target"] = diag[target_col].map(clean_target_name)

    if mean_col:
        out["Mean demand, units/day"] = diag[mean_col].map(lambda x: fmt_num(x, 2))

    if median_col:
        out["Median demand, units/day"] = diag[median_col].map(lambda x: fmt_num(x, 2))

    if zero_col:
        out["Zero-demand days, %"] = diag[zero_col].map(lambda x: fmt_pct(x, 1))

    if nonzero_col:
        out["Nonzero days, n"] = diag[nonzero_col].astype("Int64").astype(str)

    if adi_col:
        out["Average demand interval"] = diag[adi_col].map(lambda x: fmt_num(x, 2))

    if cv2_col:
        out["CV² among nonzero demand"] = diag[cv2_col].map(lambda x: fmt_num(x, 2))

    if class_col:
        out["Demand class"] = diag[class_col].astype(str)

    if hurdle_col:
        out["Sparse-aware candidate"] = diag[hurdle_col].map(
            lambda x: "Yes" if bool(x) else "No"
        )

    if rec_col:
        out["Recommended modeling approach"] = diag[rec_col].astype(str)

    target_order = [
        "PC-A", "PC-B", "PC-O", "PC-AB",
        "APC-A", "APC-B", "APC-O", "APC-AB"
    ]

    out["_order"] = out["Target"].map({t: i for i, t in enumerate(target_order)})
    out = out.sort_values("_order").drop(columns="_order")

    return out


# ============================================================
# 4. Table 4: Simulated inventory outcomes
# ============================================================

def create_table4_inventory_outcomes(
    product_inventory_csv=None,
    target_inventory_csv=None,
    policy_label="Validation-selected policy",
):
    """
    Create Table 4.
    Simulated inventory outcomes at product level.

    Preferred input:
        12_test_inventory_product_level.csv

    Alternative input:
        11_test_inventory_target_level.csv

    Expected columns:
        product, demand, procured, fulfilled, unmet, wasted,
        service_level, wastage_rate, cost

    Returns:
        pd.DataFrame
    """

    product_df = None

    if product_inventory_csv is not None and Path(product_inventory_csv).exists():
        product_df = pd.read_csv(product_inventory_csv)
        product_df.columns = [str(c).strip().lower() for c in product_df.columns]

    elif target_inventory_csv is not None and Path(target_inventory_csv).exists():
        target_df = pd.read_csv(target_inventory_csv)
        target_df.columns = [str(c).strip().lower() for c in target_df.columns]

        if "product" not in target_df.columns and "target" in target_df.columns:
            target_df["product"] = np.where(
                target_df["target"].astype(str).str.startswith("aph_"),
                "APC",
                "PC"
            )

        agg_dict = {}

        for col in ["demand", "procured", "fulfilled", "unmet", "wasted", "cost"]:
            if col in target_df.columns:
                agg_dict[col] = "sum"

        product_df = (
            target_df.groupby("product", as_index=False)
            .agg(agg_dict)
        )

        if "fulfilled" in product_df.columns and "demand" in product_df.columns:
            product_df["service_level"] = product_df["fulfilled"] / product_df["demand"]

        if "wasted" in product_df.columns and "procured" in product_df.columns:
            product_df["wastage_rate"] = np.where(
                product_df["procured"] > 0,
                product_df["wasted"] / product_df["procured"],
                0
            )

    else:
        raise FileNotFoundError(
            "Either product_inventory_csv or target_inventory_csv must exist."
        )

    if product_df is None or product_df.empty:
        raise ValueError("Inventory data are empty.")

    out = pd.DataFrame()
    out["Product"] = product_df["product"].astype(str)
    out["Ordering policy"] = policy_label

    if "demand" in product_df.columns:
        out["Demand, units"] = product_df["demand"].map(lambda x: fmt_num(x, 0))

    if "procured" in product_df.columns:
        out["Procured, units"] = product_df["procured"].map(lambda x: fmt_num(x, 0))

    if "service_level" in product_df.columns:
        out["Service level, %"] = product_df["service_level"].map(lambda x: fmt_pct(x, 1))

    if "unmet" in product_df.columns:
        out["Unmet demand, units"] = product_df["unmet"].map(lambda x: fmt_num(x, 0))

    if "wastage_rate" in product_df.columns:
        out["Wastage rate, %"] = product_df["wastage_rate"].map(lambda x: fmt_pct(x, 1))

    if "wasted" in product_df.columns:
        out["Wasted units"] = product_df["wasted"].map(lambda x: fmt_num(x, 0))

    if "cost" in product_df.columns:
        out["Operational cost"] = product_df["cost"].map(lambda x: fmt_num(x, 1))

    product_order = {"PC": 1, "APC": 2}
    out["_order"] = out["Product"].map(product_order).fillna(99)
    out = out.sort_values("_order").drop(columns="_order")

    return out


# ============================================================
# 5. Table 5: Validation-selected model and inventory policy
# ============================================================

def create_table5_selected_model_policy(
    selected_models_csv,
    selected_policy_csv,
    test_inventory_target_csv=None,
    horizon=1,
):
    """
    Create Table 5.
    Validation-selected forecast model and inventory policy by product–ABO target.

    Required input:
        06_selected_forecast_model_by_validation.csv
        10_selected_inventory_policy_by_validation.csv

    Optional input:
        11_test_inventory_target_level.csv

    Expected selected model columns:
        target, horizon, model, weighted_mae, mae, occurrence_f1

    Expected selected policy columns:
        target, target_days_supply, safety_factor, cost,
        service_level, wastage_rate

    Returns:
        pd.DataFrame
    """

    sel_model = pd.read_csv(selected_models_csv)
    sel_policy = pd.read_csv(selected_policy_csv)

    sel_model.columns = [str(c).strip().lower() for c in sel_model.columns]
    sel_policy.columns = [str(c).strip().lower() for c in sel_policy.columns]

    if "horizon" in sel_model.columns:
        sel_model = sel_model[sel_model["horizon"].astype(int) == int(horizon)].copy()

    required_model = {"target", "model"}
    missing_model = required_model - set(sel_model.columns)

    if missing_model:
        raise ValueError(f"Missing required columns in selected model file: {missing_model}")

    if "target" not in sel_policy.columns:
        raise ValueError("Selected policy file must contain target column.")

    # Keep useful model columns
    model_cols = ["target", "model"]

    for c in ["weighted_mae", "mae", "occurrence_f1", "high_f1", "bias"]:
        if c in sel_model.columns:
            model_cols.append(c)

    sel_model2 = sel_model[model_cols].copy()

    # Keep useful policy columns
    policy_cols = ["target"]

    for c in [
        "target_days_supply",
        "safety_factor",
        "cost",
        "service_level",
        "wastage_rate",
    ]:
        if c in sel_policy.columns:
            policy_cols.append(c)

    sel_policy2 = sel_policy[policy_cols].copy()

    table = sel_model2.merge(sel_policy2, on="target", how="left")

    # Add test inventory outcomes if available
    if test_inventory_target_csv is not None and Path(test_inventory_target_csv).exists():
        test_inv = pd.read_csv(test_inventory_target_csv)
        test_inv.columns = [str(c).strip().lower() for c in test_inv.columns]

        inv_cols = ["target"]

        rename_map = {}

        for c in ["demand", "procured", "unmet", "wasted", "service_level", "wastage_rate", "cost"]:
            if c in test_inv.columns:
                inv_cols.append(c)
                rename_map[c] = f"test_{c}"

        test_inv2 = test_inv[inv_cols].rename(columns=rename_map)
        table = table.merge(test_inv2, on="target", how="left")

    out = pd.DataFrame()
    out["Target"] = table["target"].map(clean_target_name)
    out["Selected forecast model"] = table["model"].map(clean_model_name)

    if "weighted_mae" in table.columns:
        out["Validation weighted MAE"] = table["weighted_mae"].map(lambda x: fmt_num(x, 2))

    if "mae" in table.columns:
        out["Validation MAE"] = table["mae"].map(lambda x: fmt_num(x, 2))

    if "occurrence_f1" in table.columns:
        out["Validation occurrence F1"] = table["occurrence_f1"].map(lambda x: fmt_num(x, 2))

    if "target_days_supply" in table.columns:
        out["Selected days of supply"] = table["target_days_supply"].map(lambda x: fmt_num(x, 1))

    if "safety_factor" in table.columns:
        out["Selected safety factor"] = table["safety_factor"].map(lambda x: fmt_num(x, 2))

    if "cost" in table.columns:
        out["Validation inventory cost"] = table["cost"].map(lambda x: fmt_num(x, 1))

    if "service_level" in table.columns:
        out["Validation service level, %"] = table["service_level"].map(lambda x: fmt_pct(x, 1))

    if "wastage_rate" in table.columns:
        out["Validation wastage rate, %"] = table["wastage_rate"].map(lambda x: fmt_pct(x, 1))

    if "test_service_level" in table.columns:
        out["Test service level, %"] = table["test_service_level"].map(lambda x: fmt_pct(x, 1))

    if "test_wastage_rate" in table.columns:
        out["Test wastage rate, %"] = table["test_wastage_rate"].map(lambda x: fmt_pct(x, 1))

    if "test_unmet" in table.columns:
        out["Test unmet demand, units"] = table["test_unmet"].map(lambda x: fmt_num(x, 0))

    if "test_wasted" in table.columns:
        out["Test wasted units"] = table["test_wasted"].map(lambda x: fmt_num(x, 0))

    target_order = [
        "PC-A", "PC-B", "PC-O", "PC-AB",
        "APC-A", "APC-B", "APC-O", "APC-AB"
    ]

    out["_order"] = out["Target"].map({t: i for i, t in enumerate(target_order)})
    out = out.sort_values("_order").drop(columns="_order")

    return out


# ============================================================
# 6. Save tables to Excel
# ============================================================

def save_tables_to_excel(tables, output_xlsx):
    """
    Save all manuscript tables to one Excel workbook.

    Parameters
    ----------
    tables : dict
        Dictionary of table name -> pd.DataFrame

    output_xlsx : str
        Output Excel path.
    """

    output_xlsx = Path(output_xlsx)
    ensure_dir(output_xlsx.parent)

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        for sheet_name, df in tables.items():
            clean_sheet = re.sub(r"[\[\]\:\*\?\/\\]", "_", sheet_name)[:31]
            df.to_excel(writer, sheet_name=clean_sheet, index=False)

    print(f"[Saved Excel] {output_xlsx}")


# ============================================================
# 7. Save tables to Word
# ============================================================

def add_dataframe_to_docx(document, df, title, note=None):
    """
    Add a pandas DataFrame as a Word table.
    """

    document.add_heading(title, level=1)

    if note:
        p = document.add_paragraph(note)
        p.style = document.styles["Normal"]

    table = document.add_table(rows=1, cols=len(df.columns))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"

    # Header
    hdr_cells = table.rows[0].cells

    for i, col in enumerate(df.columns):
        hdr_cells[i].text = str(col)

    # Body
    for _, row in df.iterrows():
        row_cells = table.add_row().cells

        for i, value in enumerate(row):
            row_cells[i].text = str(value)

    # Formatting
    for row in table.rows:
        for cell in row.cells:
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in paragraph.runs:
                    run.font.size = Pt(8)

    document.add_paragraph("")


def save_tables_to_word(tables, output_docx):
    """
    Save all manuscript tables to a Word document.
    """

    if Document is None:
        raise ImportError(
            "python-docx is not installed. Run: !pip -q install python-docx"
        )

    output_docx = Path(output_docx)
    ensure_dir(output_docx.parent)

    document = Document()

    section = document.sections[0]
    section.top_margin = Inches(0.6)
    section.bottom_margin = Inches(0.6)
    section.left_margin = Inches(0.5)
    section.right_margin = Inches(0.5)

    document.add_heading(
        "Insert-ready Tables for Platelet Demand Forecasting Manuscript",
        level=0
    )

    for title, df in tables.items():
        add_dataframe_to_docx(document, df, title)

    document.save(output_docx)

    print(f"[Saved Word] {output_docx}")


# ============================================================
# 8. Master function
# ============================================================

def create_all_manuscript_tables(
    result_dir,
    output_dir,
    split="test",
    horizon=1,
):
    """
    Create Tables 1, 2, 4, and 5 from forecasting result files.

    Parameters
    ----------
    result_dir : str
        Directory containing model-output CSV files.

    output_dir : str
        Directory where table files will be saved.

    split : str
        Evaluation split for Table 1.

    horizon : int
        Forecast horizon for Table 1 and Table 5.

    Returns
    -------
    dict
        Dictionary of table name -> pd.DataFrame
    """

    result_dir = Path(result_dir)
    output_dir = ensure_dir(output_dir)

    paths = {
        "diagnostics": result_dir / "01_sparse_intermittent_diagnostics.csv",
        "metrics": result_dir / "03_all_metrics_by_target.csv",
        "selected_models": result_dir / "06_selected_forecast_model_by_validation.csv",
        "selected_policy": result_dir / "10_selected_inventory_policy_by_validation.csv",
        "test_inventory_target": result_dir / "11_test_inventory_target_level.csv",
        "test_inventory_product": result_dir / "12_test_inventory_product_level.csv",
    }

    # Table 1
    table1 = create_table1_model_performance(
        metrics_csv=paths["metrics"],
        split=split,
        horizon=horizon,
        metric_for_target="mae",
        rank_metric="weighted_mae",
    )

    # Table 2
    table2 = create_table2_sparse_diagnostics(
        diagnostics_csv=paths["diagnostics"]
    )

    # Table 4
    table4 = create_table4_inventory_outcomes(
        product_inventory_csv=paths["test_inventory_product"],
        target_inventory_csv=paths["test_inventory_target"],
        policy_label="Validation-selected policy",
    )

    # Table 5
    table5 = create_table5_selected_model_policy(
        selected_models_csv=paths["selected_models"],
        selected_policy_csv=paths["selected_policy"],
        test_inventory_target_csv=paths["test_inventory_target"],
        horizon=horizon,
    )

    tables = {
        "Table 1. Expanded forecasting model performance": table1,
        "Table 2. Sparse and intermittent demand diagnostics": table2,
        "Table 4. Simulated inventory outcomes": table4,
        "Table 5. Validation-selected model and policy": table5,
    }

    # Save
    save_tables_to_excel(
        tables=tables,
        output_xlsx=output_dir / "platelet_manuscript_tables.xlsx",
    )

    try:
        save_tables_to_word(
            tables=tables,
            output_docx=output_dir / "platelet_manuscript_tables.docx",
        )
    except Exception as e:
        print(f"[WARNING] Word export failed: {e}")
        print("Excel export was completed successfully.")

    return tables



# ============================================================
# PART 4. Optional supplementary per-model figures
#
# The original notebook re-pasted this same "loop over every model and
# plot X" pattern four separate times (once for Figure 2 per model, once
# more for a "selected models" subset, once for ABO-specific trends per
# model, and once for PC/APC-split ABO trends per model). They are
# consolidated here into one function, kept optional (--extra-figures)
# since the manuscript only uses Figure 1 and Figure 2.
# ============================================================

def generate_supplementary_figures(
    panel: pd.DataFrame,
    base_cfg: "FigureConfig",
    all_predictions_csv: str,
    output_dir: str,
    models: Optional[List[str]] = None,
    horizon: int = 1,
) -> Dict[str, List[Path]]:
    """Generate per-model Figure 2, ABO-trend, and product-ABO-trend plots.

    Parameters
    ----------
    panel : output of load_daily_panel_and_metadata().
    base_cfg : a FigureConfig used as the template for split dates, the
        rolling window, and the recent-window start; its output_dir and
        prediction-file fields are overridden per model below.
    all_predictions_csv : path to Part 1's "02_all_predictions_long.csv".
    output_dir : base directory; three subfolders are created under it.
    models : optional explicit list of model names to plot. If None, every
        model found in `all_predictions_csv` is used.

    Returns
    -------
    dict with keys "figure2", "abo_trends", "product_abo_trends", each a
    list of saved file paths.
    """
    all_predictions_csv = str(all_predictions_csv)
    pred_all = pd.read_csv(all_predictions_csv)
    pred_all.columns = [str(c).strip().lower() for c in pred_all.columns]
    if "model" not in pred_all.columns:
        raise ValueError("all_predictions_csv must contain a 'model' column.")

    if models is None:
        models = sorted(pred_all["model"].dropna().astype(str).unique())
    print(f"[supplementary figures] {len(models)} model(s): {models}")

    fig2_dir = ensure_dir(Path(output_dir) / "Figure2_by_model")
    abo_dir = ensure_dir(Path(output_dir) / "ABO_trends_by_model")
    product_abo_dir = ensure_dir(Path(output_dir) / "Product_ABO_trends_by_model")

    saved: Dict[str, List[Path]] = {"figure2": [], "abo_trends": [], "product_abo_trends": []}

    for model_name in models:
        pred_df_model = load_predictions(
            selected_predictions_csv=None,
            all_predictions_csv=all_predictions_csv,
            model_name=model_name,
            horizon=horizon,
        )
        if pred_df_model is None or pred_df_model.empty:
            print(f"  [skip] No horizon-{horizon} prediction found for model: {model_name}")
            continue

        # --- Figure 2 for this model ---
        cfg_model = FigureConfig(
            data_path=base_cfg.data_path,
            output_dir=str(fig2_dir),
            selected_predictions_csv=None,
            all_predictions_csv=all_predictions_csv,
            figure2_model_name=model_name,
            forecast_label=f"{model_name} forecast",
            train_start=base_cfg.train_start,
            train_end=base_cfg.train_end,
            val_start=base_cfg.val_start,
            val_end=base_cfg.val_end,
            test_start=base_cfg.test_start,
            test_end=base_cfg.test_end,
            recent_start=base_cfg.recent_start,
            rolling_window=base_cfg.rolling_window,
        )
        fig2_path = Path(plot_figure2_demand_surge(panel=panel, cfg=cfg_model, pred_df=pred_df_model))
        new_fig2_path = fig2_dir / f"Figure2_{safe_filename(model_name)}.png"
        if new_fig2_path.exists():
            new_fig2_path.unlink()
        fig2_path.rename(new_fig2_path)
        saved["figure2"].append(new_fig2_path)

        # --- ABO-specific trend (one 2x4 figure) ---
        abo_path = plot_model_trends_by_blood_type(
            panel=panel,
            pred_df_model=pred_df_model,
            model_name=model_name,
            output_dir=str(abo_dir),
            recent_start=base_cfg.recent_start,
            train_start=base_cfg.train_start,
            train_end=base_cfg.train_end,
            test_start=base_cfg.test_start,
            test_end=base_cfg.test_end,
            rolling_window=base_cfg.rolling_window,
        )
        saved["abo_trends"].append(Path(abo_path))

        # --- PC / APC split ABO trend ---
        for product in ("PC", "APC"):
            product_path = plot_product_abo_trends_for_model(
                panel=panel,
                pred_df_model=pred_df_model,
                model_name=model_name,
                product=product,
                output_dir=str(product_abo_dir),
                recent_start=base_cfg.recent_start,
                train_start=base_cfg.train_start,
                train_end=base_cfg.train_end,
                test_start=base_cfg.test_start,
                test_end=base_cfg.test_end,
                rolling_window=base_cfg.rolling_window,
            )
            saved["product_abo_trends"].append(Path(product_path))

        print(f"  [ok] {model_name}")

    print(f"[supplementary figures] Saved {len(saved['figure2'])} Figure 2, "
          f"{len(saved['abo_trends'])} ABO-trend, and "
          f"{len(saved['product_abo_trends'])} product-ABO-trend plot(s).")
    return saved



# ============================================================
# PART 5. Command-line entry point
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sparse-aware platelet demand forecasting, inventory-policy "
                     "simulation, and manuscript figure/table generation.",
    )
    parser.add_argument(
        "--data", default="platelet_data_english_260529.xlsx",
        help="Path to the source Excel file (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir", default="results_sparse_hurdle_plus_models",
        help="Directory for Part 1 analysis CSV/XLSX outputs (default: %(default)s).",
    )
    parser.add_argument(
        "--figures-dir", default="paper_figures",
        help="Directory for Figure 1 / Figure 2 PNGs (default: %(default)s).",
    )
    parser.add_argument(
        "--tables-dir", default="paper_tables",
        help="Directory for the manuscript Excel/Word tables (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-analysis", action="store_true",
        help="Skip Part 1 (forecasting + inventory simulation) and reuse the "
             "existing CSVs already in --output-dir.",
    )
    parser.add_argument(
        "--skip-figures", action="store_true",
        help="Skip Figure 1 / Figure 2 generation.",
    )
    parser.add_argument(
        "--skip-tables", action="store_true",
        help="Skip Table 1/2/4/5 generation.",
    )
    parser.add_argument(
        "--extra-figures", action="store_true",
        help="Also generate the supplementary per-model figures (Figure 2, "
             "ABO-trend, and product-ABO-trend for every model found in the "
             "predictions file). Off by default because the manuscript only "
             "uses Figure 1 and Figure 2.",
    )
    parser.add_argument("--no-lightgbm", action="store_true", help="Skip the LightGBM model.")
    parser.add_argument("--no-prophet", action="store_true", help="Skip the Prophet model.")
    parser.add_argument("--no-lstm", action="store_true", help="Skip the LSTM model.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)

    cfg = SparseHurdleConfig(
        data_path=args.data,
        output_dir=args.output_dir,
        run_lightgbm=not args.no_lightgbm,
        run_prophet=not args.no_prophet,
        run_lstm=not args.no_lstm,
    )
    result_dir = Path(cfg.output_dir)

    # --- Part 1: forecasting + inventory-policy analysis ---
    if not args.skip_analysis:
        run_analysis(cfg)
    else:
        print(f"[skip] Reusing existing analysis outputs in: {result_dir}")

    # --- Part 2: Figure 1 / Figure 2 ---
    if not args.skip_figures:
        fig_cfg = FigureConfig(
            data_path=cfg.data_path,
            output_dir=args.figures_dir,
            selected_predictions_csv=str(result_dir / "07_selected_predictions_long.csv"),
            forecast_label="Validation-selected forecast",
            train_start=cfg.train_start,
            train_end=cfg.train_end,
            val_start=cfg.val_start,
            val_end=cfg.val_end,
            test_start=cfg.test_start,
            test_end=cfg.test_end,
        )
        panel, meta = load_daily_panel_and_metadata(fig_cfg.data_path)
        pred_df = load_predictions(
            selected_predictions_csv=fig_cfg.selected_predictions_csv,
            horizon=1,
        )

        fig1_path = plot_figure1_temporal_patterns(panel=panel, meta=meta, cfg=fig_cfg)
        print(f"Figure 1 saved to: {fig1_path}")

        fig2_path = plot_figure2_demand_surge(panel=panel, cfg=fig_cfg, pred_df=pred_df)
        print(f"Figure 2 saved to: {fig2_path}")

        if args.extra_figures:
            generate_supplementary_figures(
                panel=panel,
                base_cfg=fig_cfg,
                all_predictions_csv=str(result_dir / "02_all_predictions_long.csv"),
                output_dir=args.figures_dir,
            )
    else:
        print("[skip] Figure generation skipped.")

    # --- Part 3: Table 1/2/4/5 ---
    if not args.skip_tables:
        create_all_manuscript_tables(
            result_dir=result_dir,
            output_dir=args.tables_dir,
            split="test",
            horizon=1,
        )
    else:
        print("[skip] Table generation skipped.")


if __name__ == "__main__":
    main()
