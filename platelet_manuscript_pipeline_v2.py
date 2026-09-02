#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
platelet_manuscript_pipeline_v2.py

Platelet demand forecasting, inventory-policy simulation, and manuscript
table/figure generation -- the pipeline that actually produced the tables
and figures in the submitted manuscript revision.

Why this version replaces the previous merge
----------------------------------------------
An earlier draft of this "final" script was built around
platelet_colab_sparse_hurdle_plus_models.py (hurdle/two-stage models,
LightGBM, Prophet, LSTM). Comparing its output against the actual
Figure1_v2.png / Figure2_v2.png inserted in the manuscript -- and then
against Main doc_submission_revised_2.docx itself -- showed they did not
match. The manuscript's own Table 2 lists exactly ten models (Historical
mean, Seasonal naive, MA7, MA14, LASSO, Ridge, LightGBM, SARIMA, and the
LASSO-MA7 / LASSO-MA14 blends); "hurdle", "Prophet", and "LSTM" do not
appear anywhere in the main text or the supplement. That methodology
belongs to platelet_figures_20260702.py, not the sparse-hurdle script.

This script is therefore built from platelet_figures_20260702.py instead,
cleaned up the same way: Colab-only code removed, duplicate function
definitions resolved, and a single argparse-driven main() added. The
sparse-hurdle/LightGBM/Prophet/LSTM script is intentionally NOT included
here, since it was not part of what was submitted.

Figure and table numbering
---------------------------
The .py export's own internal numbering (from its later, hand-revised
plotting cells) differs from the final manuscript numbering; the mapping
below was confirmed against Main doc_submission_revised_2.docx:

  Main manuscript                                  | Source function
  --------------------------------------------------|---------------------------------------------
  Table 1  Demand-pattern diagnostics               | make_table1_demand_diagnostics
  Table 2  Forecasting performance by model/target   | compute_model_performance
  Table 3  Product-level inventory outcomes         | make_table4_product_inventory (renumbered 4->3)
  Table 4  Independent-test, target-selected policy | make_table5_selected_policy   (renumbered 5->4)
  Figure 1 Platelet issuance & clinical activity     | plot_figure1_trends_no_surgery
  Figure 2 Product-ABO-specific demand & forecasts   | plot_figure3_target_forecast_grid (renumbered 3->2)

  Supplement (generated with --supplement)          | Source function
  --------------------------------------------------|---------------------------------------------
  Supplementary Table (LASSO-selected predictors)   | make_table3_lasso_predictors
  Supplementary Table (platelet-count strata)       | make_platelet_strata_table
  Supplementary Figure S1 (PC/APC aggregate forecast)| plot_figure2_product_forecast
  Supplementary Figure S2 (inventory trade-off)     | plot_figure4_inventory_tradeoff_adjustable

Usage
-----
    python platelet_manuscript_pipeline_v2.py --data /path/to/platelet_data_english_260529.xlsx

Optional flags (see --help):
    --output-dir DIR     Base directory for tables/figures/predictions/logs.
    --supplement          Also generate the supplementary tables and figures
                           (S1/S2 figures, LASSO-predictor and strata tables).
    --no-optional-models  Skip SARIMA and LightGBM (LASSO, Ridge, and the
                           moving-average baselines still run).
    --heme-col NAME       Column used for the hematology-oncology inpatient
                           series in Figure 1, panel (C). Defaults to
                           dept_pt_IMH, auto-detected if not present.

Requirements
------------
    numpy, pandas, scikit-learn, matplotlib, openpyxl
Optional (skipped with a warning if missing):
    lightgbm, statsmodels (SARIMA)
"""

from __future__ import annotations

import argparse


import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")  # safe default for headless/CLI runs; must precede importing pyplot below


import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score


try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


# Default data/output locations. All are overridable from the command line
# (see build_arg_parser() / main() below); paths are created on demand
# inside main() rather than as an import-time side effect.
DATA_PATH = "platelet_data_english_260529.xlsx"


# ------------------------------------------------------------
# Study periods
# ------------------------------------------------------------

TRAIN_START = "2019-03-01"
VALIDATION_START = "2024-03-01"
TEST_START = "2024-09-01"
TEST_END = "2025-02-28"


# ------------------------------------------------------------
# Target definitions
# ------------------------------------------------------------

TARGET_MAP = {
    "PC-A": "plt_transf_a",
    "PC-B": "plt_transf_b",
    "PC-O": "plt_transf_o",
    "PC-AB": "plt_transf_ab",
    "APC-A": "aph_transf_a",
    "APC-B": "aph_transf_b",
    "APC-O": "aph_transf_o",
    "APC-AB": "aph_transf_ab",
}

TARGET_ORDER = list(TARGET_MAP.keys())

PRODUCT_TARGETS = {
    "PC": ["PC-A", "PC-B", "PC-O", "PC-AB"],
    "APC": ["APC-A", "APC-B", "APC-O", "APC-AB"],
}


# ------------------------------------------------------------
# Inventory simulation settings
# ------------------------------------------------------------

DEFAULT_TARGET_DAYS_SUPPLY = 2.5
DEFAULT_SAFETY_FACTOR = 1.0
SHELF_LIFE_DAYS = 4
DELIVERY_LAG_DAYS = 1

SHORTAGE_COST = 10.0
WASTAGE_COST = 1.0


def safe_sheet_name(name):
    name = str(name)
    for ch in ["\\", "/", "?", "*", "[", "]", ":"]:
        name = name.replace(ch, "_")
    return name[:31]


def product_of_target(target):
    if str(target).startswith("APC"):
        return "APC"
    return "PC"


def abo_of_target(target):
    return str(target).split("-")[-1]


def rolling_mean(series, window=30):
    return pd.Series(series).rolling(window, min_periods=max(3, window // 5)).mean()


def zscore(series):
    s = pd.Series(series)
    sd = s.std()
    if sd == 0 or pd.isna(sd):
        return s * 0
    return (s - s.mean()) / sd


def save_tables_to_excel(tables, output_xlsx):
    output_xlsx = Path(output_xlsx)
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        for sheet_name, table in tables.items():
            if table is not None and len(table) > 0:
                table.to_excel(writer, sheet_name=safe_sheet_name(sheet_name), index=False)

    return output_xlsx


def save_figure(fig, output_dir, filename_base, dpi=300):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / f"{filename_base}.png"
    pdf_path = output_dir / f"{filename_base}.pdf"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path



# ============================================================
# PART 1. Data loading, splitting, and prediction models
# (Historical mean, seasonal naive, MA7, MA14, LASSO, Ridge,
#  LightGBM, SARIMA, and LASSO-MA7 / LASSO-MA14 blends -- the
#  exact model roster reported in the manuscript's Table 2.)
# ============================================================


def load_platelet_excel_transposed(path, sheet_name=0):
    """
    Actual file structure:
    - Row 0: metadata headers + date columns
    - Columns 0-5: metadata
    - Columns 6 onward: daily values
    - Rows 1 onward: variables
    """

    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    raw = raw.dropna(how="all").dropna(axis=1, how="all")

    metadata_headers = raw.iloc[0, :6].tolist()
    date_values = pd.to_datetime(raw.iloc[0, 6:], errors="coerce")

    if date_values.notna().sum() < 1000:
        raise ValueError("Date columns were not detected correctly. Check the Excel structure.")

    meta = raw.iloc[1:, :6].copy()
    meta.columns = metadata_headers

    variable_names = meta["Revised variable name"].astype(str).str.strip().tolist()

    values = raw.iloc[1:, 6:].copy()
    values.index = variable_names
    values.columns = date_values

    # Transpose to daily time-series format
    df = values.T.copy()
    df.index.name = "date"

    # Convert all values to numeric where possible
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Remove duplicate columns if any
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Standardize target names
    missing_raw_targets = []
    rename_map = {}

    for standard_target, raw_col in TARGET_MAP.items():
        if raw_col in df.columns:
            rename_map[raw_col] = standard_target
        else:
            missing_raw_targets.append(raw_col)

    df = df.rename(columns=rename_map)

    target_cols = [t for t in TARGET_ORDER if t in df.columns]

    if len(target_cols) < 8:
        print("Warning: fewer than 8 targets were detected.")
        print("Detected targets:", target_cols)
        print("Missing raw target columns:", missing_raw_targets)

    for t in target_cols:
        df[t] = pd.to_numeric(df[t], errors="coerce").fillna(0)

    # Sort and restrict to valid dates
    df = df.sort_index()
    df = df.loc[df.index.notna()].copy()

    print("Loaded dataset")
    print("Shape:", df.shape)
    print("Date range:", df.index.min(), "to", df.index.max())
    print("Detected targets:", target_cols)

    return df, meta, target_cols


def split_train_validation_test(
    df,
    train_start=TRAIN_START,
    validation_start=VALIDATION_START,
    test_start=TEST_START,
    test_end=TEST_END,
):
    train_start = pd.to_datetime(train_start)
    validation_start = pd.to_datetime(validation_start)
    test_start = pd.to_datetime(test_start)
    test_end = pd.to_datetime(test_end)

    df = df.loc[(df.index >= train_start) & (df.index <= test_end)].copy()

    train_for_validation = df.loc[(df.index >= train_start) & (df.index < validation_start)].copy()
    validation_df = df.loc[(df.index >= validation_start) & (df.index < test_start)].copy()
    train_for_test = df.loc[(df.index >= train_start) & (df.index < test_start)].copy()
    test_df = df.loc[(df.index >= test_start) & (df.index <= test_end)].copy()

    print("\nSplit summary")
    print("Train for validation:", train_for_validation.shape, train_for_validation.index.min(), "to", train_for_validation.index.max())
    print("Validation:", validation_df.shape, validation_df.index.min(), "to", validation_df.index.max())
    print("Train for test:", train_for_test.shape, train_for_test.index.min(), "to", train_for_test.index.max())
    print("Test:", test_df.shape, test_df.index.min(), "to", test_df.index.max())

    if len(test_df) == 0:
        raise ValueError("Test period is empty. Check date parsing.")

    return train_for_validation, validation_df, train_for_test, test_df


def classify_demand_pattern(adi, cv2):
    """
    Syntetos-Boylan demand classification:
    smooth, intermittent, erratic, lumpy
    """
    if pd.isna(adi) or pd.isna(cv2):
        return "not classifiable"

    if adi < 1.32 and cv2 < 0.49:
        return "smooth"
    if adi >= 1.32 and cv2 < 0.49:
        return "intermittent"
    if adi < 1.32 and cv2 >= 0.49:
        return "erratic"
    return "lumpy"


def build_feature_matrix(df, target_cols):
    """
    Leakage-safe feature matrix.

    - Target autoregressive features use lagged values only.
    - All non-target numeric variables are lagged by 1 day.
    - Calendar features are same-day deterministic variables.
    """

    blocks = []

    # Autoregressive target features
    ar_parts = {}
    for t in target_cols:
        ar_parts[f"{t}_lag1"] = df[t].shift(1)
        ar_parts[f"{t}_lag7"] = df[t].shift(7)
        ar_parts[f"{t}_ma7"] = df[t].shift(1).rolling(7, min_periods=2).mean()
        ar_parts[f"{t}_ma14"] = df[t].shift(1).rolling(14, min_periods=3).mean()
        ar_parts[f"{t}_ma30"] = df[t].shift(1).rolling(30, min_periods=7).mean()

    blocks.append(pd.DataFrame(ar_parts, index=df.index))

    # Calendar features
    cal = pd.DataFrame(index=df.index)
    cal["dow"] = df.index.dayofweek
    cal["dow_fri"] = (df.index.dayofweek == 4).astype(int)
    cal["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    cal["month"] = df.index.month
    cal["quarter"] = df.index.quarter
    cal["dayofyear_sin"] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    cal["dayofyear_cos"] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    blocks.append(cal)

    # Exogenous numeric variables, lagged by 1 day
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exog_cols = [c for c in numeric_cols if c not in target_cols]

    exog = df[exog_cols].copy()
    exog = exog.loc[:, exog.nunique(dropna=True) > 1]
    exog = exog.shift(1)
    exog.columns = [f"{c}_lag1" for c in exog.columns]
    blocks.append(exog)

    X = pd.concat(blocks, axis=1)
    X = X.copy()

    return X


def make_baseline_predictions(df, train_index, pred_index, target_cols):
    pred_rows = []

    for target in target_cols:
        full_y = pd.to_numeric(df[target], errors="coerce").fillna(0)
        train_y = full_y.loc[train_index]
        hist_mean = train_y.mean()

        pred_map = {
            "Historical mean": pd.Series(hist_mean, index=pred_index),
            "Seasonal naive": full_y.shift(7).reindex(pred_index),
            "MA7": full_y.shift(1).rolling(7, min_periods=2).mean().reindex(pred_index),
            "MA14": full_y.shift(1).rolling(14, min_periods=3).mean().reindex(pred_index),
        }

        y_true = full_y.reindex(pred_index)

        for model_name, pred in pred_map.items():
            pred = pred.fillna(hist_mean).clip(lower=0)

            for date in pred_index:
                pred_rows.append({
                    "date": date,
                    "target": target,
                    "model": model_name,
                    "y_true": y_true.loc[date],
                    "y_pred": pred.loc[date],
                })

    return pd.DataFrame(pred_rows)


def fit_lasso_ridge_models(df, train_index, pred_index, target_cols):
    X_all = build_feature_matrix(df, target_cols)

    pred_rows = []
    coef_rows = []

    lasso_alphas = np.logspace(-4, 1, 50)
    ridge_alphas = np.logspace(-3, 3, 40)

    for target in target_cols:
        y_all = pd.to_numeric(df[target], errors="coerce").fillna(0)

        X_train = X_all.loc[train_index]
        y_train = y_all.loc[train_index]

        X_pred = X_all.loc[pred_index]
        y_true = y_all.loc[pred_index]

        valid_rows = y_train.notna()
        X_train = X_train.loc[valid_rows]
        y_train = y_train.loc[valid_rows]

        if len(y_train) < 60:
            warnings.warn(f"Too few training rows for {target}; skipping LASSO/Ridge.")
            continue

        n_splits = min(5, max(2, len(y_train) // 300))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        models = {
            "LASSO": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LassoCV(
                    alphas=lasso_alphas,
                    cv=tscv,
                    max_iter=30000,
                    random_state=42
                )),
            ]),
            "Ridge": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", RidgeCV(alphas=ridge_alphas)),
            ]),
        }

        for model_name, pipe in models.items():
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_pred)
            pred = np.clip(pred, 0, None)

            for date, yt, yp in zip(pred_index, y_true.values, pred):
                pred_rows.append({
                    "date": date,
                    "target": target,
                    "model": model_name,
                    "y_true": yt,
                    "y_pred": yp,
                })

            if model_name == "LASSO":
                coefs = pipe.named_steps["model"].coef_
                for feature, coef in zip(X_all.columns, coefs):
                    coef_rows.append({
                        "target": target,
                        "feature": feature,
                        "coef": coef,
                    })

    return pd.DataFrame(pred_rows), pd.DataFrame(coef_rows)


def fit_lightgbm_models(df, train_index, pred_index, target_cols):
    if not HAS_LIGHTGBM:
        print("LightGBM is not installed. Skipping LightGBM.")
        return pd.DataFrame()

    X_all = build_feature_matrix(df, target_cols)
    pred_rows = []

    for target in target_cols:
        y_all = pd.to_numeric(df[target], errors="coerce").fillna(0)

        X_train = X_all.loc[train_index]
        y_train = y_all.loc[train_index]

        X_pred = X_all.loc[pred_index]
        y_true = y_all.loc[pred_index]

        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_pred_imp = imputer.transform(X_pred)

        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        model.fit(X_train_imp, y_train)
        pred = model.predict(X_pred_imp)
        pred = np.clip(pred, 0, None)

        for date, yt, yp in zip(pred_index, y_true.values, pred):
            pred_rows.append({
                "date": date,
                "target": target,
                "model": "LightGBM",
                "y_true": yt,
                "y_pred": yp,
            })

    return pd.DataFrame(pred_rows)


def fit_sarima_models(df, train_index, pred_index, target_cols):
    """
    Optional SARIMA model.
    Some sparse targets may fail; failures are skipped.
    """
    if not HAS_STATSMODELS:
        print("statsmodels is not installed. Skipping SARIMA.")
        return pd.DataFrame()

    pred_rows = []

    for target in target_cols:
        y_train = pd.to_numeric(df.loc[train_index, target], errors="coerce").fillna(0)
        y_true = pd.to_numeric(df.loc[pred_index, target], errors="coerce").fillna(0)

        try:
            if y_train.sum() == 0 or y_train.nunique() <= 1:
                continue

            model = SARIMAX(
                y_train,
                order=(1, 1, 1),
                seasonal_order=(1, 0, 1, 7),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            res = model.fit(disp=False)
            pred = res.forecast(steps=len(pred_index))
            pred = pd.Series(np.clip(pred.values, 0, None), index=pred_index)

            for date in pred_index:
                pred_rows.append({
                    "date": date,
                    "target": target,
                    "model": "SARIMA",
                    "y_true": y_true.loc[date],
                    "y_pred": pred.loc[date],
                })

        except Exception as e:
            warnings.warn(f"SARIMA failed for {target}: {e}")

    return pd.DataFrame(pred_rows)


def add_adaptive_blend_predictions(pred_long):
    """
    Add LASSO-MA7 and LASSO-MA14 blend models.
    """
    if pred_long is None or len(pred_long) == 0:
        return pred_long

    rows = []
    key_cols = ["date", "target"]

    blend_specs = [
        ("LASSO-MA7 blend", "LASSO", "MA7"),
        ("LASSO-MA14 blend", "LASSO", "MA14"),
    ]

    for blend_name, model_a, model_b in blend_specs:
        a = pred_long[pred_long["model"] == model_a][key_cols + ["y_true", "y_pred"]]
        b = pred_long[pred_long["model"] == model_b][key_cols + ["y_pred"]]

        if len(a) == 0 or len(b) == 0:
            continue

        merged = a.merge(b, on=key_cols, suffixes=("_a", "_b"))
        merged["y_pred"] = 0.5 * merged["y_pred_a"] + 0.5 * merged["y_pred_b"]
        merged["model"] = blend_name

        rows.append(merged[["date", "target", "model", "y_true", "y_pred"]])

    if rows:
        pred_long = pd.concat([pred_long] + rows, ignore_index=True)

    return pred_long


def generate_predictions(df, train_df, pred_df, target_cols, include_optional_models=True):
    train_index = train_df.index
    pred_index = pred_df.index

    all_pred = []

    baseline_pred = make_baseline_predictions(df, train_index, pred_index, target_cols)
    all_pred.append(baseline_pred)

    reg_pred, lasso_coef = fit_lasso_ridge_models(df, train_index, pred_index, target_cols)
    if len(reg_pred) > 0:
        all_pred.append(reg_pred)

    if include_optional_models:
        sarima_pred = fit_sarima_models(df, train_index, pred_index, target_cols)
        if len(sarima_pred) > 0:
            all_pred.append(sarima_pred)

        lgb_pred = fit_lightgbm_models(df, train_index, pred_index, target_cols)
        if len(lgb_pred) > 0:
            all_pred.append(lgb_pred)

    pred_long = pd.concat(all_pred, ignore_index=True)
    pred_long = add_adaptive_blend_predictions(pred_long)

    return pred_long, lasso_coef


def mase_denominator(train_y, seasonality=7):
    y = pd.Series(train_y).dropna()
    if len(y) <= seasonality:
        return np.nan

    denom = np.abs(y.iloc[seasonality:].values - y.iloc[:-seasonality].values).mean()

    if denom == 0 or pd.isna(denom):
        return np.nan

    return denom


def compute_model_performance(pred_long, train_df, target_cols, under_weight=2.0):
    rows = []

    for (model, target), g in pred_long.groupby(["model", "target"]):
        g = g.dropna(subset=["y_true", "y_pred"])

        if len(g) == 0:
            continue

        y_true = g["y_true"].values
        y_pred = np.clip(g["y_pred"].values, 0, None)

        err = y_pred - y_true
        abs_err = np.abs(err)

        mae = abs_err.mean()

        weighted_abs = np.where(y_pred < y_true, under_weight * abs_err, abs_err)
        weighted_mae = weighted_abs.mean()

        denom = mase_denominator(train_df[target], seasonality=7)
        mase = mae / denom if not pd.isna(denom) else np.nan

        occurrence_true = (y_true > 0).astype(int)
        occurrence_pred = (y_pred >= 0.5).astype(int)
        occurrence_f1 = f1_score(occurrence_true, occurrence_pred, zero_division=0)

        high_threshold = train_df[target].quantile(0.90)
        high_true = (y_true >= high_threshold).astype(int)
        high_pred = (y_pred >= high_threshold).astype(int)
        high_f1 = f1_score(high_true, high_pred, zero_division=0)

        rows.append({
            "Model": model,
            "Target": target,
            "Product": product_of_target(target),
            "ABO": abo_of_target(target),
            "MAE": mae,
            "Weighted MAE": weighted_mae,
            "MASE": mase,
            "Occurrence F1": occurrence_f1,
            "High-demand F1": high_f1,
            "Bias": err.mean(),
            "Underprediction MAE": abs_err[err < 0].mean() if np.any(err < 0) else 0,
            "Overprediction MAE": abs_err[err > 0].mean() if np.any(err > 0) else 0,
        })

    target_level = pd.DataFrame(rows)

    target_level["Target rank"] = (
        target_level.groupby("Target")["MAE"]
        .rank(method="average", ascending=True)
    )

    model_summary = (
        target_level
        .groupby("Model", as_index=False)
        .agg({
            "MAE": "mean",
            "Weighted MAE": "mean",
            "MASE": "mean",
            "Occurrence F1": "mean",
            "High-demand F1": "mean",
            "Bias": "mean",
            "Target rank": "mean",
        })
        .rename(columns={
            "MAE": "Mean MAE",
            "Weighted MAE": "Mean weighted MAE",
            "MASE": "Mean MASE",
            "Occurrence F1": "Mean occurrence F1",
            "High-demand F1": "Mean high-demand F1",
            "Bias": "Mean bias",
            "Target rank": "Mean rank",
        })
        .sort_values(["Mean MAE", "Mean rank"])
    )

    wide = (
        target_level
        .pivot_table(index="Model", columns="Target", values="MAE", aggfunc="mean")
        .reset_index()
    )

    ordered_targets = [t for t in TARGET_ORDER if t in wide.columns]
    wide = wide[["Model"] + ordered_targets]

    manuscript_table = wide.merge(
        model_summary[["Model", "Mean MAE", "Mean MASE", "Mean rank"]],
        on="Model",
        how="left"
    )

    return target_level.round(3), model_summary.round(3), manuscript_table.round(3)



# ============================================================
# PART 2. Manuscript tables
# ============================================================


def make_table1_demand_diagnostics(df, target_cols):
    rows = []

    for target in target_cols:
        y = pd.to_numeric(df[target], errors="coerce").fillna(0)
        n_days = len(y)
        nonzero = y[y > 0]

        mean_demand = y.mean()
        median_demand = y.median()
        sd_demand = y.std()
        zero_pct = y.eq(0).mean() * 100
        nonzero_days = int((y > 0).sum())

        adi = n_days / nonzero_days if nonzero_days > 0 else np.nan

        if len(nonzero) > 1 and nonzero.mean() != 0:
            cv2 = (nonzero.std() / nonzero.mean()) ** 2
        else:
            cv2 = np.nan

        demand_pattern = classify_demand_pattern(adi, cv2)

        sparse_candidate = (
            zero_pct >= 20
            or demand_pattern in ["intermittent", "lumpy"]
            or nonzero_days < 0.8 * n_days
        )

        rows.append({
            "Target": target,
            "Product": product_of_target(target),
            "ABO": abo_of_target(target),
            "Mean daily demand": round(mean_demand, 2),
            "Median daily demand": round(median_demand, 2),
            "SD daily demand": round(sd_demand, 2),
            "Zero-demand days (%)": round(zero_pct, 1),
            "Nonzero-demand days": nonzero_days,
            "ADI": round(adi, 3) if not pd.isna(adi) else np.nan,
            "CV2 among nonzero days": round(cv2, 3) if not pd.isna(cv2) else np.nan,
            "Demand pattern": demand_pattern,
            "Sparse-aware candidate": "Yes" if sparse_candidate else "No",
        })

    return pd.DataFrame(rows)


def annotate_feature_category(feature):
    f = str(feature).lower()

    if "plt_" in f and "inpt" in f:
        return "Laboratory abnormality"
    if "lag" in f or "ma7" in f or "ma14" in f or "ma30" in f:
        return "Autoregressive"
    if f in ["dow", "dow_fri", "month", "quarter", "is_weekend"] or "dayofyear" in f:
        return "Calendar"
    if "surg" in f:
        return "Clinical service activity"
    if "ward" in f or "dept" in f or "inpt" in f or "outpt" in f:
        return "Clinical service activity"
    if "income" in f or "transf" in f or "expire" in f or "return" in f:
        return "Blood product flow"
    return "Other"


def make_table3_lasso_predictors(lasso_coef, min_selected=5, coef_tol=1e-10):
    if lasso_coef is None or len(lasso_coef) == 0:
        return pd.DataFrame()

    df = lasso_coef.copy()
    df["selected"] = df["coef"].abs() > coef_tol

    out = (
        df[df["selected"]]
        .groupby("feature", as_index=False)
        .agg(
            selected_targets=("target", "nunique"),
            mean_beta=("coef", "mean")
        )
    )

    out = out[out["selected_targets"] >= min_selected].copy()
    out["Category"] = out["feature"].apply(annotate_feature_category)

    out = out.rename(columns={
        "feature": "Variable",
        "selected_targets": "Selected (n targets)",
        "mean_beta": "Mean beta",
    })

    out["Mean beta"] = out["Mean beta"].round(3)

    out = out[["Category", "Variable", "Selected (n targets)", "Mean beta"]]
    out = out.sort_values(["Selected (n targets)", "Variable"], ascending=[False, True])

    return out


def make_platelet_strata_table(lasso_coef, df):
    strata_map = {
        "<5": "plt_1_inpt",
        "5-10": "plt_2_inpt",
        "10-20": "plt_3_inpt",
        "20-50": "plt_4_inpt",
        "50-70": "plt_5_inpt",
        "70-100": "plt_6_inpt",
        "100-150": "plt_7_inpt",
        "≥150": "plt_8_inpt",
    }

    rows = []

    for label, col in strata_map.items():
        if col not in df.columns:
            continue

        values = pd.to_numeric(df[col], errors="coerce")

        if lasso_coef is not None and len(lasso_coef) > 0:
            g = lasso_coef[
                (lasso_coef["feature"].str.contains(col, case=False, regex=False))
                & (lasso_coef["coef"].abs() > 1e-10)
            ]
            selected_n = g["target"].nunique()
            mean_beta = g["coef"].mean() if len(g) > 0 else 0
        else:
            selected_n = np.nan
            mean_beta = np.nan

        rows.append({
            "Platelet range (×10^9/L)": label,
            "Variable": col,
            "Selected (n targets)": selected_n,
            "Mean beta": round(mean_beta, 3) if not pd.isna(mean_beta) else np.nan,
            "Daily mean (patients)": round(values.mean(), 2),
            "Non-zero days (%)": round((values > 0).mean() * 100, 1),
        })

    return pd.DataFrame(rows)


def simulate_inventory_policy(
    demand,
    forecast,
    target_days_supply=DEFAULT_TARGET_DAYS_SUPPLY,
    safety_factor=DEFAULT_SAFETY_FACTOR,
    shelf_life_days=SHELF_LIFE_DAYS,
    delivery_lag_days=DELIVERY_LAG_DAYS,
    shortage_cost=SHORTAGE_COST,
    wastage_cost=WASTAGE_COST,
):
    demand = pd.Series(demand).fillna(0).clip(lower=0)
    forecast = pd.Series(forecast).reindex(demand.index).ffill().bfill().fillna(0).clip(lower=0)

    dates = demand.index
    n = len(dates)

    inventory = np.zeros(shelf_life_days)
    inventory[-1] = forecast.iloc[0] * target_days_supply * safety_factor

    pending = {}

    logs = []
    total_demand = 0.0
    total_procured = 0.0
    total_fulfilled = 0.0
    total_unmet = 0.0
    total_wasted = 0.0

    for i, date in enumerate(dates):

        # Receive orders
        received_today = pending.pop(i, 0.0)
        inventory[-1] += received_today

        # Fulfill demand FIFO
        d = float(demand.iloc[i])
        remaining = d
        fulfilled = 0.0

        for bucket in range(shelf_life_days):
            use = min(inventory[bucket], remaining)
            inventory[bucket] -= use
            remaining -= use
            fulfilled += use

            if remaining <= 0:
                break

        unmet = remaining

        # Expiration and aging
        wasted_today = inventory[0]
        inventory[:-1] = inventory[1:]
        inventory[-1] = 0.0

        # Order for future
        future_i = min(i + delivery_lag_days, n - 1)
        future_forecast = float(forecast.iloc[future_i])

        target_inventory = target_days_supply * future_forecast * safety_factor
        on_hand = inventory.sum()
        pending_qty = sum(pending.values())

        order_qty = max(0, math.ceil(target_inventory - on_hand - pending_qty))

        arrival_i = i + delivery_lag_days
        if arrival_i < n and order_qty > 0:
            pending[arrival_i] = pending.get(arrival_i, 0.0) + order_qty
            total_procured += order_qty

        total_demand += d
        total_fulfilled += fulfilled
        total_unmet += unmet
        total_wasted += wasted_today

        logs.append({
            "date": date,
            "demand": d,
            "forecast": float(forecast.iloc[i]),
            "received": received_today,
            "fulfilled": fulfilled,
            "unmet": unmet,
            "wasted": wasted_today,
            "order_qty": order_qty,
            "ending_inventory": inventory.sum(),
        })

    service_level = total_fulfilled / total_demand * 100 if total_demand > 0 else np.nan
    wastage_rate = total_wasted / total_procured * 100 if total_procured > 0 else np.nan
    operational_cost = shortage_cost * total_unmet + wastage_cost * total_wasted

    metrics = {
        "Demand (units)": total_demand,
        "Procured (units)": total_procured,
        "Service level (%)": service_level,
        "Unmet demand (units)": total_unmet,
        "Wastage rate (%)": wastage_rate,
        "Wasted units": total_wasted,
        "Operational cost": operational_cost,
    }

    return pd.DataFrame(logs), metrics


def prediction_long_to_wide(pred_long, model_name):
    g = pred_long[pred_long["model"] == model_name].copy()
    return g.pivot_table(index="date", columns="target", values="y_pred", aggfunc="first")


def make_table4_product_inventory(test_df, pred_long_test, train_df, target_cols):
    rows = []
    logs = {}

    policies = ["Historical mean", "MA7", "LASSO", "Perfect forecast"]

    for product, product_targets in PRODUCT_TARGETS.items():
        product_targets = [t for t in product_targets if t in target_cols]

        if len(product_targets) == 0:
            continue

        demand = test_df[product_targets].sum(axis=1)

        for policy in policies:
            if policy == "Historical mean":
                hist_mean = train_df[product_targets].sum(axis=1).mean()
                forecast = pd.Series(hist_mean, index=test_df.index)

            elif policy == "Perfect forecast":
                forecast = demand.copy()

            else:
                pred_wide = prediction_long_to_wide(pred_long_test, policy)
                available = [t for t in product_targets if t in pred_wide.columns]

                if len(available) == 0:
                    continue

                forecast = pred_wide[available].sum(axis=1).reindex(test_df.index)

            log, metrics = simulate_inventory_policy(
                demand=demand,
                forecast=forecast,
                target_days_supply=2.5,
                safety_factor=1.0
            )

            rows.append({
                "Product": product,
                "Ordering policy": policy,
                **metrics,
            })

            logs[(product, policy)] = log

    out = pd.DataFrame(rows)
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].round(2)

    return out, logs


def select_target_inventory_policy(validation_df, pred_long_val, target_cols):
    candidate_days = [2.0, 2.5, 3.0]
    candidate_safety = [0.9, 1.0, 1.1, 1.2]

    rows = []

    for target in target_cols:
        demand = validation_df[target]

        for model in pred_long_val["model"].unique():
            pred_wide = prediction_long_to_wide(pred_long_val, model)

            if target not in pred_wide.columns:
                continue

            forecast = pred_wide[target].reindex(validation_df.index)

            for days_supply in candidate_days:
                for safety_factor in candidate_safety:
                    _, metrics = simulate_inventory_policy(
                        demand=demand,
                        forecast=forecast,
                        target_days_supply=days_supply,
                        safety_factor=safety_factor
                    )

                    rows.append({
                        "target": target,
                        "selected_model": model,
                        "days_supply": days_supply,
                        "safety_factor": safety_factor,
                        "validation_service_level": metrics["Service level (%)"],
                        "validation_wastage_rate": metrics["Wastage rate (%)"],
                        "validation_unmet_demand": metrics["Unmet demand (units)"],
                        "validation_operational_cost": metrics["Operational cost"],
                    })

    grid = pd.DataFrame(rows)

    selected = (
        grid.sort_values(
            ["target", "validation_operational_cost", "validation_wastage_rate"],
            ascending=[True, True, True]
        )
        .groupby("target", as_index=False)
        .first()
    )

    return selected, grid


def make_table5_selected_policy(test_df, pred_long_test, selected_policy):
    rows = []
    logs = {}

    for _, r in selected_policy.iterrows():
        target = r["target"]
        model = r["selected_model"]
        days_supply = r["days_supply"]
        safety_factor = r["safety_factor"]

        pred_wide = prediction_long_to_wide(pred_long_test, model)

        if target not in pred_wide.columns:
            continue

        demand = test_df[target]
        forecast = pred_wide[target].reindex(test_df.index)

        log, metrics = simulate_inventory_policy(
            demand=demand,
            forecast=forecast,
            target_days_supply=days_supply,
            safety_factor=safety_factor
        )

        rows.append({
            "Target": target,
            "Product": product_of_target(target),
            "ABO": abo_of_target(target),
            "Selected model": model,
            "Selected days of supply": days_supply,
            "Safety factor": safety_factor,
            **metrics,
        })

        logs[target] = log

    out = pd.DataFrame(rows)
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].round(2)

    return out, logs



# ============================================================
# PART 3. Manuscript and supplementary figures
# ============================================================


def find_optional_column(df, keywords):
    for kw in keywords:
        for c in df.columns:
            if kw.lower() in str(c).lower():
                return c
    return None


def plot_figure1_trends_no_surgery(
    df,
    target_cols,
    train_end=TEST_START,
    heme_col="dept_pt_IMH",
    title="Figure 1. Trends in daily platelet issuance and clinical activity",
):
    """
    Figure 1:
    (A) Daily PC units issued
    (B) Daily APC units issued
    (C) Normalized 30-day rolling means of total platelet issue and hematology-oncology activity

    Note:
    - surg_local_today is intentionally excluded from panel C.
    """

    pc_targets = [t for t in target_cols if product_of_target(t) == "PC"]
    apc_targets = [t for t in target_cols if product_of_target(t) == "APC"]

    pc = df[pc_targets].sum(axis=1)
    apc = df[apc_targets].sum(axis=1)
    total = pc + apc

    # If dept_pt_IMH does not exist, automatically search for hematology/oncology-related column.
    if heme_col not in df.columns:
        heme_col = find_optional_column(df, ["dept_pt_IMH", "hem", "hema", "oncology", "imh"])

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # ------------------------------------------------------------
    # A. PC trend
    # ------------------------------------------------------------
    axes[0].bar(df.index, pc, width=1.0, alpha=0.35)
    axes[0].plot(df.index, rolling_mean(pc, 30), linewidth=2)
    axes[0].set_title("(A) Daily platelet concentrate units issued")
    axes[0].set_ylabel("PC units")

    # ------------------------------------------------------------
    # B. APC trend
    # ------------------------------------------------------------
    axes[1].bar(df.index, apc, width=1.0, alpha=0.35)
    axes[1].plot(df.index, rolling_mean(apc, 30), linewidth=2)
    axes[1].set_title("(B) Daily apheresis platelet concentrate units issued")
    axes[1].set_ylabel("APC units")

    # ------------------------------------------------------------
    # C. Normalized 30-day rolling means
    # surg_local_today removed
    # ------------------------------------------------------------
    axes[2].plot(
        df.index,
        zscore(rolling_mean(total, 30)),
        linewidth=2,
        label="Total platelet issued"
    )

    if heme_col is not None and heme_col in df.columns:
        axes[2].plot(
            df.index,
            zscore(rolling_mean(df[heme_col], 30)),
            linewidth=2,
            label=heme_col
        )

    axes[2].set_title("(C) Normalized 30-day rolling means")
    axes[2].set_ylabel("Normalized value")
    axes[2].legend(frameon=False, loc="upper center")

    # Test-period boundary
    for ax in axes:
        ax.axvline(pd.to_datetime(train_end), linestyle="--", linewidth=1)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    return fig


def plot_figure2_product_forecast(train_df, test_df, pred_long_test, target_cols):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    pred_lasso = prediction_long_to_wide(pred_long_test, "LASSO")

    for ax, product in zip(axes, ["PC", "APC"]):
        targets = [t for t in target_cols if product_of_target(t) == product]

        demand = test_df[targets].sum(axis=1)
        hist_mean = train_df[targets].sum(axis=1).mean()

        available = [t for t in targets if t in pred_lasso.columns]
        if len(available) > 0:
            forecast = pred_lasso[available].sum(axis=1).reindex(test_df.index)
        else:
            forecast = pd.Series(np.nan, index=test_df.index)

        ax.bar(test_df.index, demand, width=1.0, alpha=0.45, label="Actual daily demand")
        ax.plot(test_df.index, rolling_mean(demand, 30), linewidth=2, label="30-day rolling mean")
        ax.plot(test_df.index, forecast, linewidth=1.7, label="LASSO forecast")
        ax.axhline(hist_mean, linestyle="--", linewidth=1.7, label="Training historical mean")

        ax.set_title(f"({product}) Actual demand, LASSO forecast, and historical mean")
        ax.set_ylabel("Units/day")
        ax.legend(frameon=False)

    fig.suptitle("Figure 2. Daily platelet demand and forecasted ordering levels", fontsize=14)
    fig.tight_layout()

    return fig


def plot_figure3_target_forecast_grid(train_df, test_df, pred_long_test, target_cols):
    n_targets = len(target_cols)
    ncols = 2
    nrows = int(np.ceil(n_targets / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)

    pred_lasso = prediction_long_to_wide(pred_long_test, "LASSO")
    pred_ma7 = prediction_long_to_wide(pred_long_test, "MA7")

    for ax, target in zip(axes, target_cols):
        demand = test_df[target]
        hist_mean = train_df[target].mean()

        ax.bar(test_df.index, demand, width=1.0, alpha=0.35, label="Actual")
        ax.plot(test_df.index, rolling_mean(demand, 30), linewidth=2, label="30-day rolling mean")
        ax.axhline(hist_mean, linestyle="--", linewidth=1.2, label="Training mean")

        if target in pred_lasso.columns:
            ax.plot(test_df.index, pred_lasso[target].reindex(test_df.index), linewidth=1.2, label="LASSO")

        if target in pred_ma7.columns:
            ax.plot(test_df.index, pred_ma7[target].reindex(test_df.index), linewidth=1.2, label="MA7")

        ax.set_title(target)
        ax.set_ylabel("Units/day")

    for ax in axes[n_targets:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle("Figure 3. Product–ABO-specific platelet demand and forecasts", fontsize=14)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])

    return fig


def plot_figure4_inventory_tradeoff_adjustable(
    table4_inventory,
    label_offsets=None,
    point_offsets=None,
    title=" ",
):
    """
    Figure 4:
    Inventory trade-off between service level and wastage.

    label_offsets:
        dict with key = (Product, Ordering policy)
        value = {"dx": float, "dy": float, "ha": str, "va": str}

    point_offsets:
        dict with key = (Product, Ordering policy)
        value = {"dx": float, "dy": float}
        Used only when points overlap exactly or nearly exactly.
    """

    df_plot = table4_inventory.copy()

    # ----------------------------------------
    # Legend label mapping
    # ----------------------------------------
    product_label_map = {
        "PC": "PC: platelet concentrate",
        "APC": "APC: apheresis platelet concentrates",
    }

    if label_offsets is None:
        label_offsets = {
            ("APC", "MA7"): {
                "dx": 0.04, "dy": 0.18, "ha": "left", "va": "bottom"
            },
            ("APC", "LASSO"): {
                "dx": 0.04, "dy": -0.18, "ha": "left", "va": "top"
            },
            ("APC", "Perfect forecast"): {
                "dx": 0.05, "dy": 0.18, "ha": "left", "va": "bottom"
            },
            ("APC", "Historical mean"): {
                "dx": 0.05, "dy": 0.05, "ha": "left", "va": "bottom"
            },

            ("PC", "LASSO"): {
                "dx": 0.05, "dy": 0.18, "ha": "left", "va": "bottom"
            },
            ("PC", "MA7"): {
                "dx": 0.05, "dy": -0.20, "ha": "left", "va": "top"
            },
            ("PC", "Perfect forecast"): {
                "dx": -0.05, "dy": 0.18, "ha": "right", "va": "bottom"
            },
            ("PC", "Historical mean"): {
                "dx": 0.05, "dy": 0.05, "ha": "left", "va": "bottom"
            },
        }

    if point_offsets is None:
        point_offsets = {
            ("APC", "MA7"): {"dx": -0.01, "dy": 0.00},
            ("APC", "LASSO"): {"dx": 0.01, "dy": 0.00},
            ("PC", "MA7"): {"dx": 0.01, "dy": -0.02},
            ("PC", "Perfect forecast"): {"dx": -0.01, "dy": 0.02},
        }

    fig, ax = plt.subplots(figsize=(9, 6))

    for product, g in df_plot.groupby("Product"):
        x_values = []
        y_values = []

        for _, r in g.iterrows():
            key = (r["Product"], r["Ordering policy"])

            x = r["Wastage rate (%)"]
            y = r["Service level (%)"]

            if key in point_offsets:
                x = x + point_offsets[key].get("dx", 0)
                y = y + point_offsets[key].get("dy", 0)

            x_values.append(x)
            y_values.append(y)

        ax.scatter(
            x_values,
            y_values,
            s=90,
            label=product_label_map.get(product, product)   # <- legend label 변경
        )

        for (_, r), x, y in zip(g.iterrows(), x_values, y_values):
            key = (r["Product"], r["Ordering policy"])

            offset = label_offsets.get(
                key,
                {"dx": 0.05, "dy": 0.05, "ha": "left", "va": "bottom"}
            )

            ax.text(
                x + offset.get("dx", 0.05),
                y + offset.get("dy", 0.05),
                str(r["Ordering policy"]),
                fontsize=9,
                ha=offset.get("ha", "left"),
                va=offset.get("va", "bottom")
            )

    ax.set_xlabel("Wastage rate (%)")
    ax.set_ylabel("Service level (%)")
    ax.set_title(title)

    ax.set_xlim(
        max(-0.15, df_plot["Wastage rate (%)"].min() - 0.20),
        df_plot["Wastage rate (%)"].max() + 0.35
    )

    ax.set_ylim(
        max(80, df_plot["Service level (%)"].min() - 1.0),
        min(101.0, df_plot["Service level (%)"].max() + 0.8)
    )

    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    return fig



# ============================================================
# PART 4. Command-line entry point
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Platelet demand forecasting, inventory-policy simulation, "
                     "and manuscript table/figure generation.",
    )
    parser.add_argument(
        "--data", default=DATA_PATH,
        help="Path to the source Excel file (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir", default="manuscript_tables_figures",
        help="Base output directory; tables/, figures/, predictions/, and "
             "inventory_logs/ subfolders are created under it (default: %(default)s).",
    )
    parser.add_argument(
        "--supplement", action="store_true",
        help="Also generate the supplementary tables (LASSO-selected predictors, "
             "platelet-count strata) and supplementary figures (S1: PC/APC "
             "aggregate forecast; S2: inventory service-level/wastage trade-off).",
    )
    parser.add_argument(
        "--no-optional-models", action="store_true",
        help="Skip SARIMA and LightGBM; keep the historical-mean, seasonal-naive, "
             "MA7, MA14, LASSO, Ridge, and LASSO-MA blend models.",
    )
    parser.add_argument(
        "--heme-col", default="dept_pt_IMH",
        help="Column used as the hematology-oncology inpatient series in "
             "Figure 1, panel (C). Auto-detected if not present (default: %(default)s).",
    )
    return parser


def run_pipeline(
    data_path: str,
    output_dir: str,
    supplement: bool = False,
    include_optional_models: bool = True,
    heme_col: str = "dept_pt_IMH",
) -> dict:
    output_dir = Path(output_dir)
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    pred_dir = output_dir / "predictions"
    log_dir = output_dir / "inventory_logs"
    csv_dir = table_dir / "csv"
    for d in (table_dir, figure_dir, pred_dir, log_dir, csv_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- 1. Load data ---
    df, meta, target_cols = load_platelet_excel_transposed(data_path)
    if len(target_cols) < 8:
        raise ValueError(
            f"Only {len(target_cols)}/8 product-ABO targets were detected: {target_cols}. "
            "Check that the source file has plt_transf_{a,b,o,ab} and "
            "aph_transf_{a,b,o,ab} columns."
        )
    meta.to_excel(table_dir / "Variable_Metadata.xlsx", index=False)

    # --- 2. Split ---
    train_for_validation, validation_df, train_for_test, test_df = split_train_validation_test(df)

    # --- 3. Predictions (validation split, then test split) ---
    print("\nGenerating validation predictions...")
    pred_long_val, lasso_coef_val = generate_predictions(
        df=df, train_df=train_for_validation, pred_df=validation_df,
        target_cols=target_cols, include_optional_models=include_optional_models,
    )

    print("\nGenerating test predictions...")
    pred_long_test, lasso_coef_test = generate_predictions(
        df=df, train_df=train_for_test, pred_df=test_df,
        target_cols=target_cols, include_optional_models=include_optional_models,
    )

    # --- 4. Tables, renumbered to match the manuscript ---
    table1_demand = make_table1_demand_diagnostics(df, target_cols)

    target_level_perf, model_summary_perf, table2_performance = compute_model_performance(
        pred_long=pred_long_test, train_df=train_for_test, target_cols=target_cols, under_weight=2.0,
    )

    # Table 3 in the manuscript = "product-level inventory outcomes"
    # (make_table4_product_inventory in the original notebook cell numbering).
    table3_inventory, inventory_logs = make_table4_product_inventory(
        test_df=test_df, pred_long_test=pred_long_test, train_df=train_for_test, target_cols=target_cols,
    )

    selected_policy, validation_policy_grid = select_target_inventory_policy(
        validation_df=validation_df, pred_long_val=pred_long_val, target_cols=target_cols,
    )

    # Table 4 in the manuscript = "independent-test, target-selected policy"
    # (make_table5_selected_policy in the original notebook cell numbering).
    table4_selected_policy, selected_policy_logs = make_table5_selected_policy(
        test_df=test_df, pred_long_test=pred_long_test, selected_policy=selected_policy,
    )

    main_tables = {
        "Table1_Demand_Diagnostics": table1_demand,
        "Table2_Model_Performance": table2_performance,
        "Table3_Inventory_Outcomes": table3_inventory,
        "Table4_Selected_Policy": table4_selected_policy,
    }

    supplement_tables = {}
    if supplement:
        supplement_tables["S_LASSO_Selected_Predictors"] = make_table3_lasso_predictors(
            lasso_coef=lasso_coef_test, min_selected=5,
        )
        supplement_tables["S_Platelet_Count_Strata"] = make_platelet_strata_table(
            lasso_coef=lasso_coef_test, df=df,
        )
        supplement_tables["S_Target_Level_Performance"] = target_level_perf
        supplement_tables["S_Model_Summary"] = model_summary_perf
        supplement_tables["S_Validation_Policy_Grid"] = validation_policy_grid
        supplement_tables["S_Selected_Validation_Policy"] = selected_policy

    # --- 5. Figures, renumbered to match the manuscript ---
    fig1 = plot_figure1_trends_no_surgery(
        df=df, target_cols=target_cols, train_end=TEST_START, heme_col=heme_col,
    )
    figure_paths = {"Figure1": save_figure(fig1, figure_dir, "Figure1_Platelet_Issuance_Clinical_Activity", dpi=300)}

    # plot_figure3_target_forecast_grid() is titled "Figure 3" internally (its
    # position in the original notebook); it is the manuscript's Figure 2, so
    # the title is corrected after the figure is built.
    fig2 = plot_figure3_target_forecast_grid(
        train_df=train_for_test, test_df=test_df, pred_long_test=pred_long_test, target_cols=target_cols,
    )
    fig2.suptitle("Figure 2. Product–ABO-specific platelet demand and forecasts", fontsize=14)
    figure_paths["Figure2"] = save_figure(fig2, figure_dir, "Figure2_Product_ABO_Target_Forecasts", dpi=300)

    if supplement:
        fig_s1 = plot_figure2_product_forecast(
            train_df=train_for_test, test_df=test_df, pred_long_test=pred_long_test, target_cols=target_cols,
        )
        fig_s1.suptitle("Supplementary Figure S1. Product-level (PC/APC) demand, LASSO forecast, and historical mean", fontsize=13)
        figure_paths["FigureS1"] = save_figure(fig_s1, figure_dir, "FigureS1_Product_Level_LASSO_Forecast", dpi=300)

        fig_s2 = plot_figure4_inventory_tradeoff_adjustable(table4_inventory=table3_inventory)
        fig_s2.suptitle("Supplementary Figure S2. Inventory trade-off between service level and wastage", fontsize=13)
        figure_paths["FigureS2"] = save_figure(fig_s2, figure_dir, "FigureS2_Inventory_Service_Wastage_Tradeoff", dpi=300)

    for fig_name, (png_path, pdf_path) in figure_paths.items():
        print(f"{fig_name}: {png_path}")

    # --- 6. Save tables (Excel workbook + individual CSVs) ---
    main_table_path = table_dir / "Platelet_Manuscript_Main_Tables.xlsx"
    save_tables_to_excel(main_tables, main_table_path)
    print("Saved main tables:", main_table_path)

    if supplement:
        supp_table_path = table_dir / "Platelet_Manuscript_Supplement_Tables.xlsx"
        save_tables_to_excel(supplement_tables, supp_table_path)
        print("Saved supplementary tables:", supp_table_path)

    for name, table in {**main_tables, **supplement_tables}.items():
        if table is not None and len(table) > 0:
            table.to_csv(csv_dir / f"{safe_sheet_name(name)}.csv", index=False, encoding="utf-8-sig")
    print("Saved CSV tables:", csv_dir)

    # --- 7. Save predictions and LASSO coefficients ---
    pred_long_test.to_csv(pred_dir / "test_predictions_long.csv", index=False, encoding="utf-8-sig")
    pred_long_val.to_csv(pred_dir / "validation_predictions_long.csv", index=False, encoding="utf-8-sig")
    lasso_coef_test.to_csv(pred_dir / "lasso_coefficients_test.csv", index=False, encoding="utf-8-sig")
    lasso_coef_val.to_csv(pred_dir / "lasso_coefficients_validation.csv", index=False, encoding="utf-8-sig")
    print("Saved prediction files:", pred_dir)

    # --- 8. Save inventory logs ---
    for (product, policy), log_df in inventory_logs.items():
        filename = f"inventory_log_{product}_{policy}".replace(" ", "_").replace("/", "_")
        log_df.to_csv(log_dir / f"{filename}.csv", index=False, encoding="utf-8-sig")
    for target, log_df in selected_policy_logs.items():
        filename = f"selected_policy_log_{target}".replace(" ", "_").replace("/", "_")
        log_df.to_csv(log_dir / f"{filename}.csv", index=False, encoding="utf-8-sig")
    print("Saved inventory logs:", log_dir)

    # --- 9. Manifest ---
    manifest_rows = []
    for table_name in main_tables:
        manifest_rows.append({"Type": "Main table", "Name": table_name, "File": str(main_table_path)})
    if supplement:
        for table_name in supplement_tables:
            manifest_rows.append({"Type": "Supplementary table", "Name": table_name, "File": str(supp_table_path)})
    for fig_name, (png_path, pdf_path) in figure_paths.items():
        manifest_rows.append({"Type": "Figure", "Name": fig_name, "PNG": str(png_path), "PDF": str(pdf_path)})
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "Table_Figure_Manifest.xlsx"
    manifest_df.to_excel(manifest_path, index=False)
    print("Saved manifest:", manifest_path)

    print("\n" + "=" * 80)
    print("Pipeline complete. Output directory:", output_dir)
    print("=" * 80)

    return {
        "main_tables": main_tables,
        "supplement_tables": supplement_tables,
        "figure_paths": figure_paths,
        "pred_long_test": pred_long_test,
        "pred_long_val": pred_long_val,
    }


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_pipeline(
        data_path=args.data,
        output_dir=args.output_dir,
        supplement=args.supplement,
        include_optional_models=not args.no_optional_models,
        heme_col=args.heme_col,
    )


if __name__ == "__main__":
    main()
