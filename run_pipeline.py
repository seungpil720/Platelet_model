#!/usr/bin/env python3
"""
Target-based platelet demand forecasting and inventory simulation.

This minimal script loads the transposed platelet Excel dataset, creates
product-ABO target forecasts, evaluates forecast accuracy, runs inventory
simulation, and saves manuscript-ready tables and figures.

Example:
    python run_pipeline.py --data data/platelet_data_english_260529.xlsx --output outputs
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

TARGET_MAP: Dict[str, str] = {
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
PRODUCT_LABELS = {
    "PC": "PC: platelet concentrate",
    "APC": "APC: apheresis platelet concentrates",
}

TRAIN_START = "2019-03-01"
VALIDATION_START = "2024-03-01"
TEST_START = "2024-09-01"
TEST_END = "2025-02-28"

SHELF_LIFE_DAYS = 4
DELIVERY_LAG_DAYS = 1
SHORTAGE_COST = 10.0
WASTAGE_COST = 1.0


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def product_of_target(target: str) -> str:
    return "APC" if target.startswith("APC") else "PC"


def abo_of_target(target: str) -> str:
    return target.split("-")[-1]


def rolling_mean(series: pd.Series, window: int = 30) -> pd.Series:
    return pd.Series(series).rolling(window, min_periods=max(3, window // 5)).mean()


def zscore(series: pd.Series) -> pd.Series:
    s = pd.Series(series)
    sd = s.std()
    if sd == 0 or pd.isna(sd):
        return s * 0
    return (s - s.mean()) / sd


def safe_sheet_name(name: str) -> str:
    for ch in ['\\', '/', '?', '*', '[', ']', ':']:
        name = str(name).replace(ch, '_')
    return name[:31]


def save_tables_to_excel(tables: Dict[str, pd.DataFrame], output_xlsx: Path) -> None:
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        for name, table in tables.items():
            if table is not None and not table.empty:
                table.to_excel(writer, sheet_name=safe_sheet_name(name), index=False)


def save_figure(fig: plt.Figure, output_dir: Path, filename_base: str, dpi: int = 300) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{filename_base}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{filename_base}.pdf", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_platelet_excel_transposed(path: Path, sheet_name: int | str = 0) -> Tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load the study Excel file where rows are variables and columns are dates."""
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    raw = raw.dropna(how="all").dropna(axis=1, how="all")

    metadata_headers = raw.iloc[0, :6].tolist()
    date_values = pd.to_datetime(raw.iloc[0, 6:], errors="coerce")
    if date_values.notna().sum() < 1000:
        raise ValueError("Date columns were not detected correctly.")

    meta = raw.iloc[1:, :6].copy()
    meta.columns = metadata_headers
    variable_names = meta["Revised variable name"].astype(str).str.strip().tolist()

    values = raw.iloc[1:, 6:].copy()
    values.index = variable_names
    values.columns = date_values

    df = values.T.copy()
    df.index.name = "date"
    df = df.loc[df.index.notna()].sort_index()
    df = df.loc[:, ~df.columns.duplicated()].copy()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    rename_map = {raw_col: target for target, raw_col in TARGET_MAP.items() if raw_col in df.columns}
    df = df.rename(columns=rename_map)
    target_cols = [target for target in TARGET_ORDER if target in df.columns]

    if len(target_cols) != len(TARGET_ORDER):
        missing = [t for t in TARGET_ORDER if t not in target_cols]
        raise ValueError(f"Missing target columns after loading: {missing}")

    for target in target_cols:
        df[target] = pd.to_numeric(df[target], errors="coerce").fillna(0)

    return df, meta, target_cols


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.loc[pd.to_datetime(TRAIN_START):pd.to_datetime(TEST_END)].copy()
    train_for_validation = df.loc[pd.to_datetime(TRAIN_START):pd.to_datetime(VALIDATION_START) - pd.Timedelta(days=1)]
    validation_df = df.loc[pd.to_datetime(VALIDATION_START):pd.to_datetime(TEST_START) - pd.Timedelta(days=1)]
    train_for_test = df.loc[pd.to_datetime(TRAIN_START):pd.to_datetime(TEST_START) - pd.Timedelta(days=1)]
    test_df = df.loc[pd.to_datetime(TEST_START):pd.to_datetime(TEST_END)]
    return train_for_validation, validation_df, train_for_test, test_df


# -----------------------------------------------------------------------------
# Demand diagnostics
# -----------------------------------------------------------------------------

def classify_demand_pattern(adi: float, cv2: float) -> str:
    if pd.isna(adi) or pd.isna(cv2):
        return "not classifiable"
    if adi < 1.32 and cv2 < 0.49:
        return "smooth"
    if adi >= 1.32 and cv2 < 0.49:
        return "intermittent"
    if adi < 1.32 and cv2 >= 0.49:
        return "erratic"
    return "lumpy"


def make_demand_diagnostics(df: pd.DataFrame, target_cols: Iterable[str]) -> pd.DataFrame:
    rows = []
    for target in target_cols:
        y = pd.to_numeric(df[target], errors="coerce").fillna(0)
        nonzero = y[y > 0]
        n_days = len(y)
        nonzero_days = int((y > 0).sum())
        adi = n_days / nonzero_days if nonzero_days else np.nan
        cv2 = (nonzero.std() / nonzero.mean()) ** 2 if len(nonzero) > 1 and nonzero.mean() != 0 else np.nan
        pattern = classify_demand_pattern(adi, cv2)
        rows.append({
            "Target": target,
            "Product": product_of_target(target),
            "ABO": abo_of_target(target),
            "Mean daily demand": round(y.mean(), 2),
            "Median daily demand": round(y.median(), 2),
            "SD daily demand": round(y.std(), 2),
            "Zero-demand days (%)": round(y.eq(0).mean() * 100, 1),
            "Nonzero-demand days": nonzero_days,
            "ADI": round(adi, 3) if not pd.isna(adi) else np.nan,
            "CV2 among nonzero days": round(cv2, 3) if not pd.isna(cv2) else np.nan,
            "Demand pattern": pattern,
            "Sparse-aware candidate": "Yes" if y.eq(0).mean() * 100 >= 20 or pattern in {"intermittent", "lumpy"} else "No",
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Forecasting
# -----------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
    blocks = []

    ar = {}
    for target in target_cols:
        ar[f"{target}_lag1"] = df[target].shift(1)
        ar[f"{target}_lag7"] = df[target].shift(7)
        ar[f"{target}_ma7"] = df[target].shift(1).rolling(7, min_periods=2).mean()
        ar[f"{target}_ma14"] = df[target].shift(1).rolling(14, min_periods=3).mean()
        ar[f"{target}_ma30"] = df[target].shift(1).rolling(30, min_periods=7).mean()
    blocks.append(pd.DataFrame(ar, index=df.index))

    cal = pd.DataFrame(index=df.index)
    cal["dow"] = df.index.dayofweek
    cal["dow_fri"] = (df.index.dayofweek == 4).astype(int)
    cal["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    cal["month"] = df.index.month
    cal["quarter"] = df.index.quarter
    cal["dayofyear_sin"] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    cal["dayofyear_cos"] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    blocks.append(cal)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exog_cols = [c for c in numeric_cols if c not in target_cols]
    exog = df[exog_cols].copy()
    exog = exog.loc[:, exog.nunique(dropna=True) > 1]
    exog = exog.shift(1)
    exog.columns = [f"{c}_lag1" for c in exog.columns]
    blocks.append(exog)

    return pd.concat(blocks, axis=1).copy()


def make_baseline_predictions(df: pd.DataFrame, train_index: pd.Index, pred_index: pd.Index, target_cols: list[str]) -> pd.DataFrame:
    rows = []
    for target in target_cols:
        y = pd.to_numeric(df[target], errors="coerce").fillna(0)
        hist_mean = y.loc[train_index].mean()
        predictions = {
            "Historical mean": pd.Series(hist_mean, index=pred_index),
            "Seasonal naive": y.shift(7).reindex(pred_index),
            "MA7": y.shift(1).rolling(7, min_periods=2).mean().reindex(pred_index),
            "MA14": y.shift(1).rolling(14, min_periods=3).mean().reindex(pred_index),
        }
        for model, pred in predictions.items():
            pred = pred.fillna(hist_mean).clip(lower=0)
            for date in pred_index:
                rows.append({"date": date, "target": target, "model": model, "y_true": y.loc[date], "y_pred": pred.loc[date]})
    return pd.DataFrame(rows)


def fit_lasso_ridge(df: pd.DataFrame, train_index: pd.Index, pred_index: pd.Index, target_cols: list[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X = build_feature_matrix(df, target_cols)
    pred_rows, coef_rows = [], []
    lasso_alphas = np.logspace(-4, 1, 50)
    ridge_alphas = np.logspace(-3, 3, 40)

    for target in target_cols:
        y = pd.to_numeric(df[target], errors="coerce").fillna(0)
        X_train, y_train = X.loc[train_index], y.loc[train_index]
        X_pred, y_true = X.loc[pred_index], y.loc[pred_index]
        n_splits = min(5, max(2, len(y_train) // 300))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        models = {
            "LASSO": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LassoCV(alphas=lasso_alphas, cv=tscv, max_iter=30000, random_state=42)),
            ]),
            "Ridge": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", RidgeCV(alphas=ridge_alphas)),
            ]),
        }
        for model_name, pipe in models.items():
            pipe.fit(X_train, y_train)
            preds = np.clip(pipe.predict(X_pred), 0, None)
            for date, yt, yp in zip(pred_index, y_true.values, preds):
                pred_rows.append({"date": date, "target": target, "model": model_name, "y_true": yt, "y_pred": yp})
            if model_name == "LASSO":
                for feature, coef in zip(X.columns, pipe.named_steps["model"].coef_):
                    coef_rows.append({"target": target, "feature": feature, "coef": coef})
    return pd.DataFrame(pred_rows), pd.DataFrame(coef_rows)


def add_blends(pred_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for blend_name, left, right in [("LASSO-MA7 blend", "LASSO", "MA7"), ("LASSO-MA14 blend", "LASSO", "MA14")]:
        a = pred_long[pred_long["model"] == left][["date", "target", "y_true", "y_pred"]]
        b = pred_long[pred_long["model"] == right][["date", "target", "y_pred"]]
        if a.empty or b.empty:
            continue
        merged = a.merge(b, on=["date", "target"], suffixes=("_a", "_b"))
        merged["y_pred"] = 0.5 * merged["y_pred_a"] + 0.5 * merged["y_pred_b"]
        merged["model"] = blend_name
        rows.append(merged[["date", "target", "model", "y_true", "y_pred"]])
    return pd.concat([pred_long] + rows, ignore_index=True) if rows else pred_long


def generate_predictions(df: pd.DataFrame, train_df: pd.DataFrame, pred_df: pd.DataFrame, target_cols: list[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    baseline = make_baseline_predictions(df, train_df.index, pred_df.index, target_cols)
    regularized, coef = fit_lasso_ridge(df, train_df.index, pred_df.index, target_cols)
    pred_long = pd.concat([baseline, regularized], ignore_index=True)
    return add_blends(pred_long), coef


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def mase_denominator(train_y: pd.Series, seasonality: int = 7) -> float:
    y = pd.Series(train_y).dropna()
    if len(y) <= seasonality:
        return np.nan
    denom = np.abs(y.iloc[seasonality:].values - y.iloc[:-seasonality].values).mean()
    return denom if denom != 0 else np.nan


def compute_model_performance(pred_long: pd.DataFrame, train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (model, target), g in pred_long.groupby(["model", "target"]):
        g = g.dropna(subset=["y_true", "y_pred"])
        y_true = g["y_true"].to_numpy()
        y_pred = np.clip(g["y_pred"].to_numpy(), 0, None)
        err = y_pred - y_true
        abs_err = np.abs(err)
        denom = mase_denominator(train_df[target])
        rows.append({
            "Model": model,
            "Target": target,
            "Product": product_of_target(target),
            "ABO": abo_of_target(target),
            "MAE": abs_err.mean(),
            "Weighted MAE": np.where(y_pred < y_true, 2 * abs_err, abs_err).mean(),
            "MASE": abs_err.mean() / denom if not pd.isna(denom) else np.nan,
            "Occurrence F1": f1_score((y_true > 0).astype(int), (y_pred >= 0.5).astype(int), zero_division=0),
            "High-demand F1": f1_score((y_true >= train_df[target].quantile(0.90)).astype(int), (y_pred >= train_df[target].quantile(0.90)).astype(int), zero_division=0),
            "Bias": err.mean(),
        })
    target_level = pd.DataFrame(rows)
    target_level["Target rank"] = target_level.groupby("Target")["MAE"].rank(method="average")
    model_summary = target_level.groupby("Model", as_index=False).agg({
        "MAE": "mean", "Weighted MAE": "mean", "MASE": "mean", "Occurrence F1": "mean",
        "High-demand F1": "mean", "Bias": "mean", "Target rank": "mean"
    }).rename(columns={
        "MAE": "Mean MAE", "Weighted MAE": "Mean weighted MAE", "MASE": "Mean MASE",
        "Occurrence F1": "Mean occurrence F1", "High-demand F1": "Mean high-demand F1",
        "Bias": "Mean bias", "Target rank": "Mean rank"
    }).sort_values(["Mean MAE", "Mean rank"])
    wide = target_level.pivot_table(index="Model", columns="Target", values="MAE", aggfunc="mean").reset_index()
    wide = wide[["Model"] + [t for t in TARGET_ORDER if t in wide.columns]]
    manuscript = wide.merge(model_summary[["Model", "Mean MAE", "Mean MASE", "Mean rank"]], on="Model", how="left")
    return target_level.round(3), manuscript.round(3)


# -----------------------------------------------------------------------------
# Inventory simulation
# -----------------------------------------------------------------------------

def simulate_inventory_policy(demand: pd.Series, forecast: pd.Series, days_supply: float = 2.5, safety_factor: float = 1.0) -> Tuple[pd.DataFrame, dict]:
    demand = pd.Series(demand).fillna(0).clip(lower=0)
    forecast = pd.Series(forecast).reindex(demand.index).ffill().bfill().fillna(0).clip(lower=0)
    inventory = np.zeros(SHELF_LIFE_DAYS)
    inventory[-1] = forecast.iloc[0] * days_supply * safety_factor
    pending, logs = {}, []
    total_demand = total_procured = total_fulfilled = total_unmet = total_wasted = 0.0

    for i, date in enumerate(demand.index):
        received = pending.pop(i, 0.0)
        inventory[-1] += received
        remaining = float(demand.iloc[i])
        fulfilled = 0.0
        for bucket in range(SHELF_LIFE_DAYS):
            used = min(inventory[bucket], remaining)
            inventory[bucket] -= used
            remaining -= used
            fulfilled += used
            if remaining <= 0:
                break
        wasted = inventory[0]
        inventory[:-1] = inventory[1:]
        inventory[-1] = 0.0
        future_forecast = float(forecast.iloc[min(i + DELIVERY_LAG_DAYS, len(forecast) - 1)])
        target_inventory = days_supply * future_forecast * safety_factor
        order_qty = max(0, math.ceil(target_inventory - inventory.sum() - sum(pending.values())))
        arrival = i + DELIVERY_LAG_DAYS
        if arrival < len(demand) and order_qty > 0:
            pending[arrival] = pending.get(arrival, 0.0) + order_qty
            total_procured += order_qty
        total_demand += float(demand.iloc[i])
        total_fulfilled += fulfilled
        total_unmet += remaining
        total_wasted += wasted
        logs.append({"date": date, "demand": float(demand.iloc[i]), "forecast": float(forecast.iloc[i]), "fulfilled": fulfilled, "unmet": remaining, "wasted": wasted, "order_qty": order_qty, "ending_inventory": inventory.sum()})

    metrics = {
        "Demand (units)": total_demand,
        "Procured (units)": total_procured,
        "Service level (%)": total_fulfilled / total_demand * 100 if total_demand > 0 else np.nan,
        "Unmet demand (units)": total_unmet,
        "Wastage rate (%)": total_wasted / total_procured * 100 if total_procured > 0 else np.nan,
        "Wasted units": total_wasted,
        "Operational cost": SHORTAGE_COST * total_unmet + WASTAGE_COST * total_wasted,
    }
    return pd.DataFrame(logs), metrics


def prediction_long_to_wide(pred_long: pd.DataFrame, model_name: str) -> pd.DataFrame:
    g = pred_long[pred_long["model"] == model_name]
    return g.pivot_table(index="date", columns="target", values="y_pred", aggfunc="first")


def make_product_inventory(test_df: pd.DataFrame, pred_long_test: pd.DataFrame, train_df: pd.DataFrame, target_cols: list[str]) -> Tuple[pd.DataFrame, dict]:
    rows, logs = [], {}
    for product, targets in PRODUCT_TARGETS.items():
        targets = [t for t in targets if t in target_cols]
        demand = test_df[targets].sum(axis=1)
        for policy in ["Historical mean", "MA7", "LASSO", "Perfect forecast"]:
            if policy == "Historical mean":
                forecast = pd.Series(train_df[targets].sum(axis=1).mean(), index=test_df.index)
            elif policy == "Perfect forecast":
                forecast = demand.copy()
            else:
                pred_wide = prediction_long_to_wide(pred_long_test, policy)
                forecast = pred_wide[[t for t in targets if t in pred_wide.columns]].sum(axis=1).reindex(test_df.index)
            log, metrics = simulate_inventory_policy(demand, forecast)
            rows.append({"Product": product, "Ordering policy": policy, **metrics})
            logs[(product, policy)] = log
    out = pd.DataFrame(rows)
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].round(2)
    return out, logs


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

def find_optional_column(df: pd.DataFrame, keywords: Iterable[str]) -> Optional[str]:
    for keyword in keywords:
        for col in df.columns:
            if keyword.lower() in str(col).lower():
                return col
    return None


def plot_trends(df: pd.DataFrame, target_cols: list[str], train_end: str = TEST_START, heme_col: str = "dept_pt_IMH") -> plt.Figure:
    """Main Figure 1. Panel C intentionally excludes surg_local_today."""
    pc = df[[t for t in target_cols if product_of_target(t) == "PC"]].sum(axis=1)
    apc = df[[t for t in target_cols if product_of_target(t) == "APC"]].sum(axis=1)
    total = pc + apc
    if heme_col not in df.columns:
        heme_col = find_optional_column(df, ["dept_pt_IMH", "hem", "hema", "oncology", "imh"])

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].bar(df.index, pc, width=1.0, alpha=0.35)
    axes[0].plot(df.index, rolling_mean(pc, 30), linewidth=2)
    axes[0].set_title("(A) Daily platelet concentrate units issued")
    axes[0].set_ylabel("PC units")

    axes[1].bar(df.index, apc, width=1.0, alpha=0.35)
    axes[1].plot(df.index, rolling_mean(apc, 30), linewidth=2)
    axes[1].set_title("(B) Daily apheresis platelet concentrate units issued")
    axes[1].set_ylabel("APC units")

    axes[2].plot(df.index, zscore(rolling_mean(total, 30)), linewidth=2, label="Total platelet issued")
    if heme_col is not None and heme_col in df.columns:
        axes[2].plot(df.index, zscore(rolling_mean(df[heme_col], 30)), linewidth=2, label=heme_col)
    axes[2].set_title("(C) Normalized 30-day rolling means")
    axes[2].set_ylabel("Normalized value")
    axes[2].legend(frameon=False, loc="upper center")

    for ax in axes:
        ax.axvline(pd.to_datetime(train_end), linestyle="--", linewidth=1)
    fig.suptitle("Figure 1. Trends in daily platelet issuance and clinical activity", fontsize=14)
    fig.tight_layout()
    return fig


def plot_target_forecast_grid(train_df: pd.DataFrame, test_df: pd.DataFrame, pred_long_test: pd.DataFrame, target_cols: list[str]) -> plt.Figure:
    fig, axes = plt.subplots(4, 2, figsize=(14, 13), sharex=True)
    axes = axes.reshape(-1)
    pred_lasso = prediction_long_to_wide(pred_long_test, "LASSO")
    pred_ma7 = prediction_long_to_wide(pred_long_test, "MA7")
    for ax, target in zip(axes, target_cols):
        demand = test_df[target]
        ax.bar(test_df.index, demand, width=1.0, alpha=0.35, label="Actual")
        ax.plot(test_df.index, rolling_mean(demand, 30), linewidth=2, label="30-day rolling mean")
        ax.axhline(train_df[target].mean(), linestyle="--", linewidth=1.2, label="Training mean")
        if target in pred_lasso.columns:
            ax.plot(test_df.index, pred_lasso[target].reindex(test_df.index), linewidth=1.2, label="LASSO")
        if target in pred_ma7.columns:
            ax.plot(test_df.index, pred_ma7[target].reindex(test_df.index), linewidth=1.2, label="MA7")
        ax.set_title(target)
        ax.set_ylabel("Units/day")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle("Figure 2. Product–ABO-specific platelet demand and forecasts", fontsize=14)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    return fig


def plot_inventory_tradeoff(table_inventory: pd.DataFrame, label_offsets: Optional[dict] = None, point_offsets: Optional[dict] = None) -> plt.Figure:
    if label_offsets is None:
        label_offsets = {
            ("APC", "MA7"): {"dx": 0.04, "dy": 0.18, "ha": "left", "va": "bottom"},
            ("APC", "LASSO"): {"dx": 0.04, "dy": -0.18, "ha": "left", "va": "top"},
            ("APC", "Perfect forecast"): {"dx": 0.05, "dy": 0.18, "ha": "left", "va": "bottom"},
            ("APC", "Historical mean"): {"dx": 0.05, "dy": 0.05, "ha": "left", "va": "bottom"},
            ("PC", "LASSO"): {"dx": 0.05, "dy": 0.18, "ha": "left", "va": "bottom"},
            ("PC", "MA7"): {"dx": 0.05, "dy": -0.20, "ha": "left", "va": "top"},
            ("PC", "Perfect forecast"): {"dx": -0.05, "dy": 0.18, "ha": "right", "va": "bottom"},
            ("PC", "Historical mean"): {"dx": 0.05, "dy": 0.05, "ha": "left", "va": "bottom"},
        }
    if point_offsets is None:
        point_offsets = {
            ("APC", "MA7"): {"dx": -0.01, "dy": 0.00},
            ("APC", "LASSO"): {"dx": 0.01, "dy": 0.00},
            ("PC", "MA7"): {"dx": 0.01, "dy": -0.02},
            ("PC", "Perfect forecast"): {"dx": -0.01, "dy": 0.02},
        }

    fig, ax = plt.subplots(figsize=(9, 6))
    for product, group in table_inventory.groupby("Product"):
        xs, ys = [], []
        for _, row in group.iterrows():
            key = (row["Product"], row["Ordering policy"])
            xs.append(row["Wastage rate (%)"] + point_offsets.get(key, {}).get("dx", 0))
            ys.append(row["Service level (%)"] + point_offsets.get(key, {}).get("dy", 0))
        ax.scatter(xs, ys, s=90, label=PRODUCT_LABELS.get(product, product))
        for (_, row), x, y in zip(group.iterrows(), xs, ys):
            key = (row["Product"], row["Ordering policy"])
            off = label_offsets.get(key, {"dx": 0.05, "dy": 0.05, "ha": "left", "va": "bottom"})
            ax.text(x + off["dx"], y + off["dy"], str(row["Ordering policy"]), fontsize=9, ha=off["ha"], va=off["va"])
    ax.set_xlabel("Wastage rate (%)")
    ax.set_ylabel("Service level (%)")
    ax.set_title("Inventory trade-off between service level and wastage")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(max(-0.15, table_inventory["Wastage rate (%)"].min() - 0.20), table_inventory["Wastage rate (%)"].max() + 0.35)
    ax.set_ylim(max(80, table_inventory["Service level (%)"].min() - 1.0), min(101.0, table_inventory["Service level (%)"].max() + 0.8))
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def run_pipeline(data_path: Path, output_dir: Path) -> None:
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    pred_dir = output_dir / "predictions"
    log_dir = output_dir / "inventory_logs"
    for directory in [table_dir, figure_dir, pred_dir, log_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    df, meta, target_cols = load_platelet_excel_transposed(data_path)
    train_val, val_df, train_test, test_df = split_data(df)

    pred_val, coef_val = generate_predictions(df, train_val, val_df, target_cols)
    pred_test, coef_test = generate_predictions(df, train_test, test_df, target_cols)

    table1 = make_demand_diagnostics(df, target_cols)
    target_level_perf, table2 = compute_model_performance(pred_test, train_test)
    table3, inventory_logs = make_product_inventory(test_df, pred_test, train_test, target_cols)

    save_tables_to_excel({
        "Table1_Demand_Diagnostics": table1,
        "Table2_Model_Performance": table2,
        "Table3_Inventory_Outcomes": table3,
    }, table_dir / "manuscript_tables.xlsx")
    save_tables_to_excel({
        "Target_Level_Performance": target_level_perf,
        "Validation_Predictions": pred_val,
        "Test_Predictions": pred_test,
        "LASSO_Coefficients_Test": coef_test,
        "Variable_Metadata": meta,
    }, table_dir / "supplementary_tables.xlsx")

    pred_test.to_csv(pred_dir / "test_predictions_long.csv", index=False, encoding="utf-8-sig")
    pred_val.to_csv(pred_dir / "validation_predictions_long.csv", index=False, encoding="utf-8-sig")
    coef_test.to_csv(pred_dir / "lasso_coefficients_test.csv", index=False, encoding="utf-8-sig")

    save_figure(plot_trends(df, target_cols), figure_dir, "Figure1_trends_no_surgery")
    save_figure(plot_target_forecast_grid(train_test, test_df, pred_test, target_cols), figure_dir, "Figure2_product_ABO_forecasts")
    save_figure(plot_inventory_tradeoff(table3), figure_dir, "Supplementary_Figure_inventory_tradeoff")

    for (product, policy), log in inventory_logs.items():
        filename = f"inventory_log_{product}_{policy}".replace(" ", "_").replace("/", "_")
        log.to_csv(log_dir / f"{filename}.csv", index=False, encoding="utf-8-sig")

    print(f"Pipeline completed. Outputs saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run platelet demand forecasting and inventory simulation.")
    parser.add_argument("--data", required=True, type=Path, help="Path to platelet_data_english_260529.xlsx")
    parser.add_argument("--output", default=Path("outputs"), type=Path, help="Output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.data, args.output)
