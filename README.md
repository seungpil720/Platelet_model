# Platelet manuscript pipeline (`platelet_manuscript_pipeline_v2_FIXED.py`)

This is the internal pipeline that produces the tables and figures in the
submitted manuscript revision of "Target-Based Platelet Demand Forecasting
and Inventory Simulation During Rapidly Changing Demand": Tables 1-4, the
supplementary LASSO-predictor and platelet-strata tables, and Figures 1, 2,
S1, and S2. It is **not** the public-repository script (that is
`platelet_forecast_inventory.py`, in the sibling `public_repo/` folder /
GitHub repository) -- this script keeps the full manuscript-specific model
roster (including LightGBM and SARIMA) and the exact table/figure layout
used in the submission, and is meant to be re-run locally against the
private institutional source file, not published.

## What changed in this file, and why

This script went through two rounds of correction this revision cycle,
documented in full in the module docstring at the top of the `.py` file
itself; summarized here:

**Round 1 -- fixed-penalty and integer-inventory corrections.**
`fit_lasso_ridge_models()` previously fit `LassoCV`/`RidgeCV`
(cross-validated penalty selection). The primary analysis, and the current
Table 2/3/4/Supplementary Table S6 numbers in the manuscript, use a single
fixed, prespecified penalty instead (**LASSO alpha=0.05; Ridge alpha=5.0**)
-- no `LassoCV`/`RidgeCV` is used anywhere in the current primary analysis.
Because Figure 2 and Supplementary Figure S1 both overlay a "LASSO
forecast" line built from this function's output, the `LassoCV` version had
made those two figures show a visibly different (more smoothed/
regularized) forecast line than the fixed-alpha LASSO described in the text
and used everywhere else. Separately, `simulate_inventory_policy()`
previously used float on-hand inventory with an implicit floor/round; it
now uses an integer, age-structured, ceiling-based order-up-to engine, with
a per-day integer mass-balance `assert`, matching Supplementary Methods
S1.15.

**Round 2 -- unified feature engineering with `platelet_forecast_inventory.py`.**
After round 1, this script's own from-scratch Table 2/3 refit still did not
match the published numbers as closely as expected, because its feature
matrix (`build_feature_matrix()`) was engineered separately from, and
differently than, the public-repo script's (`build_feature_frame()` in
`platelet_forecast_inventory.py`) -- e.g. this script added an `ma30`
autoregressive feature and a `dow_fri` calendar dummy, and filtered out
constant exogenous columns, none of which the public-repo script does. This
round replaces `build_feature_matrix()` with `build_feature_frame()`,
ported directly from `platelet_forecast_inventory.py`, so LASSO, Ridge, and
LightGBM in this script now all build predictors exactly the same way as
the already end-to-end-verified public-repo script. Along with this,
LASSO/Ridge/LightGBM/the moving-average baselines were all switched to the
public-repo script's forecast-origin-date convention (a prediction row's
own date is the date its features were built from; the forecast target is
one day later), with a "+1 day" shift applied immediately after
prediction so every downstream table/figure function -- unchanged from
before -- still receives predictions indexed by the actual (target) date it
forecasts. SARIMA needed no such change: it already produces a genuine
multi-step forecast for the calendar days immediately following its
training window.

**Net effect:** this script's own refit now tracks
`platelet_forecast_inventory.py`'s (already validated against the real
data) forecasts and Table 4 demand totals much more closely than the round-1
version did. Example, from an end-to-end run against the real study data:
LASSO mean MAE 5.888 (round 1: 6.0+; published Table 2: 5.91), SARIMA mean
MAE 8.112 (published: 8.113), and the demand total for every target (e.g.
3,931 for PC-A) matches the published Table 4 exactly. Some residual
numeric divergence from the *published* Table 2/3/4 remains and is expected:
the primary analysis itself was run with the full ~169-predictor set from
`platelet_analysis_v2.py`, which is more extensive than the illustrative
predictor set this script now shares with the public-repo script. The
qualitative pattern (blends best, LASSO/Ridge next, baselines/SARIMA worst)
is preserved.

## Model roster (Table 2)

Historical mean, Seasonal naive, MA7, MA14, LASSO, Ridge, LightGBM, SARIMA,
LASSO-MA7 blend, LASSO-MA14 blend -- all ten models currently reported in
the manuscript's Table 2. LightGBM and SARIMA are optional at runtime (see
`requirements.txt`) and can also be skipped deliberately with
`--no-optional-models`.

## Requirements

```bash
pip install -r requirements.txt
```

`lightgbm` and `statsmodels` are only needed for the LightGBM and SARIMA
rows of Table 2 respectively; if either is missing, that one model is
skipped with a printed warning and the rest of the pipeline still runs to
completion.

## Usage

```bash
python platelet_manuscript_pipeline_v2_FIXED.py \
    --data /path/to/platelet_data_english_260529.xlsx \
    --output-dir manuscript_tables_figures \
    --supplement
```

The source Excel file is **not included** here (see "Data availability"
below); supply your own local copy in the same transposed institutional
layout (metadata columns followed by one column per calendar day, with
`plt_transf_{a,b,o,ab}` and `aph_transf_{a,b,o,ab}` target columns).

Flags:

| Flag | Effect |
|---|---|
| `--data PATH` | Path to the source Excel file (default: `platelet_data_english_260529.xlsx` in the working directory). |
| `--output-dir DIR` | Base output directory (default: `manuscript_tables_figures`); `tables/`, `figures/`, `predictions/`, and `inventory_logs/` subfolders are created under it. |
| `--supplement` | Also generate the supplementary tables (LASSO-selected predictors, platelet-count strata) and supplementary figures (S1: PC/APC aggregate forecast; S2: inventory service-level/wastage trade-off). |
| `--no-optional-models` | Skip SARIMA and LightGBM; keep the historical-mean, seasonal-naive, MA7, MA14, LASSO, Ridge, and LASSO-MA blend models (faster, and avoids needing `lightgbm`/`statsmodels` installed). |
| `--heme-col NAME` | Column used as the hematology-oncology inpatient series in Figure 1, panel (C). Auto-detected if not present (default: `dept_pt_IMH`). |

## Output structure

Everything is written under `--output-dir` (default `manuscript_tables_figures/`):

```
tables/
  Platelet_Manuscript_Main_Tables.xlsx        # Table 1-4, one sheet each
  Platelet_Manuscript_Supplement_Tables.xlsx  # only with --supplement
  csv/*.csv                                    # every table, individually, as CSV
  Variable_Metadata.xlsx                       # raw metadata sheet from the source file
figures/
  Figure1_Platelet_Issuance_Clinical_Activity.png / .pdf
  Figure2_Product_ABO_Target_Forecasts.png / .pdf
  FigureS1_Product_Level_LASSO_Forecast.png / .pdf   # only with --supplement
  FigureS2_Inventory_Service_Wastage_Tradeoff.png / .pdf  # only with --supplement
predictions/
  test_predictions_long.csv / validation_predictions_long.csv
  lasso_coefficients_test.csv / lasso_coefficients_validation.csv
inventory_logs/
  inventory_log_<Product>_<Policy>.csv          # Table 3's per-policy daily logs
  selected_policy_log_<Target>.csv               # Table 4's per-target daily logs
Table_Figure_Manifest.xlsx                       # index of every table/figure produced
```

Table/figure numbering in the code (see the mapping table in the module
docstring) differs from the manuscript numbering in a few places for
historical reasons (carried over from the original analysis notebook); the
script relabels the relevant `suptitle`s and this README uses the final
manuscript numbers throughout.

## Data availability

The underlying hospital-level source data are not publicly available
because they contain institution-level clinical and operational
information and are subject to institutional data-sharing and privacy
requirements. This script is analysis code only.

## Relationship to `platelet_forecast_inventory.py`

`platelet_forecast_inventory.py` (in `public_repo/`) is the clean,
public-repository counterpart to this script: same fixed-alpha LASSO/Ridge
penalty, same integer/ceiling inventory engine, and -- as of this revision
-- the same `build_feature_frame()` feature engineering that this script
now also uses. It deliberately omits LightGBM, SARIMA, and the manuscript's
specific table/figure formatting, since those are not required to
illustrate the core forecasting and inventory-simulation methodology for a
public audience. See `public_repo/README.md` for that script's own scope
and usage.
