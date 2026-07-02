# Target-based Platelet Demand Forecasting

Minimal GitHub repository for product-ABO-specific platelet demand forecasting and inventory simulation.

## What this repository does

- Loads the platelet Excel dataset in transposed format.
- Creates daily forecasting targets for PC/APC by ABO group.
- Generates baseline, LASSO, Ridge, and LASSO-moving-average blend forecasts.
- Computes demand diagnostics and forecasting performance tables.
- Runs product-level inventory simulation.
- Saves manuscript-ready tables, predictions, inventory logs, and figures.

## Required input

Place the raw Excel file in `data/` or provide its path directly:

```text
platelet_data_english_260529.xlsx
```

The file is not included in this repository because it may contain institution-specific data.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python run_pipeline.py \
  --data data/platelet_data_english_260529.xlsx \
  --output outputs
```

## Main outputs

```text
outputs/
├── tables/
│   ├── manuscript_tables.xlsx
│   └── supplementary_tables.xlsx
├── figures/
│   ├── Figure1_trends_no_surgery.png
│   ├── Figure2_product_ABO_forecasts.png
│   └── Supplementary_Figure_inventory_tradeoff.png
├── predictions/
│   ├── test_predictions_long.csv
│   ├── validation_predictions_long.csv
│   └── lasso_coefficients_test.csv
└── inventory_logs/
```

## Target variables

The script maps the following raw variables to standardized targets:

| Target | Raw variable |
|---|---|
| PC-A | `plt_transf_a` |
| PC-B | `plt_transf_b` |
| PC-O | `plt_transf_o` |
| PC-AB | `plt_transf_ab` |
| APC-A | `aph_transf_a` |
| APC-B | `aph_transf_b` |
| APC-O | `aph_transf_o` |
| APC-AB | `aph_transf_ab` |

## Notes

- The inventory trade-off legend uses full labels: `PC: platelet concentrate` and `APC: apheresis platelet concentrates`.
- Inventory simulation is a simplified decision-support model, not a complete representation of real-world blood-bank operations.
