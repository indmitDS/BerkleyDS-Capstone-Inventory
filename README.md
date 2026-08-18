# Machine-Learning Demand Forecasting for Working-Capital Optimization

**Repository:** [BerkleyDS-Capstone-Inventory](https://github.com/indmitDS/BerkleyDS-Capstone-Inventory)

## Table of contents

- [Executive summary](#executive-summary)
- [Executed findings](#executed-findings)
- [Why Lasso performed best](#why-lasso-performed-best)
- [Main deliverable](#main-deliverable)
- [Research question and business rationale](#research-question-and-business-rationale)
- [Data sources](#data-sources)
- [Repository structure](#repository-structure)
- [Setup and execution](#setup-and-execution)
- [Methodology](#methodology)
- [How to interpret the final results](#how-to-interpret-the-final-results)
- [Visualizations produced](#visualizations-produced)
- [Limitations](#limitations)
- [Recommendations and future work](#recommendations-and-future-work)
- [Reproducibility and grading-rubric coverage](#reproducibility-and-grading-rubric-coverage)
- [Responsible use](#responsible-use)

## Executive summary

This capstone asks: **Can machine-learning-driven demand forecasting improve working-capital efficiency by optimizing inventory levels while maintaining service levels?** It answers that question through an end-to-end, reproducible workflow using the M5 Forecasting—Accuracy data for the main daily demand and inventory experiment and the Walmart Recruiting—Store Sales Forecasting data for an independent weekly robustness analysis.

The project benchmarks seasonal-naive forecasts against linear regularized models, random forests, extremely randomized trees, gradient boosting, histogram gradient boosting, and optional XGBoost, SARIMAX, Prophet, and LSTM models. It evaluates every forecast on a chronological holdout with MAE, RMSE, WAPE, sMAPE, MASE, R², and an explicitly labeled business-friendly **forecast accuracy = 100 − WAPE**. It also poses a separate stockout-risk classification problem and reports accuracy, precision, recall, F1, ROC-AUC, PR-AUC, ROC/PR curves, and confusion matrices.

The best M5 forecast is passed into a lead-time base-stock simulation and compared with a seasonal-naive policy. The operational evaluation reports fill rate, cycle-service level, stockout periods, lost units, average inventory, working capital, DIO, and safety stock. EOQ and service-level/lead-time sensitivity analyses make the cash-versus-service trade-off explicit.

## Executed findings

- **M5:** Lasso achieved MAE 58.98, RMSE 76.78, WAPE 5.53%, forecast accuracy 94.47%, MASE 0.432, and R² 0.861. WAPE improved 41.1% versus the 7-day seasonal-naive benchmark.
- **Classification:** Histogram Gradient Boosting produced 94.6% accuracy and F1 0.857; Logistic Regression and Random Forest each achieved 100% recall.
- **Inventory:** the Lasso policy increased fill rate from 93.74% to 93.87%, increased cycle-service level from 82.14% to 85.71%, and reduced stockout periods from 10 to 8. It also increased working capital from $7,669 to $8,121 (+5.9%) and DIO from 1.8 to 1.9 days.
- **Walmart:** the 52-week seasonal-naive benchmark remained best, with WAPE about 1.9% and forecast accuracy about 98.1%.

**Conclusion:** machine learning improved M5 forecast accuracy, but the tested inventory policy did not improve working-capital efficiency under the present assumptions. It produced a modest service improvement by holding more inventory. A same-service-level policy comparison and safety-stock recalibration are required before claiming a cash-efficiency benefit.

## Why Lasso performed best

The result is conditional on this aggregate M5 slice, engineered features, model specifications, and 56-day holdout. Lasso benefits from a smooth aggregate series whose lag, rolling, calendar, price, SNAP, and event effects can be represented by a regularized linear combination. Its L1 penalty suppresses redundant correlated predictors, which can reduce variance. XGBoost may add unnecessary nonlinear flexibility; the univariate SARIMAX and Prophet specifications do not receive the same exogenous feature matrix; and the recursive 28-day LSTM has limited training history and can compound multi-step errors. The notebook now quantifies these explanations with a relative-performance table, forecast-horizon error profiles, a Lasso coefficient/sparsity audit, and expanding-window stability comparisons. These diagnostics support a bias–variance explanation but do not establish that Lasso is universally superior.

## Main deliverable

- [CapstoneEvaluation_FINAL_WITH_DIAGNOSTICS.ipynb](CapstoneEvaluation_FINAL_WITH_DIAGNOSTICS.ipynb) — complete data loading, cleaning, EDA, feature engineering, model training/tuning, plots, diagnostics, classification, explainability, inventory optimization, sensitivity analysis, and conclusions.

## Research question and business rationale

Inventory is a major consumer of working capital. Overstocking ties up cash and creates holding, obsolescence, and markdown costs. Understocking creates lost revenue, expediting, and poor customer service. Forecast accuracy matters only when it improves this operational trade-off.

The project tests two hypotheses:

1. Lag, rolling, calendar, price, event, holiday, markdown, and macroeconomic signals improve out-of-time forecasting over seasonal-naive baselines.
2. A forecast-driven base-stock policy reduces average inventory, working capital, and DIO without materially lowering fill rate or cycle-service level.

## Data sources

### M5 Forecasting—Accuracy

Official competition page: https://www.kaggle.com/competitions/m5-forecasting-accuracy

The files provide daily Walmart unit sales in a hierarchical product/store structure, a calendar with events and SNAP indicators, and weekly selling prices. The notebook selects a configurable store/category slice and a deterministic item sample for practical runtime, then aggregates it to a daily series.

Required files:

- `sales_train_validation.csv` or `sales_train_evaluation.csv`
- `calendar.csv`
- `sell_prices.csv`

### Walmart Recruiting—Store Sales Forecasting

Official competition page: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting

This dataset contains historical weekly sales for 45 stores plus holiday, markdown, weather, fuel, CPI, unemployment, and store type/size signals. It is modeled separately as a robustness check.

Required files:

- `train.csv`
- `features.csv`
- `stores.csv`

### Why the datasets are not joined

They represent different time periods, grains, targets, and identifier systems. A row-level join would create false relationships. Applying a common, leakage-safe methodology to each source is more defensible and reveals whether conclusions generalize across daily unit demand and weekly dollar sales.

## Repository structure

```text
.
├── CapstoneEvaluation_FINAL_WITH_DIAGNOSTICS.ipynb
├── README.md
├── requirements.txt
├── .gitignore
├── .gitattributes             # Git LFS rules for dataset files
└── data/                       # dataset files tracked with Git LFS
    ├── m5/
    │   ├── sales_train_evaluation.csv
    │   ├── calendar.csv
    │   └── sell_prices.csv
    └── walmart/
        ├── train.csv
        ├── features.csv
        └── stores.csv
```

## Setup and execution

1. Create and activate a virtual environment.
2. Install dependencies: `pip install -r requirements.txt`.
3. Install Git LFS with `git lfs install`, because some dataset files exceed GitHub's standard file-size limit.
4. Download both Kaggle competitions and unzip them into the paths above. You may need to accept the competition rules and configure the Kaggle API.
5. Open `CapstoneEvaluation_FINAL_WITH_DIAGNOSTICS.ipynb`.
6. Run all cells in order.
7. For optional computationally intensive models, set `RUN_HEAVY_MODELS = True`. For a faster first run, leave it `False`.

Example Kaggle commands:

```bash
kaggle competitions download -c m5-forecasting-accuracy -p data/m5
kaggle competitions download -c walmart-recruiting-store-sales-forecasting -p data/walmart
```

Dataset files in this repository are managed with Git LFS:

```bash
git lfs install
git lfs pull
```

## Methodology

### Data quality and EDA

The notebook checks missingness, schema, uniqueness, chronological order, and summary statistics. Visualizations cover demand trends, distributions, weekday/holiday/event effects, rolling mean and volatility, autocorrelation, correlations, and external variables.

### Leakage-safe feature engineering

Features include calendar cycles, lags, and rolling means/standard deviations/minima/maxima. All rolling demand features use `shift(1)` before rolling, so the target row is never used to predict itself. Price changes, events, SNAP, holidays, markdowns, and economic fields are included when available.

### Validation

- Final test sets are the most recent 56 M5 days and 16 Walmart weeks.
- Hyperparameter search uses expanding-window `TimeSeriesSplit` only inside training data.
- No random train/test split is used.
- The notebook explicitly asserts that the last training date precedes the first test date.

### Forecasting models

- Seasonal naive (7-day M5, 52-week Walmart)
- Ordinary least squares
- Ridge, Lasso, and Elastic Net
- Random Forest
- Extra Trees
- Gradient Boosting
- Histogram Gradient Boosting
- Tuned Random Forest with randomized search
- Ridge with explicit time-aware GridSearchCV
- Optional XGBoost
- Optional SARIMAX
- Optional Prophet
- Optional LSTM

### Forecast evaluation

- **MAE:** average absolute error in target units.
- **RMSE:** penalizes large errors more heavily.
- **WAPE:** total absolute error divided by total actual demand; primary business metric.
- **Forecast accuracy:** `100 × (1 − WAPE)`; convenient but clearly defined because “accuracy” is not a universal regression metric.
- **sMAPE:** symmetric percentage error.
- **MASE:** error scaled to a seasonal-naive benchmark; values below 1 are desirable.
- **R²:** variance explained; retained for rubric completeness but not used alone for model selection.

### Classification and confusion matrices

A confusion matrix is not valid for a continuous forecast. The notebook therefore defines a distinct binary label: whether realized demand exceeds a pre-period inventory plan formed from lagged rolling demand and a service buffer. Logistic Regression, Random Forest, and Histogram Gradient Boosting classifiers are compared. Recall receives special attention because a false negative is a missed stockout-risk period.

### Explainability

Holdout permutation importance is computed for the selected sklearn forecast model. Optional SHAP analysis can be enabled for compatible models. Interpretation remains associational, not causal.

### Inventory and working-capital layer

The selected M5 forecast drives a daily base-stock simulation with lead time and safety stock. The ML policy is compared to seasonal naive using:

- fill rate and cycle-service level;
- stockout periods and lost units;
- average inventory units;
- working capital = average inventory × unit cost;
- DIO = 365 × average inventory / annualized demand;
- safety stock.

EOQ is computed as `sqrt(2DS/H)`. Sensitivity analysis varies target service level and lead time to show the working-capital/service frontier.

## How to interpret the final results

The notebook selects the lowest-WAPE model and prints an automated cross-dataset summary. A defensible “yes” to the research question requires:

1. better M5 holdout WAPE than the 7-day seasonal naive baseline;
2. ideally MASE below 1 and no major residual bias;
3. reduced working capital or DIO at an acceptable service level;
4. qualitatively consistent evidence from the Walmart weekly robustness analysis.

Do not claim causality from model importance or correlations. Do not claim a working-capital win if service deterioration exceeds the business tolerance.

## Visualizations produced

- M5 demand time series, distribution, weekday and event comparisons
- Interactive Plotly actual-versus-forecast chart
- Rolling trend/volatility and autocorrelation
- Model WAPE comparison
- Holdout actual-versus-predicted forecasts
- Residual scatter and distribution
- Permutation importance and optional SHAP summary
- Stockout-risk confusion matrices, ROC curves, and precision–recall curves
- Ending inventory and service-level comparisons
- EOQ cost curves
- Working-capital and fill-rate sensitivity curves
- Walmart weekly trend, holiday comparison, correlation heatmap, forecasts, and model comparison

## Limitations

- Observed sales may be censored by historical stockouts and therefore differ from unconstrained demand.
- Aggregation hides SKU-level intermittent demand, substitution, and hierarchy effects.
- The simulation assumes lost sales, fixed unit cost, simplified replenishment, and no MOQ/capacity constraints.
- Competition data demonstrate methods but do not represent current conditions.
- M5 and Walmart results are comparable methodologically, not row by row.
- Promotion and price associations are predictive rather than causal.

## Recommendations and future work

- Use rolling-origin evaluation over several windows, not a single holdout, for a production gate.
- Forecast at item/store level, reconcile hierarchical forecasts, and aggregate decisions.
- Add Croston/TSB for intermittent demand and probabilistic/quantile forecasts for safety stock.
- Incorporate actual unit costs, margins, lead times, MOQs, shelf life, capacity, and stockout penalties.
- Optimize multi-echelon inventory and test policies in a controlled pilot.
- Monitor WAPE, bias, fill rate, stockouts, DIO, model drift, and data quality by segment.

## Reproducibility and grading-rubric coverage

- Clear notebook headings, comments, sensible variables, and no long raw outputs
- README with findings workflow and notebook link
- Clean project structure and documented data paths
- pandas/numpy data handling and Matplotlib/Seaborn/Plotly plots
- Categorical and continuous visualizations with readable labels
- Multiple regression and classification models
- Time-series cross-validation, explicit GridSearchCV, and randomized hyperparameter search
- Paired Wilcoxon inference and a 10,000-sample paired-bootstrap confidence interval
- Clearly defined metrics and rationale
- Confusion matrices for the valid classification subproblem
- Business interpretation, limitations, and next steps

## Responsible use

The inventory simulation is an educational decision-support prototype. Cost and service assumptions must be replaced with validated business inputs before operational deployment.

