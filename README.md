# Malaysian Residential Listing Price Prediction

This final submission predicts Malaysian residential listing prices and compares
four assignment models using leakage-safe validation. The primary evidence is a
shared Scenario B group-safe five-fold evaluation that keeps exact duplicates and
strong description-matched repeat listings within one fold.

## Final assignment models

1. **Ridge Regression** — baseline
2. **Random Forest**
3. **Gradient Boosting**
4. **LightGBM + Position Features** — selected deployment model

All four models learn price per square foot (PPSF) and reconstruct total listing
price by multiplying predicted PPSF by property size. Hyperparameters are frozen;
the final workflow performs no retuning.

The selected LightGBM model adds deterministic size/location interactions and
five target-free indicators extracted from listing descriptions:

- High floor
- Low floor
- Top floor
- Balcony
- Large balcony

It was selected because it provides competitive, balanced RMSE and MAE while
incorporating meaningful property-position information. Its differences from the
strongest competing models were not statistically significant at the 95%
confidence level.

## Final model performance

Regenerated Scenario B out-of-fold total-price metrics (RM):

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Ridge Regression | 145,209.19 | 75,561.38 | 0.8060 |
| Random Forest | 125,115.27 | 61,434.97 | 0.8560 |
| Gradient Boosting | 128,882.88 | 63,363.76 | 0.8472 |
| LightGBM + Position Features | 122,730.62 | 61,914.62 | 0.8614 |

## Primary validation

The final comparison uses the saved assignments in
`experiments/repeat_group_sensitivity/scenario_b_fold_assignments.csv`:

- Scenario B: Level 1 + Level 2 repeat protection
- five group-safe folds
- 3,791 canonical listings
- identical validation rows for every model
- total-price RMSE, MAE, R², adjusted R², median absolute error, and premium-tail
  diagnostics

Scenario B is the primary reporting assumption because it protects clear
duplicate/reposted listings without assuming every Level-3 structured match is
the same physical unit. Scenario C remains available in the sensitivity
experiment as a conservative bound.

Saved final artifacts are under `results/final_models/`:

- `model_comparison.csv` and `model_comparison.json`
- `oof_predictions.csv`
- `fold_metrics.csv`
- `metadata.json`
- `feature_importance.csv`

The completed upper-tail study is exported separately to
`results/outlier_trimming/`. The presentation now defaults to the **10% restricted
market scope**, while preserving 0%, 0.5%, 1%, 2.5%, 5%, and 10% as the six
validated choices. The separate training-only experiment still records 0% as its
full-market recommendation; the dashboard does not reinterpret that historical
result. `retained_cv_summary.csv` records the actual training and validation
counts for each of the five preserved Scenario B folds at every scope.

## Data

- `data/raw/houses.csv`: unchanged 4,000-row source data
- `data/processed/enhanced_city_dataset.csv`: canonical 3,791-row final dataset
- `data/processed/production_prepared_dataset.csv`: retained for production
  cleaning and assignment-model smoke tests

No premium observations or repeat-like listings are deleted by the final
evaluation.

## Streamlit application

The application has one maintained implementation in `prototype/app.py`; root
`app.py` is a thin launcher. It provides:

1. Model Comparison with an independent market-scope selector and source-backed tuning details
2. Feature Importance
3. Actual vs Predicted using saved Scenario B OOF predictions
4. Outlier & Trimming Analysis using saved presentation artifacts, including
   retained-population five-fold training/validation counts
5. Live House Price Predictor comparing all four selected-scope models with their 0% full-market counterparts

The dashboard loads saved final metrics instead of rerunning cross-validation.
The trimming page reads only from `results/outlier_trimming/` and performs no
model fitting. Its source experiment can therefore be archived later without
breaking the application.
The comparison page reads saved validation results and never fits models. For
live inference, each requested scope fits the four frozen deployment families
once per Streamlit process through `st.cache_resource`; ordinary property-form
changes do not retrain them. Both market-scope selectors default to 10%.

Install and launch:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Rebuilding final results

From the repository root:

```bash
python scripts/build_final_model_results.py
python scripts/build_outlier_trimming_results.py
```

The script fails loudly if the canonical data and Scenario B assignments do not
contain the same 3,791 listing IDs, if a repeat group crosses folds, or if OOF
coverage is incomplete.

The outlier/trimming builder validates and copies completed experiment outputs;
it never retrains a model.

## Tests

```bash
python -m unittest discover -s tests -v
python -m unittest experiments.repeat_group_sensitivity.test_invariants -v
```

The active tests cover artifact recomputation, fold safety, deployment training
on all rows, feature-schema alignment, unseen categories, position-regex
extraction, live prediction, and all five Streamlit navigation pages.

## Repository structure

```text
.
|-- app.py
|-- configs/
|-- data/
|-- experiments/
|   |-- repeat_group_sensitivity/
|   `-- upper_tail_trimming/
|-- prototype/
|   `-- app.py
|-- results/
|   |-- final_models/
|   `-- outlier_trimming/
|-- scripts/
|   |-- build_final_model_results.py
|   `-- build_outlier_trimming_results.py
|-- src/
|   |-- cleaning/
|   `-- models/
|       `-- final/
|-- tests/
`-- archive/
```

The final model uses maintained helpers under `src/models/final/`. The retained
`src/experimental_support/` package supports historical sensitivity evidence
only and is not imported by the application or final model workflow. Historical
experiments remain under `archive/` and are not imported by active code.
