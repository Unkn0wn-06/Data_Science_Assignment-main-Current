# Repository Cleanup Summary

The repository was reorganized into a submission-ready active project while
preserving historical evidence. No historical work was deleted; only generated
cache files were removed.

## Active final workflow

- `experiments/repeat_group_sensitivity/` is the sole active experiment.
- `data/processed/enhanced_city_dataset.csv` remains the canonical 3,791-row
  final dataset.
- `src/experimental_support/` contains the exact reusable feature, grouping,
  LightGBM, and fold-safe target-encoding logic required by the final experiment.
- `prototype/app.py` and root `app.py` remain the Streamlit implementation and
  deployment entry point.
- `results/final_models/` is now the sole active dashboard comparison directory;
  the superseded shuffled-CV `results/enhanced_city/` tree was archived under
  `archive/legacy_results/enhanced_city/`.
- `scripts/build_enhanced_city_dataset.py` and
  `scripts/run_enhanced_city_comparison.py` remain the active rebuild commands.

## Historical experiments archived

The following complete directories were moved from `experiments/` to
`archive/experiments/`:

- `advanced_real_estate_models`
- `city_feature`
- `completion_year_ablation`
- `data_validity_audit`
- `description_text_features`
- `enhanced_models`
- `log_target`
- `missing_data_quality`
- `noncoordinate_target_encoding`
- `premium_mixture_of_experts`
- `price_per_square_foot`
- `repeat_listing_leakage`
- `spatial_target_encoding`
- `winsorization`
- `winsorization_enhanced_city`

Their source, results, figures, tests, and OOF evidence remain intact except for
generated Python caches.

## Other archived material

- `archive/legacy_root_files/`: duplicate root-level dataset copies
  (`houses.csv`, `production_prepared_dataset.csv`, and
  `recleaned_prepared_dataset.csv`). Canonical copies remain under `data/`.
- `archive/legacy_scripts/`: historical cleaning, standard-comparison, and
  selected-old-model entry points.
- `archive/legacy_scripts/model_tuning/`: opt-in historical tuning modules removed
  from the active model tree.
- `archive/legacy_tests/`: the advanced-experiment-only unit test.
- `archive/legacy_results/`: historical production-comparison and old best-model
  pointer directories.
- `archive/legacy_data/`: pre-existing superseded prepared datasets.

## Active test adjustment

`tests/models/test_enhanced_city.py` retains its canonical-schema and dashboard
artifact checks. Its historical Winsorization comparison was removed because the
referenced experiment is now archived.

## Safety and validation

Before cleanup, the canonical dataset and all final experiment outputs were
hashed. After cleanup, active imports, project tests, final experiment invariants,
Streamlit import/startup behavior, archived-path references, dataset hashes, and
output hashes were checked. The final report records the executed commands and
results.

Validation completed successfully:

- Active project suite: 4/4 tests passed.
- Final sensitivity invariant suite: 15/15 tests passed.
- Bare `app.py` import completed through model training and rendering code.
- All active support and final experiment modules imported successfully.
- Relocated model helpers reproduced saved fold predictions with a maximum
  absolute delta below `1.4e-9 RM`.
- No active Python source imports an archived module.
- Canonical dataset SHA-256 remained
  `4A295007FC5FDF6DEF33A612797606DFB60D811E7B19ADE930466033A1FD66CF`.
