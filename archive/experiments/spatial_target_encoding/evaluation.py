"""Shared OOF evaluation and paired comparison for the focused experiment."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from experiments.advanced_real_estate_models.evaluation import metric_bundle


def rmse(actual, predicted) -> float:
    return float(mean_squared_error(actual, predicted) ** 0.5)


def evaluate_variant(
    name: str,
    estimator,
    X: pd.DataFrame,
    y,
    folds,
    predictor_count: int,
    listing_ids,
) -> dict:
    """Fit every learned feature inside shared outer folds and return long OOF rows."""
    actual = np.asarray(y, dtype=float)
    premium_threshold = float(np.quantile(actual, 0.95))
    prediction = np.empty(len(actual), dtype=float)
    fold_records = []
    training_records = []
    selected_m_values = []
    smoothing_scores = []
    oof_rows = []
    for fold_number, (train_index, validation_index) in enumerate(folds, start=1):
        started = time.perf_counter()
        fitted = clone(estimator).fit(X.iloc[train_index], actual[train_index])
        fit_seconds = time.perf_counter() - started
        validation_prediction = np.asarray(
            fitted.predict(X.iloc[validation_index]), dtype=float
        )
        if hasattr(fitted, "predict_training_oof_features"):
            training_prediction = fitted.predict_training_oof_features(
                X.iloc[train_index]
            )
        else:
            training_prediction = np.asarray(
                fitted.predict(X.iloc[train_index]), dtype=float
            )
        prediction[validation_index] = validation_prediction
        validation_metrics = metric_bundle(
            actual[validation_index],
            validation_prediction,
            predictor_count,
            premium_threshold,
        )
        training_metrics = {
            "RMSE_RM": rmse(actual[train_index], training_prediction),
            "MAE_RM": float(
                mean_absolute_error(actual[train_index], training_prediction)
            ),
            "R2": float(r2_score(actual[train_index], training_prediction)),
        }
        record = {
            "variant": name,
            "fold": fold_number,
            "training_rows": int(len(train_index)),
            "validation_rows": int(len(validation_index)),
            "fit_seconds": float(fit_seconds),
            "RMSE_RM": validation_metrics["RMSE_RM"],
            "MAE_RM": validation_metrics["MAE_RM"],
            "R2": validation_metrics["R2"],
            "Top5_RMSE_RM": validation_metrics["top_5_percent"]["RMSE_RM"],
            "Top5_MAE_RM": validation_metrics["top_5_percent"]["MAE_RM"],
        }
        if hasattr(fitted, "selected_m_"):
            record["selected_m"] = float(fitted.selected_m_)
            selected_m_values.append(float(fitted.selected_m_))
            smoothing_scores.append(
                {
                    "fold": fold_number,
                    "scores": fitted.smoothing_proxy_scores_,
                }
            )
        fold_records.append(record)
        training_records.append(training_metrics)
        error = validation_prediction - actual[validation_index]
        for local, row_index in enumerate(validation_index):
            oof_rows.append(
                {
                    "listing_id": listing_ids.iloc[row_index],
                    "row_index": int(row_index),
                    "fold": fold_number,
                    "actual_price_RM": float(actual[row_index]),
                    "predicted_price_RM": float(validation_prediction[local]),
                    "residual_RM": float(error[local]),
                    "absolute_error_RM": float(abs(error[local])),
                    "premium_flag": bool(actual[row_index] >= premium_threshold),
                    "model_variant": name,
                }
            )

    metrics = metric_bundle(actual, prediction, predictor_count, premium_threshold)
    training = {
        key: float(np.mean([row[key] for row in training_records]))
        for key in ("RMSE_RM", "MAE_RM", "R2")
    }
    result = {
        "name": name,
        "metrics": metrics,
        "folds": fold_records,
        "generalization_gap": {
            "Training_RMSE_RM": training["RMSE_RM"],
            "Validation_RMSE_RM": metrics["RMSE_RM"],
            "RMSE_gap_RM": metrics["RMSE_RM"] - training["RMSE_RM"],
            "Training_MAE_RM": training["MAE_RM"],
            "Validation_MAE_RM": metrics["MAE_RM"],
            "MAE_gap_RM": metrics["MAE_RM"] - training["MAE_RM"],
            "Training_R2": training["R2"],
            "Validation_R2": metrics["R2"],
            "R2_gap": training["R2"] - metrics["R2"],
        },
        "predictor_count": int(predictor_count),
    }
    if selected_m_values:
        result["smoothing"] = {
            "candidate_m_values": [5.0, 10.0, 20.0, 50.0, 100.0],
            "selection": "training-only inner-OOF PPSF proxy RMSE",
            "selected_m_by_fold": selected_m_values,
            "proxy_scores_by_fold": smoothing_scores,
        }
    return {
        "result": result,
        "prediction": prediction,
        "oof_rows": oof_rows,
        "fold_rows": fold_records,
    }


def attach_reference_comparisons(result: dict, random_forest: dict, lightgbm: dict):
    """Attach signed changes where negative RMSE/MAE values are improvements."""
    current = result["metrics"]
    comparisons = {}
    for key, reference in (
        ("random_forest", random_forest),
        ("lightgbm_interaction", lightgbm),
    ):
        reference_metrics = reference["metrics"]
        rmse_delta = current["RMSE_RM"] - reference_metrics["RMSE_RM"]
        mae_delta = current["MAE_RM"] - reference_metrics["MAE_RM"]
        top_rmse_delta = (
            current["top_5_percent"]["RMSE_RM"]
            - reference_metrics["top_5_percent"]["RMSE_RM"]
        )
        top_mae_delta = (
            current["top_5_percent"]["MAE_RM"]
            - reference_metrics["top_5_percent"]["MAE_RM"]
        )
        comparisons[key] = {
            "RMSE_difference_RM": float(rmse_delta),
            "RMSE_percentage_change": float(
                rmse_delta / reference_metrics["RMSE_RM"] * 100.0
            ),
            "MAE_difference_RM": float(mae_delta),
            "MAE_percentage_change": float(
                mae_delta / reference_metrics["MAE_RM"] * 100.0
            ),
            "R2_difference": float(current["R2"] - reference_metrics["R2"]),
            "Top5_RMSE_difference_RM": float(top_rmse_delta),
            "Top5_RMSE_percentage_change": float(
                top_rmse_delta
                / reference_metrics["top_5_percent"]["RMSE_RM"]
                * 100.0
            ),
            "Top5_MAE_difference_RM": float(top_mae_delta),
            "Top5_MAE_percentage_change": float(
                top_mae_delta
                / reference_metrics["top_5_percent"]["MAE_RM"]
                * 100.0
            ),
        }
    result["comparisons"] = comparisons


def paired_bootstrap(
    actual,
    candidate_prediction,
    reference_prediction,
    samples: int = 5000,
    random_state: int = 42,
) -> dict:
    """Bootstrap paired fixed-OOF metric differences; negative means better."""
    actual = np.asarray(actual, dtype=float)
    candidate = np.asarray(candidate_prediction, dtype=float)
    reference = np.asarray(reference_prediction, dtype=float)
    generator = np.random.default_rng(random_state)
    positions = np.arange(len(actual))
    rmse_differences = np.empty(samples, dtype=float)
    mae_differences = np.empty(samples, dtype=float)
    for iteration in range(samples):
        sample = generator.choice(positions, len(positions), replace=True)
        rmse_differences[iteration] = (
            rmse(actual[sample], candidate[sample])
            - rmse(actual[sample], reference[sample])
        )
        mae_differences[iteration] = (
            mean_absolute_error(actual[sample], candidate[sample])
            - mean_absolute_error(actual[sample], reference[sample])
        )
    return {
        "samples": samples,
        "difference_definition": "candidate metric minus LightGBM interaction baseline metric",
        "negative_is_better": True,
        "RMSE_difference_RM": float(rmse(actual, candidate) - rmse(actual, reference)),
        "RMSE_95_percent_interval_RM": np.quantile(
            rmse_differences, [0.025, 0.975]
        ).tolist(),
        "MAE_difference_RM": float(
            mean_absolute_error(actual, candidate)
            - mean_absolute_error(actual, reference)
        ),
        "MAE_95_percent_interval_RM": np.quantile(
            mae_differences, [0.025, 0.975]
        ).tolist(),
        "RMSE_statistically_significant": bool(
            np.quantile(rmse_differences, 0.975) < 0.0
            or np.quantile(rmse_differences, 0.025) > 0.0
        ),
        "MAE_statistically_significant": bool(
            np.quantile(mae_differences, 0.975) < 0.0
            or np.quantile(mae_differences, 0.025) > 0.0
        ),
        "limitation": "Conditions on fixed OOF model fits; it does not include model-refitting uncertainty.",
    }
