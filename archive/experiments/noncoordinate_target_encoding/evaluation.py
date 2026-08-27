"""Shared-fold OOF evaluation, reference deltas, and paired bootstrap tests."""

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
    listing_ids: pd.Series,
) -> dict:
    """Produce fixed-index OOF predictions with all learned features inside folds."""
    actual = np.asarray(y, dtype=float)
    threshold = float(np.quantile(actual, 0.95))
    prediction = np.empty(len(actual), dtype=float)
    fold_rows = []
    oof_rows = []
    training_rows = []
    smoothing = []
    weight_summaries = []

    for fold_number, (train_index, validation_index) in enumerate(folds, start=1):
        started = time.perf_counter()
        fitted = clone(estimator).fit(X.iloc[train_index], actual[train_index])
        fit_seconds = time.perf_counter() - started
        validation_prediction = np.asarray(
            fitted.predict(X.iloc[validation_index]), dtype=float
        )
        if hasattr(fitted, "predict_training_oof_features"):
            training_prediction = fitted.predict_training_oof_features(X.iloc[train_index])
        else:
            training_prediction = np.asarray(fitted.predict(X.iloc[train_index]), dtype=float)
        prediction[validation_index] = validation_prediction

        validation_metrics = metric_bundle(
            actual[validation_index], validation_prediction, predictor_count, threshold
        )
        training_metrics = {
            "RMSE_RM": rmse(actual[train_index], training_prediction),
            "MAE_RM": float(mean_absolute_error(actual[train_index], training_prediction)),
            "R2": float(r2_score(actual[train_index], training_prediction)),
        }
        training_rows.append(training_metrics)
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
            smoothing.append(
                {
                    "fold": fold_number,
                    "selected_m": float(fitted.selected_m_),
                    "proxy_scores": fitted.smoothing_proxy_scores_,
                }
            )
        if hasattr(fitted, "sample_weight_summary_"):
            weight_summaries.append(
                {"fold": fold_number, **fitted.sample_weight_summary_}
            )
        fold_rows.append(record)

        residual = validation_prediction - actual[validation_index]
        for local_position, row_index in enumerate(validation_index):
            oof_rows.append(
                {
                    "row_id": listing_ids.iloc[row_index],
                    "row_index": int(row_index),
                    "fold": fold_number,
                    "actual_price_RM": float(actual[row_index]),
                    "predicted_price_RM": float(validation_prediction[local_position]),
                    "residual_RM": float(residual[local_position]),
                    "absolute_error_RM": float(abs(residual[local_position])),
                    "premium_flag": bool(actual[row_index] >= threshold),
                    "variant": name,
                }
            )

    metrics = metric_bundle(actual, prediction, predictor_count, threshold)
    training = {
        key: float(np.mean([record[key] for record in training_rows]))
        for key in ("RMSE_RM", "MAE_RM", "R2")
    }
    result = {
        "name": name,
        "metrics": metrics,
        "generalization_gap": {
            "Training_RMSE_RM": training["RMSE_RM"],
            "OOF_RMSE_RM": metrics["RMSE_RM"],
            "RMSE_gap_RM": metrics["RMSE_RM"] - training["RMSE_RM"],
            "Training_MAE_RM": training["MAE_RM"],
            "OOF_MAE_RM": metrics["MAE_RM"],
            "MAE_gap_RM": metrics["MAE_RM"] - training["MAE_RM"],
            "Training_R2": training["R2"],
            "OOF_R2": metrics["R2"],
            "R2_gap": training["R2"] - metrics["R2"],
        },
        "folds": fold_rows,
        "predictor_count": int(predictor_count),
    }
    if smoothing:
        result["smoothing"] = {
            "candidate_m_values": [5.0, 10.0, 20.0, 50.0, 100.0],
            "selection": "outer-training-only inner-OOF PPSF proxy RMSE",
            "by_fold": smoothing,
        }
    if weight_summaries:
        result["sample_weights"] = weight_summaries
    return {
        "result": result,
        "prediction": prediction,
        "fold_rows": fold_rows,
        "oof_rows": oof_rows,
    }


def reference_deltas(metrics: dict, reference_metrics: dict) -> dict:
    """Return signed candidate-minus-reference changes; negative error is better."""
    rmse_difference = metrics["RMSE_RM"] - reference_metrics["RMSE_RM"]
    mae_difference = metrics["MAE_RM"] - reference_metrics["MAE_RM"]
    top_rmse_difference = (
        metrics["top_5_percent"]["RMSE_RM"]
        - reference_metrics["top_5_percent"]["RMSE_RM"]
    )
    top_mae_difference = (
        metrics["top_5_percent"]["MAE_RM"]
        - reference_metrics["top_5_percent"]["MAE_RM"]
    )
    return {
        "RMSE_difference_RM": float(rmse_difference),
        "RMSE_percentage_change": float(
            rmse_difference / reference_metrics["RMSE_RM"] * 100.0
        ),
        "MAE_difference_RM": float(mae_difference),
        "MAE_percentage_change": float(
            mae_difference / reference_metrics["MAE_RM"] * 100.0
        ),
        "R2_difference": float(metrics["R2"] - reference_metrics["R2"]),
        "Top5_RMSE_difference_RM": float(top_rmse_difference),
        "Top5_RMSE_percentage_change": float(
            top_rmse_difference / reference_metrics["top_5_percent"]["RMSE_RM"] * 100.0
        ),
        "Top5_MAE_difference_RM": float(top_mae_difference),
        "Top5_MAE_percentage_change": float(
            top_mae_difference / reference_metrics["top_5_percent"]["MAE_RM"] * 100.0
        ),
    }


def paired_bootstrap(
    actual,
    candidate_prediction,
    reference_prediction,
    premium_threshold: float,
    samples: int = 5000,
    random_state: int = 42,
) -> dict:
    """Paired fixed-OOF CIs for RMSE, MAE, and premium RMSE differences."""
    actual = np.asarray(actual, dtype=float)
    candidate = np.asarray(candidate_prediction, dtype=float)
    reference = np.asarray(reference_prediction, dtype=float)
    premium = actual >= premium_threshold
    generator = np.random.default_rng(random_state)
    all_positions = np.arange(len(actual))
    premium_positions = np.flatnonzero(premium)
    rmse_differences = np.empty(samples)
    mae_differences = np.empty(samples)
    premium_rmse_differences = np.empty(samples)
    for iteration in range(samples):
        sample = generator.choice(all_positions, len(all_positions), replace=True)
        premium_sample = generator.choice(
            premium_positions, len(premium_positions), replace=True
        )
        rmse_differences[iteration] = (
            rmse(actual[sample], candidate[sample])
            - rmse(actual[sample], reference[sample])
        )
        mae_differences[iteration] = (
            mean_absolute_error(actual[sample], candidate[sample])
            - mean_absolute_error(actual[sample], reference[sample])
        )
        premium_rmse_differences[iteration] = (
            rmse(actual[premium_sample], candidate[premium_sample])
            - rmse(actual[premium_sample], reference[premium_sample])
        )

    def summary(values, point):
        interval = np.quantile(values, [0.025, 0.975]).tolist()
        return {
            "difference_RM": float(point),
            "interval_95_percent_RM": interval,
            "statistically_significant": bool(interval[1] < 0 or interval[0] > 0),
        }

    return {
        "samples": int(samples),
        "difference_definition": "candidate metric minus reference metric",
        "negative_is_better": True,
        "RMSE": summary(
            rmse_differences,
            rmse(actual, candidate) - rmse(actual, reference),
        ),
        "MAE": summary(
            mae_differences,
            mean_absolute_error(actual, candidate)
            - mean_absolute_error(actual, reference),
        ),
        "Top5_RMSE": summary(
            premium_rmse_differences,
            rmse(actual[premium], candidate[premium])
            - rmse(actual[premium], reference[premium]),
        ),
        "limitation": "Fixed-OOF bootstrap; model-refitting and candidate-selection uncertainty are not included.",
    }
