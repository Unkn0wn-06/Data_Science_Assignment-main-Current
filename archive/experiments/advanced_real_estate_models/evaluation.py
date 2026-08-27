"""Shared OOF evaluation, diagnostics, and nested ensemble routines."""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_pinball_loss,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold


def _rmse(actual, predicted) -> float:
    return float(mean_squared_error(actual, predicted) ** 0.5)


def _adjusted_r2(r2: float, observations: int, predictors: int) -> float | None:
    if observations <= predictors + 1:
        return None
    return float(1 - (1 - r2) * (observations - 1) / (observations - predictors - 1))


def _segment_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    error = predicted - actual
    return {
        "count": int(len(actual)),
        "RMSE_RM": _rmse(actual, predicted),
        "MAE_RM": float(mean_absolute_error(actual, predicted)),
        "Mean_Error_RM": float(np.mean(error)),
        "Median_Error_RM": float(np.median(error)),
        "Underpredicted_Percent": float(np.mean(predicted < actual) * 100.0),
    }


def metric_bundle(
    actual,
    predicted,
    predictors: int,
    premium_threshold: float | None = None,
) -> dict:
    """Calculate original-RM headline, scale-normalized, and tail metrics."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if not np.all(np.isfinite(predicted)):
        raise ValueError("Predictions contain non-finite values.")
    if premium_threshold is None:
        premium_threshold = float(np.quantile(actual, 0.95))
    error = predicted - actual
    absolute = np.abs(error)
    r2 = float(r2_score(actual, predicted))
    rmsle = None
    if np.all(actual >= 0) and np.all(predicted >= 0):
        rmsle = _rmse(np.log1p(actual), np.log1p(predicted))
    ape = absolute / actual
    premium = actual >= premium_threshold
    return {
        "RMSE_RM": _rmse(actual, predicted),
        "MAE_RM": float(mean_absolute_error(actual, predicted)),
        "R2": r2,
        "Adjusted_R2": _adjusted_r2(r2, len(actual), predictors),
        "Median_AE_RM": float(np.median(absolute)),
        "RMSLE": rmsle,
        "MAPE_Percent": float(np.mean(ape) * 100.0),
        "Median_APE_Percent": float(np.median(ape) * 100.0),
        "Mean_Error_RM": float(np.mean(error)),
        "Median_Error_RM": float(np.median(error)),
        "Negative_Prediction_Count": int(np.sum(predicted < 0)),
        "top_5_percent_price_threshold_RM": float(premium_threshold),
        "top_5_percent": _segment_metrics(actual[premium], predicted[premium]),
        "remaining_95_percent": _segment_metrics(actual[~premium], predicted[~premium]),
    }


def _fold_summary(actual, predicted) -> dict:
    return {
        "RMSE_RM": _rmse(actual, predicted),
        "MAE_RM": float(mean_absolute_error(actual, predicted)),
        "R2": float(r2_score(actual, predicted)),
    }


def _finish_evaluation(
    name: str,
    actual: np.ndarray,
    prediction: np.ndarray,
    predictors: int,
    fold_rows: list[dict],
    train_rows: list[dict],
    metadata: dict | None = None,
) -> dict:
    metrics = metric_bundle(actual, prediction, predictors)
    training = {
        metric: float(np.mean([row[metric] for row in train_rows]))
        for metric in ("RMSE_RM", "MAE_RM", "R2")
    }
    result = {
        "name": name,
        "metrics": metrics,
        "generalization_gap": {
            "Training_RMSE_RM": training["RMSE_RM"],
            "CV_RMSE_RM": metrics["RMSE_RM"],
            "RMSE_gap_RM": metrics["RMSE_RM"] - training["RMSE_RM"],
            "Training_MAE_RM": training["MAE_RM"],
            "CV_MAE_RM": metrics["MAE_RM"],
            "MAE_gap_RM": metrics["MAE_RM"] - training["MAE_RM"],
            "Training_R2": training["R2"],
            "CV_R2": metrics["R2"],
            "R2_gap": training["R2"] - metrics["R2"],
        },
        "folds": fold_rows,
    }
    if metadata:
        result.update(metadata)
    return {"result": result, "prediction": prediction}


def evaluate_fixed(
    name: str,
    estimator,
    X: pd.DataFrame,
    y,
    folds,
    predictors: int,
) -> dict:
    """Evaluate a fixed estimator on shared outer folds and record train gaps."""
    actual = np.asarray(y, dtype=float)
    prediction = np.empty(len(actual), dtype=float)
    fold_rows, train_rows, lambdas = [], [], []
    for fold_number, (train_index, validation_index) in enumerate(folds, start=1):
        started = time.perf_counter()
        fitted = clone(estimator).fit(X.iloc[train_index], actual[train_index])
        fit_seconds = time.perf_counter() - started
        validation_prediction = fitted.predict(X.iloc[validation_index])
        training_prediction = fitted.predict(X.iloc[train_index])
        prediction[validation_index] = validation_prediction
        validation_metrics = _fold_summary(actual[validation_index], validation_prediction)
        training_metrics = _fold_summary(actual[train_index], training_prediction)
        fold_rows.append(
            {
                "fold": fold_number,
                "training_rows": int(len(train_index)),
                "validation_rows": int(len(validation_index)),
                "fit_seconds": float(fit_seconds),
                **validation_metrics,
            }
        )
        train_rows.append(training_metrics)
        fold_lambda = getattr(fitted, "boxcox_lambda_", None)
        if fold_lambda is not None:
            lambdas.append(float(fold_lambda))
    metadata = {"fold_boxcox_lambdas": lambdas} if lambdas else None
    return _finish_evaluation(
        name, actual, prediction, predictors, fold_rows, train_rows, metadata
    )


def evaluate_nested_candidates(
    name: str,
    estimators: list,
    candidate_params: list[dict],
    X: pd.DataFrame,
    y,
    folds,
    predictors: int,
    inner_splits: int = 3,
) -> dict:
    """Select parameters inside each outer training fold, then score untouched rows."""
    actual = np.asarray(y, dtype=float)
    prediction = np.empty(len(actual), dtype=float)
    fold_rows, train_rows, selected = [], [], []
    all_positions = np.arange(len(actual))
    for fold_number, (train_index, validation_index) in enumerate(folds, start=1):
        inner = list(
            KFold(inner_splits, shuffle=True, random_state=42).split(train_index)
        )
        candidate_scores = []
        for estimator in estimators:
            rmses, maes = [], []
            for inner_train, inner_validation in inner:
                train_rows_index = train_index[inner_train]
                validation_rows_index = train_index[inner_validation]
                fitted = clone(estimator).fit(
                    X.iloc[train_rows_index], actual[train_rows_index]
                )
                inner_prediction = fitted.predict(X.iloc[validation_rows_index])
                rmses.append(_rmse(actual[validation_rows_index], inner_prediction))
                maes.append(
                    mean_absolute_error(actual[validation_rows_index], inner_prediction)
                )
            candidate_scores.append(
                {"RMSE_RM": float(np.mean(rmses)), "MAE_RM": float(np.mean(maes))}
            )
        best_index = min(
            range(len(candidate_scores)),
            key=lambda index: (
                candidate_scores[index]["RMSE_RM"],
                candidate_scores[index]["MAE_RM"],
            ),
        )
        started = time.perf_counter()
        fitted = clone(estimators[best_index]).fit(
            X.iloc[train_index], actual[train_index]
        )
        fit_seconds = time.perf_counter() - started
        validation_prediction = fitted.predict(X.iloc[validation_index])
        training_prediction = fitted.predict(X.iloc[train_index])
        prediction[validation_index] = validation_prediction
        validation_metrics = _fold_summary(actual[validation_index], validation_prediction)
        training_metrics = _fold_summary(actual[train_index], training_prediction)
        fold_rows.append(
            {
                "fold": fold_number,
                "training_rows": int(len(train_index)),
                "validation_rows": int(len(validation_index)),
                "fit_seconds": float(fit_seconds),
                "selected_candidate": int(best_index),
                "inner_candidate_scores": candidate_scores,
                **validation_metrics,
            }
        )
        train_rows.append(training_metrics)
        selected.append(best_index)
        print(
            f"{name}: completed outer fold {fold_number}/{len(folds)} "
            f"(selected candidate {best_index}).",
            flush=True,
        )
    counts = {str(index): selected.count(index) for index in sorted(set(selected))}
    return _finish_evaluation(
        name,
        actual,
        prediction,
        predictors,
        fold_rows,
        train_rows,
        {
            "tuning": {
                "method": (
                    f"nested {inner_splits}-fold selection inside each outer "
                    "training fold"
                ),
                "selection_metric": "RMSE_RM then MAE_RM",
                "candidate_parameters": candidate_params,
                "selected_candidate_counts": counts,
            }
        },
    )


def attach_baseline_comparison(result: dict, baseline: dict) -> None:
    current = result["metrics"]
    reference = baseline["metrics"]
    rmse_delta = current["RMSE_RM"] - reference["RMSE_RM"]
    mae_delta = current["MAE_RM"] - reference["MAE_RM"]
    premium_delta = (
        current["top_5_percent"]["RMSE_RM"]
        - reference["top_5_percent"]["RMSE_RM"]
    )
    result["comparison_vs_random_forest"] = {
        "RMSE_difference_RM": float(rmse_delta),
        "RMSE_percentage_change": float(rmse_delta / reference["RMSE_RM"] * 100),
        "MAE_difference_RM": float(mae_delta),
        "MAE_percentage_change": float(mae_delta / reference["MAE_RM"] * 100),
        "R2_difference": float(current["R2"] - reference["R2"]),
        "Top5_RMSE_difference_RM": float(premium_delta),
        "Top5_RMSE_percentage_change": float(
            premium_delta / reference["top_5_percent"]["RMSE_RM"] * 100
        ),
    }


def residual_correlations(actual, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    residuals = {
        name: np.asarray(prediction, dtype=float) - np.asarray(actual, dtype=float)
        for name, prediction in predictions.items()
    }
    return pd.DataFrame(residuals).corr()


def _blend_weights(meta_features: np.ndarray, target: np.ndarray) -> np.ndarray:
    count = meta_features.shape[1]
    initial = np.full(count, 1.0 / count)
    fitted = minimize(
        lambda weights: np.mean(np.square(meta_features @ weights - target)),
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints={"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not fitted.success:
        return initial
    weights = np.maximum(fitted.x, 0.0)
    return weights / weights.sum()


def _select_complementary(
    actual: np.ndarray, inner_predictions: dict[str, np.ndarray], maximum: int = 3
) -> list[str]:
    scores = {
        name: _rmse(actual, prediction)
        for name, prediction in inner_predictions.items()
    }
    ordered = sorted(scores, key=scores.get)
    selected = [ordered[0]]
    residuals = {
        name: prediction - actual for name, prediction in inner_predictions.items()
    }
    while len(selected) < min(maximum, len(ordered)):
        candidates = [name for name in ordered if name not in selected]
        eligible = [name for name in candidates if scores[name] <= scores[ordered[0]] * 1.10]
        if not eligible:
            eligible = candidates
        next_name = min(
            eligible,
            key=lambda name: np.mean(
                [abs(np.corrcoef(residuals[name], residuals[old])[0, 1]) for old in selected]
            ),
        )
        selected.append(next_name)
    return selected


def evaluate_nested_ensembles(
    base_estimators: dict[str, object],
    X: pd.DataFrame,
    y,
    folds,
    predictors: int,
) -> dict[str, dict]:
    """Build blend and stack with inner-OOF meta-features in each outer fold."""
    actual = np.asarray(y, dtype=float)
    blend_prediction = np.empty(len(actual), dtype=float)
    stack_prediction = np.empty(len(actual), dtype=float)
    blend_folds, stack_folds, blend_train, stack_train, details = [], [], [], [], []
    for fold_number, (train_index, validation_index) in enumerate(folds, start=1):
        inner_splits = list(KFold(4, shuffle=True, random_state=42).split(train_index))
        inner_predictions = {}
        validation_predictions = {}
        for name, estimator in base_estimators.items():
            inner_oof = np.empty(len(train_index), dtype=float)
            for inner_train, inner_validation in inner_splits:
                fitted = clone(estimator).fit(
                    X.iloc[train_index[inner_train]], actual[train_index[inner_train]]
                )
                inner_oof[inner_validation] = fitted.predict(
                    X.iloc[train_index[inner_validation]]
                )
            inner_predictions[name] = inner_oof
        selected = _select_complementary(actual[train_index], inner_predictions)
        for name in selected:
            fitted = clone(base_estimators[name]).fit(
                X.iloc[train_index], actual[train_index]
            )
            validation_predictions[name] = fitted.predict(X.iloc[validation_index])

        inner_matrix = np.column_stack([inner_predictions[name] for name in selected])
        validation_matrix = np.column_stack(
            [validation_predictions[name] for name in selected]
        )
        weights = _blend_weights(inner_matrix, actual[train_index])
        blend_validation = validation_matrix @ weights
        blend_training = inner_matrix @ weights
        ridge = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(
            inner_matrix, actual[train_index]
        )
        stack_validation = ridge.predict(validation_matrix)
        stack_training = ridge.predict(inner_matrix)
        blend_prediction[validation_index] = blend_validation
        stack_prediction[validation_index] = stack_validation

        blend_validation_metrics = _fold_summary(
            actual[validation_index], blend_validation
        )
        stack_validation_metrics = _fold_summary(
            actual[validation_index], stack_validation
        )
        blend_folds.append({"fold": fold_number, **blend_validation_metrics})
        stack_folds.append({"fold": fold_number, **stack_validation_metrics})
        blend_train.append(_fold_summary(actual[train_index], blend_training))
        stack_train.append(_fold_summary(actual[train_index], stack_training))
        details.append(
            {
                "fold": fold_number,
                "selected_models": selected,
                "blend_weights": {
                    name: float(weight) for name, weight in zip(selected, weights)
                },
                "stack_alpha": float(ridge.alpha_),
                "stack_coefficients": {
                    name: float(value) for name, value in zip(selected, ridge.coef_)
                },
                "stack_intercept": float(ridge.intercept_),
            }
        )
    blend = _finish_evaluation(
        "Best Blend", actual, blend_prediction, predictors, blend_folds, blend_train,
        {"ensemble_folds": details, "leakage_safe_nested_meta_features": True},
    )
    stack = _finish_evaluation(
        "Best Stack", actual, stack_prediction, predictors, stack_folds, stack_train,
        {"ensemble_folds": details, "leakage_safe_nested_meta_features": True},
    )
    return {"best_blend": blend, "best_stack": stack}


def quantile_diagnostics(actual, predictions: dict[float, np.ndarray]) -> dict:
    """Report pinball loss and the untouched P10-P90 interval behavior."""
    actual = np.asarray(actual, dtype=float)
    diagnostics = {
        f"P{int(alpha * 100)}_pinball_loss_RM": float(
            mean_pinball_loss(actual, prediction, alpha=alpha)
        )
        for alpha, prediction in predictions.items()
    }
    lower, median, upper = predictions[0.1], predictions[0.5], predictions[0.9]
    diagnostics.update(
        {
            "P10_P90_coverage_Percent": float(
                np.mean((actual >= lower) & (actual <= upper)) * 100
            ),
            "P10_P90_mean_interval_width_RM": float(np.mean(upper - lower)),
            "P10_above_P50_count": int(np.sum(lower > median)),
            "P50_above_P90_count": int(np.sum(median > upper)),
        }
    )
    return diagnostics
