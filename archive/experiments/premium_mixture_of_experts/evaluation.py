"""Evaluation utilities for the premium mixture-of-experts experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from experiments.advanced_real_estate_models.evaluation import metric_bundle


PRICE_BANDS = (
    ("P00_P50", 0.00, 0.50),
    ("P50_P80", 0.50, 0.80),
    ("P80_P90", 0.80, 0.90),
    ("P90_P95", 0.90, 0.95),
    ("P95_P99", 0.95, 0.99),
    ("P99_P100", 0.99, 1.00),
)


def rmse(actual, predicted) -> float:
    return float(mean_squared_error(actual, predicted) ** 0.5)


def regression_summary(actual, predicted) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    return {
        "count": int(len(actual)),
        "RMSE_RM": rmse(actual, predicted),
        "MAE_RM": float(mean_absolute_error(actual, predicted)),
        "Mean_Error_RM": float(np.mean(error)),
        "Median_Error_RM": float(np.median(error)),
    }


def price_band_metrics(actual, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    """Score fixed global price quantile bands; boundaries are descriptive only."""
    actual = np.asarray(actual, dtype=float)
    quantiles = {q: float(np.quantile(actual, q)) for q in {v for _, a, b in PRICE_BANDS for v in (a, b)}}
    rows = []
    for model, predicted in predictions.items():
        predicted = np.asarray(predicted, dtype=float)
        for band, low_q, high_q in PRICE_BANDS:
            low = quantiles[low_q]
            high = quantiles[high_q]
            mask = actual >= low
            mask &= actual <= high if high_q == 1.0 else actual < high
            rows.append(
                {
                    "model": model,
                    "price_band": band,
                    "lower_quantile": low_q,
                    "upper_quantile": high_q,
                    "lower_price_RM": low,
                    "upper_price_RM": high,
                    **regression_summary(actual[mask], predicted[mask]),
                }
            )
    return pd.DataFrame(rows)


def classifier_metrics(labels, probabilities, threshold: float) -> dict:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = probabilities >= float(threshold)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "ROC_AUC": float(roc_auc_score(labels, probabilities)),
        "PR_AUC": float(average_precision_score(labels, probabilities)),
        "Precision": float(precision_score(labels, predicted, zero_division=0)),
        "Recall": float(recall_score(labels, predicted, zero_division=0)),
        "F1": float(f1_score(labels, predicted, zero_division=0)),
        "Specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "Balanced_Accuracy": float(balanced_accuracy_score(labels, predicted)),
        "Brier": float(brier_score_loss(labels, probabilities)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "positive_count": int(labels.sum()),
        "row_count": int(len(labels)),
    }


def routing_error_impact(
    actual,
    predicted,
    true_premium,
    routed_premium,
) -> pd.DataFrame:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    truth = np.asarray(true_premium, dtype=bool)
    routed = np.asarray(routed_premium, dtype=bool)
    groups = {
        "TP": truth & routed,
        "FN": truth & ~routed,
        "FP": ~truth & routed,
        "TN": ~truth & ~routed,
    }
    rows = []
    for name, mask in groups.items():
        row = {"routing_group": name}
        if mask.any():
            row.update(regression_summary(actual[mask], predicted[mask]))
        else:
            row.update({"count": 0, "RMSE_RM": np.nan, "MAE_RM": np.nan, "Mean_Error_RM": np.nan, "Median_Error_RM": np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap(
    actual,
    candidate,
    reference,
    premium_mask,
    draws: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Paired row bootstrap; negative deltas favor the candidate."""
    actual = np.asarray(actual, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    premium_mask = np.asarray(premium_mask, dtype=bool)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(actual))
    premium_indices = indices[premium_mask]
    samples = {"RMSE_RM": [], "MAE_RM": [], "Top5_RMSE_RM": []}
    for _ in range(draws):
        sampled = rng.choice(indices, size=len(indices), replace=True)
        sampled_premium = rng.choice(premium_indices, size=len(premium_indices), replace=True)
        samples["RMSE_RM"].append(rmse(actual[sampled], candidate[sampled]) - rmse(actual[sampled], reference[sampled]))
        samples["MAE_RM"].append(float(mean_absolute_error(actual[sampled], candidate[sampled]) - mean_absolute_error(actual[sampled], reference[sampled])))
        samples["Top5_RMSE_RM"].append(rmse(actual[sampled_premium], candidate[sampled_premium]) - rmse(actual[sampled_premium], reference[sampled_premium]))
    rows = []
    for metric, values in samples.items():
        values = np.asarray(values)
        rows.append(
            {
                "metric": metric,
                "candidate_minus_reference": float(np.mean(values)),
                "CI95_lower": float(np.quantile(values, 0.025)),
                "CI95_upper": float(np.quantile(values, 0.975)),
                "probability_candidate_better": float(np.mean(values < 0)),
                "bootstrap_draws": draws,
            }
        )
    return pd.DataFrame(rows)


def full_metric_bundle(actual, predicted, predictors: int, global_premium_threshold: float) -> dict:
    return metric_bundle(actual, predicted, predictors, premium_threshold=global_premium_threshold)
