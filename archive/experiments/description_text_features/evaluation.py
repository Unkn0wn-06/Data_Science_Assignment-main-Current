"""Original-RM evaluation and uncertainty utilities for description models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from experiments.advanced_real_estate_models.evaluation import metric_bundle


PRICE_BANDS = (
    ("P00_P50", 0.00, 0.50),
    ("P50_P80", 0.50, 0.80),
    ("P80_P90", 0.80, 0.90),
    ("P90_P95", 0.90, 0.95),
    ("P95_P99", 0.95, 0.99),
    ("P99_P100", 0.99, 1.00),
)


def rmse(actual, predicted):
    return float(mean_squared_error(actual, predicted) ** 0.5)


def regression_summary(actual, predicted):
    actual = np.asarray(actual, dtype=float); predicted = np.asarray(predicted, dtype=float)
    if len(actual) == 0:
        return {
            "count": 0,
            "RMSE_RM": np.nan,
            "MAE_RM": np.nan,
            "R2": None,
            "Mean_Error_RM": np.nan,
            "Median_Error_RM": np.nan,
        }
    error = predicted - actual
    return {
        "count": int(len(actual)),
        "RMSE_RM": rmse(actual, predicted),
        "MAE_RM": float(mean_absolute_error(actual, predicted)),
        "R2": float(r2_score(actual, predicted)) if len(actual) > 1 else None,
        "Mean_Error_RM": float(np.mean(error)),
        "Median_Error_RM": float(np.median(error)),
    }


def price_band_masks(actual):
    actual = np.asarray(actual, dtype=float)
    boundaries = {q: float(np.quantile(actual, q)) for _, low, high in PRICE_BANDS for q in (low, high)}
    result = {}
    for name, low_q, high_q in PRICE_BANDS:
        mask = actual >= boundaries[low_q]
        mask &= actual <= boundaries[high_q] if high_q == 1.0 else actual < boundaries[high_q]
        result[name] = mask
    return result, boundaries


def price_band_table(actual, predictions):
    masks, boundaries = price_band_masks(actual)
    rows = []
    for variant, predicted in predictions.items():
        for name, low_q, high_q in PRICE_BANDS:
            mask = masks[name]
            rows.append({"variant": variant, "price_band": name, "lower_quantile": low_q, "upper_quantile": high_q, "lower_price_RM": boundaries[low_q], "upper_price_RM": boundaries[high_q], **regression_summary(np.asarray(actual)[mask], np.asarray(predicted)[mask])})
    return pd.DataFrame(rows)


def underprediction_summary(actual, predicted, mask):
    actual = np.asarray(actual, dtype=float); predicted = np.asarray(predicted, dtype=float); mask = np.asarray(mask, dtype=bool)
    shortfall = actual[mask] - predicted[mask]
    under = shortfall > 0
    return {
        "count": int(mask.sum()),
        "underpredicted_percent": float(under.mean() * 100.0),
        "mean_underprediction_RM": float(shortfall[under].mean()) if under.any() else 0.0,
        "median_underprediction_RM": float(np.median(shortfall[under])) if under.any() else 0.0,
    }


def complete_metrics(actual, predicted, predictors, premium_threshold=905_000.0):
    metrics = metric_bundle(actual, predicted, predictors, premium_threshold=premium_threshold)
    masks, _ = price_band_masks(actual)
    metrics["P95_P99"] = regression_summary(np.asarray(actual)[masks["P95_P99"]], np.asarray(predicted)[masks["P95_P99"]])
    metrics["P99_P100"] = regression_summary(np.asarray(actual)[masks["P99_P100"]], np.asarray(predicted)[masks["P99_P100"]])
    metrics["premium_underprediction"] = underprediction_summary(actual, predicted, np.asarray(actual) >= premium_threshold)
    metrics["P99_underprediction"] = underprediction_summary(actual, predicted, masks["P99_P100"])
    return metrics


def paired_bootstrap(actual, candidate, reference, premium_mask, draws=5000, seed=42):
    actual = np.asarray(actual, float); candidate = np.asarray(candidate, float); reference = np.asarray(reference, float)
    indices = np.arange(len(actual)); premium_indices = indices[np.asarray(premium_mask, bool)]
    rng = np.random.default_rng(seed)
    samples = {"RMSE_RM": [], "MAE_RM": [], "Top5_RMSE_RM": []}
    for _ in range(draws):
        selected = rng.choice(indices, len(indices), replace=True)
        selected_premium = rng.choice(premium_indices, len(premium_indices), replace=True)
        samples["RMSE_RM"].append(rmse(actual[selected], candidate[selected]) - rmse(actual[selected], reference[selected]))
        samples["MAE_RM"].append(float(mean_absolute_error(actual[selected], candidate[selected]) - mean_absolute_error(actual[selected], reference[selected])))
        samples["Top5_RMSE_RM"].append(rmse(actual[selected_premium], candidate[selected_premium]) - rmse(actual[selected_premium], reference[selected_premium]))
    rows = []
    for metric, values in samples.items():
        values = np.asarray(values)
        rows.append({"metric": metric, "candidate_minus_reference": float(np.mean(values)), "CI95_lower": float(np.quantile(values, .025)), "CI95_upper": float(np.quantile(values, .975)), "probability_candidate_better": float(np.mean(values < 0)), "bootstrap_draws": draws})
    return pd.DataFrame(rows)
