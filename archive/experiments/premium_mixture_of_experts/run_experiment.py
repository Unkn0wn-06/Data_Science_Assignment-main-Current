"""Run the leakage-safe premium mixture-of-experts experiment."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.premium_mixture_of_experts.classifier import (
    PremiumClassifier,
    description_feature_table,
    fold_premium_threshold,
    inner_oof_probabilities,
    select_routing_threshold,
)
from experiments.premium_mixture_of_experts.evaluation import (
    classifier_metrics,
    full_metric_bundle,
    paired_bootstrap,
    price_band_metrics,
    regression_summary,
    routing_error_impact,
)
from experiments.premium_mixture_of_experts.regressors import (
    PREMIUM_SCOPES,
    premium_expert,
    premium_scope_mask,
    standard_lightgbm,
    standard_rf,
)
from experiments.premium_mixture_of_experts.routing import hard_route, soft_route


EXPERIMENT = ROOT / "experiments" / "premium_mixture_of_experts"
FIGURES = EXPERIMENT / "figures"
DATA_PATH = ROOT / "data" / "processed" / "enhanced_city_dataset.csv"
RAW_PATH = ROOT / "data" / "raw" / "houses.csv"
GLOBAL_PREMIUM_RM = 905_000.0
REFERENCE_COLUMNS = {
    "random_forest_reference": "prediction__random_forest",
    "lightgbm_interaction_reference": "prediction__feature_minus_micro_market",
}


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(type(value).__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protected_snapshot() -> dict[str, str]:
    files = [
        RAW_PATH,
        DATA_PATH,
        ROOT / "results" / "enhanced_city" / "model_comparison.json",
        ROOT / "results" / "best_model" / "best_model_summary.json",
        ROOT / "prototype" / "app.py",
        ROOT / "app.py",
    ]
    for directory in (
        ROOT / "experiments" / "advanced_real_estate_models",
        ROOT / "experiments" / "noncoordinate_target_encoding",
    ):
        files.extend(path for path in directory.rglob("*") if path.is_file())
    return {str(path.relative_to(ROOT)): _sha256(path) for path in sorted(set(files))}


def _load_references(frame: pd.DataFrame):
    advanced = pd.read_csv(ROOT / "experiments" / "advanced_real_estate_models" / "oof_predictions.csv")
    advanced = advanced.set_index("listing_id").loc[frame["listing_id"].astype(int)].reset_index()
    if not np.array_equal(advanced["actual_price_RM"].to_numpy(float), frame["price"].to_numpy(float)):
        raise AssertionError("Advanced-model reference OOF rows do not align.")
    predictions = {name: advanced[column].to_numpy(float) for name, column in REFERENCE_COLUMNS.items()}
    noncoordinate = pd.read_csv(ROOT / "experiments" / "noncoordinate_target_encoding" / "oof_predictions.csv")
    building = noncoordinate.loc[noncoordinate["variant"] == "building_name_te"].copy()
    building = building.set_index("row_id").loc[frame["listing_id"].astype(int)].reset_index()
    if not np.array_equal(building["actual_price_RM"].to_numpy(float), frame["price"].to_numpy(float)):
        raise AssertionError("Building-name TE reference OOF rows do not align.")
    predictions["building_name_te_reference"] = building["predicted_price_RM"].to_numpy(float)
    return predictions, advanced["fold"].to_numpy(int)


def _fold_regression(actual, predicted, premium_mask):
    result = regression_summary(actual, predicted)
    result["R2"] = float(r2_score(actual, predicted))
    if np.any(premium_mask):
        premium = regression_summary(np.asarray(actual)[premium_mask], np.asarray(predicted)[premium_mask])
        result["Top5_RMSE_RM"] = premium["RMSE_RM"]
        result["Top5_MAE_RM"] = premium["MAE_RM"]
    else:
        result["Top5_RMSE_RM"] = np.nan
        result["Top5_MAE_RM"] = np.nan
    return result


def _system_specs():
    specs = {}
    for scope in PREMIUM_SCOPES:
        code = int(scope * 100)
        specs[f"hard_rf_lgbm_p{code}"] = {
            "routing": "hard", "standard": "rf_all", "premium": f"lgbm_ppsf_p{code}", "scope": scope,
        }
    for scope in (0.10, 0.15, 0.20):
        code = int(scope * 100)
        specs[f"soft_rf_lgbm_p{code}"] = {
            "routing": "soft", "standard": "rf_all", "premium": f"lgbm_ppsf_p{code}", "scope": scope,
        }
    specs.update(
        {
            "soft_lgbm_lgbm_p15": {"routing": "soft", "standard": "lgbm_all", "premium": "lgbm_ppsf_p15", "scope": 0.15},
            "soft_rf_lgbm_direct_p15": {"routing": "soft", "standard": "rf_all", "premium": "lgbm_direct_p15", "scope": 0.15},
            "soft_rf_rf_p15": {"routing": "soft", "standard": "rf_all", "premium": "rf_ppsf_p15", "scope": 0.15},
            "soft_rf_ridge_p15": {"routing": "soft", "standard": "rf_all", "premium": "ridge_ppsf_p15", "scope": 0.15},
            "soft_rf_below_lgbm_p15": {"routing": "soft", "standard": "rf_below", "premium": "lgbm_ppsf_p15", "scope": 0.15},
            "soft_rf_lgbm_p15_calibrated": {"routing": "soft_calibrated", "standard": "rf_all", "premium": "lgbm_ppsf_p15", "scope": 0.15},
            "soft_rf_lgbm_p15_description": {"routing": "soft_description", "standard": "rf_description", "premium": "lgbm_ppsf_p15_description", "scope": 0.15},
        }
    )
    return specs


def _fit_expert(estimator, X, y, train_index, validation_index, selection=None):
    fit_index = train_index if selection is None else train_index[selection]
    fitted = clone(estimator).fit(X.iloc[fit_index], y[fit_index])
    return {
        "validation": fitted.predict(X.iloc[validation_index]),
        "training": fitted.predict(X.iloc[train_index]),
        "fit_training": fitted.predict(X.iloc[fit_index]),
        "fit_index": fit_index,
    }


def _overall_classifier(labels, probabilities, decisions):
    labels = np.asarray(labels, dtype=bool)
    decisions = np.asarray(decisions, dtype=bool)
    probabilities = np.asarray(probabilities, dtype=float)
    tp = int(np.sum(labels & decisions)); fn = int(np.sum(labels & ~decisions))
    fp = int(np.sum(~labels & decisions)); tn = int(np.sum(~labels & ~decisions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "ROC_AUC": float(roc_auc_score(labels, probabilities)),
        "PR_AUC": float(average_precision_score(labels, probabilities)),
        "Precision": precision,
        "Premium_Recall": recall,
        "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "Specificity": tn / (tn + fp) if tn + fp else 0.0,
        "Balanced_Accuracy": 0.5 * (recall + tn / (tn + fp)),
        "Brier": float(np.mean(np.square(probabilities - labels))),
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
    }


def _plot_bar(comparison, metric, filename, title):
    data = comparison.sort_values(metric).head(12)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(data["variant"], data[metric], color=["#16697a" if "reference" not in v else "#8d99ae" for v in data["variant"]])
    ax.invert_yaxis(); ax.set_xlabel("RM"); ax.set_title(title); ax.grid(axis="x", alpha=0.25)
    fig.tight_layout(); fig.savefig(FIGURES / filename, dpi=160); plt.close(fig)


def _make_figures(comparison, bands, classification_oof, routing_best, fold_metrics, best):
    _plot_bar(comparison, "RMSE_RM", "01_model_rmse_comparison.png", "OOF RMSE comparison")
    _plot_bar(comparison, "MAE_RM", "02_model_mae_comparison.png", "OOF MAE comparison")
    _plot_bar(comparison, "Top5_RMSE_RM", "03_top5_rmse_comparison.png", "Top-5% RMSE comparison")
    selected = ["random_forest_reference", "lightgbm_interaction_reference", "building_name_te_reference", best]
    for metric, filename, title in (
        ("RMSE_RM", "04_price_band_rmse.png", "RMSE by actual-price band"),
        ("MAE_RM", "05_price_band_mae.png", "MAE by actual-price band"),
    ):
        pivot = bands[bands["model"].isin(selected)].pivot(index="price_band", columns="model", values=metric)
        ax = pivot.plot.bar(figsize=(11, 6)); ax.set_ylabel("RM"); ax.set_title(title); ax.grid(axis="y", alpha=0.25)
        plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(FIGURES / filename, dpi=160); plt.close()
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = classification_oof["true_premium"].to_numpy(bool)
    for family, column in (("LightGBM", "lightgbm_probability"), ("Random Forest", "random_forest_probability"), ("Calibrated LightGBM", "calibrated_probability")):
        precision, recall, _ = precision_recall_curve(labels, classification_oof[column])
        ax.plot(recall, precision, label=f"{family} (AP={average_precision_score(labels, classification_oof[column]):.3f})")
    ax.set(xlabel="Premium recall", ylabel="Precision", title="Premium classifier precision-recall curve"); ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(FIGURES / "06_premium_classifier_pr_curve.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(classification_oof.loc[~labels, "lightgbm_probability"], bins=30, alpha=0.65, density=True, label="Standard")
    ax.hist(classification_oof.loc[labels, "lightgbm_probability"], bins=30, alpha=0.65, density=True, label="True premium")
    ax.set(xlabel="Predicted premium probability", ylabel="Density", title="OOF premium probability distribution"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES / "07_premium_probability_distribution.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 7)); ax.scatter(routing_best["actual_price_RM"], routing_best["final_prediction_RM"], s=10, alpha=.45)
    limits = [min(routing_best["actual_price_RM"].min(), routing_best["final_prediction_RM"].min()), max(routing_best["actual_price_RM"].max(), routing_best["final_prediction_RM"].max())]
    ax.plot(limits, limits, "--", color="black"); ax.set(xlabel="Actual price (RM)", ylabel="OOF prediction (RM)", title=f"Actual vs predicted: {best}")
    fig.tight_layout(); fig.savefig(FIGURES / "08_actual_vs_predicted_best.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6)); ax.scatter(routing_best["actual_price_RM"], routing_best["signed_error_RM"], s=10, alpha=.45)
    ax.axhline(0, color="black", linestyle="--"); ax.set(xlabel="Actual price (RM)", ylabel="Prediction - actual (RM)", title=f"Residuals: {best}")
    fig.tight_layout(); fig.savefig(FIGURES / "09_residuals_vs_actual_best.png", dpi=160); plt.close(fig)
    impact = routing_error_impact(routing_best["actual_price_RM"], routing_best["final_prediction_RM"], routing_best["true_premium"], routing_best["predicted_premium"])
    fig, ax = plt.subplots(figsize=(7, 5)); ax.bar(impact["routing_group"], impact["RMSE_RM"], color="#16697a"); ax.set(ylabel="RMSE (RM)", title="Downstream error by routing outcome"); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(FIGURES / "10_routing_error_analysis.png", dpi=160); plt.close(fig)
    stability = fold_metrics[fold_metrics["variant"].isin(selected)]
    fig, ax = plt.subplots(figsize=(9, 6))
    for variant, group in stability.groupby("variant"):
        ax.plot(group["fold"], group["RMSE_RM"], marker="o", label=variant)
    ax.set(xlabel="Outer fold", ylabel="RMSE (RM)", title="Fold stability"); ax.set_xticks(range(1, 6)); ax.legend(fontsize=8); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(FIGURES / "11_fold_stability.png", dpi=160); plt.close(fig)


def main():
    started = time.perf_counter(); FIGURES.mkdir(parents=True, exist_ok=True)
    before = _protected_snapshot()
    frame = pd.read_csv(DATA_PATH)
    if len(frame) != 3791:
        raise AssertionError(f"Expected 3,791 canonical rows, found {len(frame)}.")
    y = frame["price"].to_numpy(float)
    X = frame.drop(columns=["price"]).copy()
    descriptions, frequencies, eligible = description_feature_table(RAW_PATH, frame["listing_id"], minimum_count=10)
    for column in eligible:
        X[column] = descriptions[column].to_numpy(int)
    extra = tuple(eligible)
    reference_predictions, reference_fold = _load_references(frame)
    splits = list(KFold(n_splits=5, shuffle=True, random_state=42).split(X))
    generated_fold = np.empty(len(frame), dtype=int)
    for fold, (_, validation) in enumerate(splits, 1): generated_fold[validation] = fold
    if not np.array_equal(reference_fold, generated_fold):
        raise AssertionError("Outer folds differ from the verified references.")

    specs = _system_specs()
    oof = {name: np.empty(len(y), dtype=float) for name in specs}
    train_summaries = {name: [] for name in specs}
    fold_rows, routing_rows, classification_rows, classifier_oof_rows = [], [], [], []
    premium_thresholds, scope_records, expert_gap_rows = [], [], []

    for fold, (train_index, validation_index) in enumerate(splits, 1):
        y_train, y_validation = y[train_index], y[validation_index]
        threshold = fold_premium_threshold(y_train)
        train_labels = y_train >= threshold; validation_labels = y_validation >= threshold
        premium_thresholds.append({"fold": fold, "threshold_RM": threshold, "training_premium_count": int(train_labels.sum()), "validation_premium_count": int(validation_labels.sum())})

        probabilities = {}; train_probabilities = {}; decisions = {}
        for family in ("lightgbm", "random_forest"):
            classifier = PremiumClassifier(family)
            inner_probability = inner_oof_probabilities(X.iloc[train_index], train_labels, classifier)
            selected_threshold, threshold_scores = select_routing_threshold(train_labels, inner_probability)
            fitted = clone(classifier).fit(X.iloc[train_index], train_labels)
            probabilities[family] = fitted.predict_proba(X.iloc[validation_index])[:, 1]
            train_probabilities[family] = fitted.predict_proba(X.iloc[train_index])[:, 1]
            decisions[family] = probabilities[family] >= selected_threshold
            classification_rows.append({"classifier": family, "fold": fold, "selected_inner_threshold": selected_threshold, "inner_F2_scores": json.dumps(threshold_scores), **classifier_metrics(validation_labels, probabilities[family], selected_threshold)})
        calibrated = CalibratedClassifierCV(PremiumClassifier("lightgbm"), method="sigmoid", cv=3)
        calibrated.fit(X.iloc[train_index], train_labels)
        probabilities["calibrated"] = calibrated.predict_proba(X.iloc[validation_index])[:, 1]
        train_probabilities["calibrated"] = calibrated.predict_proba(X.iloc[train_index])[:, 1]
        lgb_threshold = classification_rows[-2]["selected_inner_threshold"]
        decisions["calibrated"] = probabilities["calibrated"] >= lgb_threshold
        classification_rows.append({"classifier": "calibrated_lightgbm", "fold": fold, "selected_inner_threshold": lgb_threshold, "inner_F2_scores": "calibration only; inherited training-selected LightGBM threshold", **classifier_metrics(validation_labels, probabilities["calibrated"], lgb_threshold)})

        description_classifier = PremiumClassifier("lightgbm", extra_numerical=extra)
        description_inner = inner_oof_probabilities(X.iloc[train_index], train_labels, description_classifier)
        description_threshold, description_scores = select_routing_threshold(train_labels, description_inner)
        description_fitted = clone(description_classifier).fit(X.iloc[train_index], train_labels)
        probabilities["description"] = description_fitted.predict_proba(X.iloc[validation_index])[:, 1]
        train_probabilities["description"] = description_fitted.predict_proba(X.iloc[train_index])[:, 1]
        decisions["description"] = probabilities["description"] >= description_threshold
        classification_rows.append({"classifier": "lightgbm_description", "fold": fold, "selected_inner_threshold": description_threshold, "inner_F2_scores": json.dumps(description_scores), **classifier_metrics(validation_labels, probabilities["description"], description_threshold)})

        classifier_oof_rows.extend(
            {"row_index": int(row), "listing_id": int(frame.iloc[row]["listing_id"]), "fold": fold, "true_premium": bool(validation_labels[pos]), "fold_premium_threshold_RM": threshold, "lightgbm_probability": probabilities["lightgbm"][pos], "random_forest_probability": probabilities["random_forest"][pos], "calibrated_probability": probabilities["calibrated"][pos], "description_probability": probabilities["description"][pos], "lightgbm_predicted_premium": bool(decisions["lightgbm"][pos]), "random_forest_predicted_premium": bool(decisions["random_forest"][pos]), "calibrated_predicted_premium": bool(decisions["calibrated"][pos])}
            for pos, row in enumerate(validation_index)
        )

        standard = {
            "rf_all": _fit_expert(standard_rf(), X, y, train_index, validation_index),
            "lgbm_all": _fit_expert(standard_lightgbm(), X, y, train_index, validation_index),
            "rf_below": _fit_expert(standard_rf(), X, y, train_index, validation_index, ~train_labels),
            "rf_description": _fit_expert(standard_rf(extra), X, y, train_index, validation_index),
        }
        premium = {}
        for scope in PREMIUM_SCOPES:
            code = int(scope * 100); mask, scope_threshold = premium_scope_mask(y_train, scope)
            scope_records.append({"fold": fold, "scope": f"P{code}", "scope_fraction": scope, "threshold_RM": scope_threshold, "training_rows": int(mask.sum())})
            premium[f"lgbm_ppsf_p{code}"] = _fit_expert(premium_expert("lightgbm"), X, y, train_index, validation_index, mask)
        scope15, _ = premium_scope_mask(y_train, .15)
        premium["lgbm_direct_p15"] = _fit_expert(premium_expert("lightgbm", "direct_price"), X, y, train_index, validation_index, scope15)
        premium["rf_ppsf_p15"] = _fit_expert(premium_expert("random_forest"), X, y, train_index, validation_index, scope15)
        premium["ridge_ppsf_p15"] = _fit_expert(premium_expert("ridge"), X, y, train_index, validation_index, scope15)
        premium["lgbm_ppsf_p15_description"] = _fit_expert(premium_expert("lightgbm", "ppsf", extra), X, y, train_index, validation_index, scope15)

        for expert_name, outputs in premium.items():
            validation_premium = validation_labels
            train_fit = outputs["fit_index"]
            fit_metrics = _fold_regression(y[train_fit], outputs["fit_training"], np.ones(len(train_fit), bool))
            validation_metrics = _fold_regression(y_validation[validation_premium], outputs["validation"][validation_premium], np.ones(validation_premium.sum(), bool))
            expert_gap_rows.append({"expert": expert_name, "fold": fold, "training_rows": len(train_fit), "validation_premium_rows": int(validation_premium.sum()), "Training_RMSE_RM": fit_metrics["RMSE_RM"], "OOF_Premium_RMSE_RM": validation_metrics["RMSE_RM"], "Training_MAE_RM": fit_metrics["MAE_RM"], "OOF_Premium_MAE_RM": validation_metrics["MAE_RM"], "Training_R2": fit_metrics["R2"], "OOF_Premium_R2": validation_metrics["R2"]})

        for name, spec in specs.items():
            standard_output = standard[spec["standard"]]; premium_output = premium[spec["premium"]]
            if spec["routing"] == "hard":
                selected_threshold = next(row["selected_inner_threshold"] for row in classification_rows if row["fold"] == fold and row["classifier"] == "lightgbm")
                validation_prediction, routed = hard_route(standard_output["validation"], premium_output["validation"], probabilities["lightgbm"], selected_threshold)
                training_prediction, _ = hard_route(standard_output["training"], premium_output["training"], train_probabilities["lightgbm"], selected_threshold)
                route_probability = probabilities["lightgbm"]
            else:
                probability_key = "calibrated" if spec["routing"] == "soft_calibrated" else "description" if spec["routing"] == "soft_description" else "lightgbm"
                validation_prediction = soft_route(standard_output["validation"], premium_output["validation"], probabilities[probability_key])
                training_prediction = soft_route(standard_output["training"], premium_output["training"], train_probabilities[probability_key])
                routed = decisions[probability_key]
                route_probability = probabilities[probability_key]
            oof[name][validation_index] = validation_prediction
            train_summaries[name].append(_fold_regression(y_train, training_prediction, train_labels))
            fold_metric = _fold_regression(y_validation, validation_prediction, validation_labels)
            classification_metric = classifier_metrics(validation_labels, route_probability, next(row["selected_inner_threshold"] for row in classification_rows if row["fold"] == fold and row["classifier"] == ("lightgbm_description" if spec["routing"] == "soft_description" else "lightgbm")))
            fold_rows.append({"variant": name, "fold": fold, "training_rows": len(train_index), "validation_rows": len(validation_index), **fold_metric, "Premium_Recall": classification_metric["Recall"], "Premium_Precision": classification_metric["Precision"]})
            for position, row in enumerate(validation_index):
                error = validation_prediction[position] - y[row]
                routing_rows.append({"row_index": int(row), "listing_id": int(frame.iloc[row]["listing_id"]), "variant": name, "fold": fold, "actual_price_RM": y[row], "standard_prediction_RM": standard_output["validation"][position], "premium_prediction_RM": premium_output["validation"][position], "premium_probability": route_probability[position], "final_prediction_RM": validation_prediction[position], "true_premium": bool(validation_labels[position]), "predicted_premium": bool(routed[position]), "absolute_error_RM": abs(error), "signed_error_RM": error, "fold_premium_threshold_RM": threshold, "premium_scope": spec["scope"], "routing": spec["routing"], "standard_expert": spec["standard"], "premium_expert": spec["premium"]})
        print(f"Completed outer fold {fold}/5 (premium threshold RM{threshold:,.0f}).", flush=True)

    # Add verified global references to common result tables.
    global_premium = y >= GLOBAL_PREMIUM_RM
    for name, prediction in reference_predictions.items():
        for fold in range(1, 6):
            mask = generated_fold == fold
            fold_rows.append({"variant": name, "fold": fold, "training_rows": int((~mask).sum()), "validation_rows": int(mask.sum()), **_fold_regression(y[mask], prediction[mask], global_premium[mask]), "Premium_Recall": np.nan, "Premium_Precision": np.nan})

    all_predictions = {**reference_predictions, **oof}
    bundles = {name: full_metric_bundle(y, prediction, X.shape[1], GLOBAL_PREMIUM_RM) for name, prediction in all_predictions.items()}
    comparison_rows = []
    for name, metrics in bundles.items():
        comparison_rows.append({"variant": name, "architecture": "reference" if name in reference_predictions else specs[name]["routing"], "RMSE_RM": metrics["RMSE_RM"], "MAE_RM": metrics["MAE_RM"], "R2": metrics["R2"], "Adjusted_R2": metrics["Adjusted_R2"], "Median_AE_RM": metrics["Median_AE_RM"], "Mean_Error_RM": metrics["Mean_Error_RM"], "Median_Error_RM": metrics["Median_Error_RM"], "Top5_RMSE_RM": metrics["top_5_percent"]["RMSE_RM"], "Top5_MAE_RM": metrics["top_5_percent"]["MAE_RM"], "Remaining95_RMSE_RM": metrics["remaining_95_percent"]["RMSE_RM"], "Remaining95_MAE_RM": metrics["remaining_95_percent"]["MAE_RM"]})
    comparison = pd.DataFrame(comparison_rows).sort_values("RMSE_RM")
    tiered_names = list(specs)
    best_rmse = min(tiered_names, key=lambda name: bundles[name]["RMSE_RM"])
    best_mae = min(tiered_names, key=lambda name: bundles[name]["MAE_RM"])
    best_premium = min(tiered_names, key=lambda name: bundles[name]["top_5_percent"]["RMSE_RM"])
    best_balanced = min(tiered_names, key=lambda name: (bundles[name]["RMSE_RM"] / 120201.22 + bundles[name]["MAE_RM"] / 61217.27 + bundles[name]["top_5_percent"]["RMSE_RM"] / 418749.39))

    routing_frame = pd.DataFrame(routing_rows).sort_values(["variant", "row_index"])
    classification_frame = pd.DataFrame(classification_rows)
    classification_oof = pd.DataFrame(classifier_oof_rows).sort_values("row_index")
    fold_frame = pd.DataFrame(fold_rows).sort_values(["variant", "fold"])
    oof_rows = []
    for name, prediction in all_predictions.items():
        oof_rows.append(pd.DataFrame({"row_index": np.arange(len(y)), "listing_id": frame["listing_id"].astype(int), "fold": generated_fold, "actual_price_RM": y, "predicted_price_RM": prediction, "residual_RM": prediction - y, "absolute_error_RM": np.abs(prediction - y), "premium_global_flag": global_premium, "variant": name}))
    oof_frame = pd.concat(oof_rows, ignore_index=True)
    bands = price_band_metrics(y, {name: all_predictions[name] for name in [*reference_predictions, *tiered_names]})

    bootstrap_rows = []
    for reference in reference_predictions:
        table = paired_bootstrap(y, oof[best_balanced], reference_predictions[reference], global_premium, draws=5000)
        table.insert(0, "candidate", best_balanced); table.insert(1, "reference", reference); bootstrap_rows.append(table)
    bootstrap = pd.concat(bootstrap_rows, ignore_index=True)
    scope_rows = []
    scope_mask = y <= 850_000
    for name in [*reference_predictions, best_balanced]:
        metric = _fold_regression(y[scope_mask], all_predictions[name][scope_mask], np.zeros(scope_mask.sum(), bool))
        scope_rows.append({"label": "SCOPE-RESTRICTED DIAGNOSTIC - NOT MODEL IMPROVEMENT", "variant": name, "price_max_RM": 850000, **metric})

    classifier_summary = {}
    for family, probability_column, decision_column in (("lightgbm", "lightgbm_probability", "lightgbm_predicted_premium"), ("random_forest", "random_forest_probability", "random_forest_predicted_premium")):
        classifier_summary[family] = _overall_classifier(classification_oof["true_premium"], classification_oof[probability_column], classification_oof[decision_column])
    classifier_summary["calibrated_lightgbm"] = _overall_classifier(classification_oof["true_premium"], classification_oof["calibrated_probability"], classification_oof["calibrated_predicted_premium"])
    best_routing = routing_frame[routing_frame["variant"] == best_balanced].sort_values("row_index")
    routing_impact = routing_error_impact(best_routing["actual_price_RM"], best_routing["final_prediction_RM"], best_routing["true_premium"], best_routing["predicted_premium"])
    wins = {}
    for reference in reference_predictions:
        merged = fold_frame[fold_frame["variant"].isin([best_balanced, reference])].pivot(index="fold", columns="variant", values="RMSE_RM")
        wins[reference] = int((merged[best_balanced] < merged[reference]).sum())

    generalization = {}
    for name in tiered_names:
        train_rmse = float(np.mean([row["RMSE_RM"] for row in train_summaries[name]])); train_mae = float(np.mean([row["MAE_RM"] for row in train_summaries[name]])); train_r2 = float(np.mean([row["R2"] for row in train_summaries[name]]))
        generalization[name] = {"Training_RMSE_RM": train_rmse, "OOF_RMSE_RM": bundles[name]["RMSE_RM"], "RMSE_gap_RM": bundles[name]["RMSE_RM"] - train_rmse, "Training_MAE_RM": train_mae, "OOF_MAE_RM": bundles[name]["MAE_RM"], "MAE_gap_RM": bundles[name]["MAE_RM"] - train_mae, "Training_R2": train_r2, "OOF_R2": bundles[name]["R2"], "R2_gap": train_r2 - bundles[name]["R2"]}

    reference_generalization = {}
    with (ROOT / "experiments" / "advanced_real_estate_models" / "results.json").open(encoding="utf-8") as handle: advanced_results = json.load(handle)
    with (ROOT / "experiments" / "noncoordinate_target_encoding" / "results.json").open(encoding="utf-8") as handle: te_results = json.load(handle)
    reference_generalization["random_forest_reference"] = advanced_results["baseline"]["generalization_gap"]
    reference_generalization["lightgbm_interaction_reference"] = advanced_results["generalization_gap"]["feature_minus_micro_market"]
    reference_generalization["building_name_te_reference"] = te_results["generalization"]["building_name_te"]

    _make_figures(comparison, bands, classification_oof, best_routing, fold_frame, best_balanced)
    feature_summary = {"method": "predefined domain-motivated case-insensitive regex over cleaned description text", "minimum_frequency_for_modeling": 10, "frequencies": frequencies, "modeled_features": eligible, "excluded_rare_features": [name for name in frequencies if name not in eligible], "target_used_for_feature_definition": False, "coordinate_features_used": False}
    results = {
        "dataset": {"path": str(DATA_PATH.relative_to(ROOT)).replace("\\", "/"), "sha256": _sha256(DATA_PATH), "rows": len(frame), "columns": frame.shape[1], "headline_rows_removed": 0, "premium_rows_removed": 0, "target_winsorized_or_capped": False, "global_descriptive_p95_RM": GLOBAL_PREMIUM_RM, "global_descriptive_premium_count": int(global_premium.sum())},
        "references": {name: bundles[name] for name in reference_predictions},
        "premium_definition": {"routing_definition": "outer-training target 95th percentile only", "per_fold": premium_thresholds, "global_RM905k_used_for_routing": False},
        "classifier": {"primary": "LightGBMClassifier", "selection": "inner 5-fold F2 over thresholds 0.20-0.70", "overall": classifier_summary, "calibration": "sigmoid fitted by 3-fold calibration strictly inside each outer training fold", "calibration_tested": True},
        "premium_training_scopes": {"definitions": ["P5", "P10", "P15", "P20"], "per_fold": scope_records, "expert_generalization_by_fold": expert_gap_rows},
        "hard_routing": {name: bundles[name] for name in tiered_names if specs[name]["routing"] == "hard"},
        "soft_routing": {name: bundles[name] for name in tiered_names if specs[name]["routing"] != "hard"},
        "description_features": feature_summary,
        "price_band_performance": bands.to_dict("records"),
        "routing_errors": {"variant": best_balanced, "groups": routing_impact.to_dict("records")},
        "generalization": {**reference_generalization, **generalization},
        "bootstrap": {"candidate": best_balanced, "difference_definition": "candidate - reference; negative is better", "fixed_oof_limitation": "Does not include model-refit or candidate-selection uncertainty.", "comparisons": bootstrap.to_dict("records")},
        "fold_wins": wins,
        "scope_restricted_diagnostic": scope_rows,
        "best_rmse_variant": best_rmse,
        "best_mae_variant": best_mae,
        "best_premium_variant": best_premium,
        "best_balanced_variant": best_balanced,
        "success_criteria": {"RMSE_below_120201": bundles[best_balanced]["RMSE_RM"] < 120201, "MAE_below_61217": bundles[best_balanced]["MAE_RM"] < 61217, "Top5_RMSE_below_418749": bundles[best_balanced]["top_5_percent"]["RMSE_RM"] < 418749, "all_three": bundles[best_balanced]["RMSE_RM"] < 120201 and bundles[best_balanced]["MAE_RM"] < 61217 and bundles[best_balanced]["top_5_percent"]["RMSE_RM"] < 418749},
        "recommendation": "Do not promote automatically. Review balanced OOF performance, fold stability, routing errors, and paired-bootstrap intervals.",
        "leakage_audit": {"outer_validation_price_used_for_premium_threshold": False, "outer_validation_price_used_for_classifier_training": False, "outer_validation_price_used_for_routing_threshold": False, "outer_validation_price_used_for_premium_scope": False, "outer_validation_price_used_for_target_encoding": False, "outer_validation_price_used_for_feature_selection": False, "outer_validation_price_used_for_tuning": False, "routing_computable_at_inference_without_actual_price": True, "all_validation_rows_evaluated_once_per_variant": bool((routing_frame.groupby("variant")["row_index"].nunique() == len(frame)).all()), "coordinate_features_used": False},
        "runtime_seconds": time.perf_counter() - started,
    }

    comparison.to_csv(EXPERIMENT / "model_comparison.csv", index=False)
    fold_frame.to_csv(EXPERIMENT / "fold_metrics.csv", index=False)
    oof_frame.to_csv(EXPERIMENT / "oof_predictions.csv", index=False)
    classification_frame.to_csv(EXPERIMENT / "classification_metrics.csv", index=False)
    routing_frame.to_csv(EXPERIMENT / "routing_analysis.csv", index=False)
    with (EXPERIMENT / "feature_summary.json").open("w", encoding="utf-8") as handle: json.dump(feature_summary, handle, indent=2, default=_json_default)
    after = _protected_snapshot()
    results["leakage_audit"]["protected_files_unchanged"] = before == after
    results["production_safety"] = {"protected_file_count": len(before), "all_sha256_unchanged": before == after, "before_sha256": before, "after_sha256": after}
    results["artifacts"] = [str(path.relative_to(ROOT)).replace("\\", "/") for path in [EXPERIMENT / "results.json", EXPERIMENT / "model_comparison.csv", EXPERIMENT / "fold_metrics.csv", EXPERIMENT / "oof_predictions.csv", EXPERIMENT / "classification_metrics.csv", EXPERIMENT / "routing_analysis.csv", EXPERIMENT / "feature_summary.json", *sorted(FIGURES.glob("*.png"))]]
    with (EXPERIMENT / "results.json").open("w", encoding="utf-8") as handle: json.dump(results, handle, indent=2, default=_json_default)
    print(f"Completed experiment in {results['runtime_seconds']:.1f}s. Best balanced: {best_balanced}.", flush=True)


if __name__ == "__main__":
    main()
