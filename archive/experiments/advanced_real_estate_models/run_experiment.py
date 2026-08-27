"""Run the complete leakage-safe advanced real-estate model experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from experiments.advanced_real_estate_models.evaluation import (
    attach_baseline_comparison,
    evaluate_fixed,
    evaluate_nested_candidates,
    evaluate_nested_ensembles,
    quantile_diagnostics,
    residual_correlations,
)
from experiments.advanced_real_estate_models.feature_engineering import (
    INTERACTION_FEATURES,
    MICRO_FEATURES,
    MICRO_LEVELS,
    MICRO_STATS,
    SIZE_NUMERICAL_FEATURES,
    FeatureEngineeringPPSFRegressor,
    engineered_feature_lists,
)
from experiments.advanced_real_estate_models.model_builders import (
    TargetStrategyRegressor,
    build_base_regressor,
    build_estimator,
    candidate_parameters,
    dependency_status,
    quantile_parameters,
    tweedie_candidates,
)
from src.cleaning.enhanced_city import ENHANCED_CITY_DATA_PATH
from src.models.common.features import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)
from src.models.enhanced_city import (
    build_base_regressor as build_existing_base_regressor,
    build_ppsf_estimator,
    shared_folds,
)


EXPERIMENT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = EXPERIMENT_DIR / "figures"
RESULTS_PATH = EXPERIMENT_DIR / "results.json"
COMPARISON_CSV = EXPERIMENT_DIR / "model_comparison.csv"
OOF_CSV = EXPERIMENT_DIR / "oof_predictions.csv"
FEATURE_SUMMARY_PATH = EXPERIMENT_DIR / "feature_summary.json"
RESIDUAL_CORRELATIONS_CSV = EXPERIMENT_DIR / "residual_correlations.csv"
FOLD_METRICS_CSV = EXPERIMENT_DIR / "fold_metrics.csv"
PREMIUM_METRICS_CSV = EXPERIMENT_DIR / "premium_segment_metrics.csv"
QUANTILE_METRICS_CSV = EXPERIMENT_DIR / "quantile_metrics.csv"
BASELINE_RESULTS_PATH = PROJECT_ROOT / "results/enhanced_city/model_comparison.json"
BASELINE_TOLERANCE = 1e-8

PROTECTED_PATHS = (
    PROJECT_ROOT / "data/raw/houses.csv",
    ENHANCED_CITY_DATA_PATH,
    BASELINE_RESULTS_PATH,
    PROJECT_ROOT / "results/best_model/best_model_summary.json",
    PROJECT_ROOT / "prototype/app.py",
    PROJECT_ROOT / "app.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data() -> tuple[pd.DataFrame, dict]:
    data = pd.read_csv(ENHANCED_CITY_DATA_PATH)
    missing = sorted({TARGET_COLUMN, *MODEL_FEATURES}.difference(data.columns))
    if missing:
        raise ValueError(f"Enhanced City dataset is missing columns: {missing}")
    price = pd.to_numeric(data[TARGET_COLUMN], errors="coerce").to_numpy(float)
    size = pd.to_numeric(data["property_size_sqft"], errors="coerce").to_numpy(float)
    invalid_price = ~np.isfinite(price) | (price <= 0)
    invalid_size = ~np.isfinite(size) | (size <= 0)
    if invalid_price.any() or invalid_size.any():
        raise ValueError(
            "Canonical rows contain invalid target/size values: "
            f"price={invalid_price.sum()}, size={invalid_size.sum()}"
        )
    data[TARGET_COLUMN] = price
    data["property_size_sqft"] = size
    profile = {
        "path": ENHANCED_CITY_DATA_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": sha256(ENHANCED_CITY_DATA_PATH),
        "rows": int(len(data)),
        "columns": int(data.shape[1]),
        "rows_removed": 0,
        "target_min_RM": float(price.min()),
        "target_max_RM": float(price.max()),
        "target_p95_RM": float(np.quantile(price, 0.95)),
        "target_skew": float(pd.Series(price).skew()),
        "target_winsorized_or_capped": False,
        "predictions_clipped_to_premium_limit": False,
    }
    return data.reset_index(drop=True), profile


def assert_baseline(result: dict) -> dict:
    reference = json.loads(BASELINE_RESULTS_PATH.read_text(encoding="utf-8"))
    expected = reference["results"]["Random Forest"]
    observed = result["metrics"]
    differences = {
        "RMSE_RM": observed["RMSE_RM"] - expected["RMSE_RM"],
        "MAE_RM": observed["MAE_RM"] - expected["MAE_RM"],
        "R2": observed["R2"] - expected["R2"],
        "Adjusted_R2": observed["Adjusted_R2"] - expected["Adjusted_R2"],
        "Median_AE_RM": observed["Median_AE_RM"] - expected["Median_AE_RM"],
    }
    if any(abs(value) > BASELINE_TOLERANCE for value in differences.values()):
        raise AssertionError(f"Baseline reproduction failed: {differences}")
    return {
        "matched": True,
        "absolute_tolerance": BASELINE_TOLERANCE,
        "source": BASELINE_RESULTS_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "metric_differences": differences,
    }


def run_nested_family(family: str, params: list[dict], X, y, folds) -> dict:
    estimators = [
        build_estimator(
            family,
            candidate,
            NUMERICAL_FEATURES,
            CATEGORICAL_FEATURES,
            target_strategy="ppsf",
        )
        for candidate in params
    ]
    label = {
        "catboost": "CatBoost",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "huber": "Huber Boosting",
    }[family]
    return evaluate_nested_candidates(
        label,
        estimators,
        params,
        X,
        y,
        folds,
        len(MODEL_FEATURES),
        inner_splits=2 if family == "catboost" else 3,
    )


def build_feature_estimator(
    family: str,
    params: dict,
    *,
    include_micro: bool,
    include_interactions: bool,
    remove_city: bool = False,
    remove_building_developer: bool = False,
):
    numerical, categorical = engineered_feature_lists(
        NUMERICAL_FEATURES,
        CATEGORICAL_FEATURES,
        include_micro=include_micro,
        include_interactions=include_interactions,
        remove_city=remove_city,
        remove_building_developer=remove_building_developer,
    )
    base = build_base_regressor(family, params, numerical, categorical)
    estimator = FeatureEngineeringPPSFRegressor(
        base,
        include_micro=include_micro,
        include_interactions=include_interactions,
    )
    return estimator, numerical, categorical


def comparison_row(key: str, result: dict) -> dict:
    metrics = result["metrics"]
    gap = result["generalization_gap"]
    change = result["comparison_vs_random_forest"]
    return {
        "Model_Key": key,
        "Model": result["name"],
        "RMSE_RM": metrics["RMSE_RM"],
        "MAE_RM": metrics["MAE_RM"],
        "R2": metrics["R2"],
        "Adjusted_R2": metrics["Adjusted_R2"],
        "Median_AE_RM": metrics["Median_AE_RM"],
        "RMSLE": metrics["RMSLE"],
        "MAPE_Percent": metrics["MAPE_Percent"],
        "Median_APE_Percent": metrics["Median_APE_Percent"],
        "Mean_Error_RM": metrics["Mean_Error_RM"],
        "Median_Error_RM": metrics["Median_Error_RM"],
        "Top5_RMSE_RM": metrics["top_5_percent"]["RMSE_RM"],
        "Top5_MAE_RM": metrics["top_5_percent"]["MAE_RM"],
        "Remaining95_RMSE_RM": metrics["remaining_95_percent"]["RMSE_RM"],
        "Remaining95_MAE_RM": metrics["remaining_95_percent"]["MAE_RM"],
        "Training_RMSE_RM": gap["Training_RMSE_RM"],
        "Training_MAE_RM": gap["Training_MAE_RM"],
        "Training_R2": gap["Training_R2"],
        "RMSE_Gap_RM": gap["RMSE_gap_RM"],
        "MAE_Gap_RM": gap["MAE_gap_RM"],
        "R2_Gap": gap["R2_gap"],
        "Fold_RMSE_SD_RM": float(np.std([row["RMSE_RM"] for row in result["folds"]])),
        **change,
    }


def _style_axis(axis, title: str, subtitle: str = "") -> None:
    axis.set_title(title, loc="left", fontsize=13, color="#1f2937", pad=17)
    if subtitle:
        axis.text(0, 1.01, subtitle, transform=axis.transAxes, fontsize=9, color="#6b7280")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    axis.set_axisbelow(True)


def _display_model_name(value: str) -> str:
    return (
        value.replace("Lightgbm", "LightGBM")
        .replace("minus_micro_market", "without micro-market")
        .replace("minus_interactions", "without interactions")
        .replace("_", " ")
    )


def create_figures(
    comparison: pd.DataFrame,
    oof: pd.DataFrame,
    best_key: str,
    residual_corr: pd.DataFrame,
    quantile_predictions: dict[float, np.ndarray],
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    colors = {"blue": "#2563eb", "orange": "#f59e0b", "ink": "#374151"}
    plot_table = comparison.head(10).sort_values("RMSE_RM", ascending=True).copy()
    plot_table["Display_Model"] = plot_table["Model"].map(_display_model_name)

    fig, ax = plt.subplots(figsize=(10, 6))
    positions = np.arange(len(plot_table))
    bars = ax.barh(positions, plot_table["RMSE_RM"], color=colors["blue"])
    ax.set_yticks(positions, plot_table["Display_Model"])
    ax.bar_label(bars, labels=[f"{value / 1000:.1f}k" for value in plot_table["RMSE_RM"]], padding=4, fontsize=8)
    ax.margins(x=0.12)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE (RM)")
    _style_axis(ax, "Model RMSE comparison", "Shared 5-fold out-of-fold predictions; lower is better")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_model_rmse_comparison.png", dpi=160)
    plt.close(fig)

    mae_table = comparison.head(10).sort_values("MAE_RM", ascending=True).copy()
    mae_table["Display_Model"] = mae_table["Model"].map(_display_model_name)
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = np.arange(len(mae_table))
    bars = ax.barh(positions, mae_table["MAE_RM"], color=colors["orange"])
    ax.set_yticks(positions, mae_table["Display_Model"])
    ax.bar_label(bars, labels=[f"{value / 1000:.1f}k" for value in mae_table["MAE_RM"]], padding=4, fontsize=8)
    ax.margins(x=0.12)
    ax.invert_yaxis()
    ax.set_xlabel("MAE (RM)")
    _style_axis(ax, "Model MAE comparison", "Shared 5-fold out-of-fold predictions; lower is better")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "02_model_mae_comparison.png", dpi=160)
    plt.close(fig)

    actual = oof["actual_price_RM"].to_numpy(float)
    predicted = oof[best_key].to_numpy(float)
    positive = (actual > 0) & (predicted > 0)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actual[positive], predicted[positive], s=12, alpha=0.35, color=colors["blue"])
    limits = [min(actual[positive].min(), predicted[positive].min()), max(actual.max(), predicted[positive].max())]
    ax.plot(limits, limits, linestyle="--", color=colors["ink"], linewidth=1.2, label="Ideal")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Actual price (RM, log scale)")
    ax.set_ylabel("Predicted price (RM, log scale)")
    ax.legend(frameon=False)
    _style_axis(ax, "Actual vs predicted price", f"Best OOF model: {_display_model_name(best_key.removeprefix('prediction__'))}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "03_actual_vs_predicted_best_model.png", dpi=160)
    plt.close(fig)

    residual = predicted - actual
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(actual, residual, s=12, alpha=0.35, color=colors["blue"])
    ax.axhline(0, linestyle="--", color=colors["ink"], linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("Actual price (RM, log scale)")
    ax.set_ylabel("Prediction minus actual (RM)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1_000_000:.1f}M"))
    _style_axis(ax, "Residuals vs actual price", "Original uncapped price scale")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "04_residuals_vs_actual_price.png", dpi=160)
    plt.close(fig)

    top_table = comparison.nsmallest(10, "Top5_RMSE_RM").sort_values("Top5_RMSE_RM").copy()
    top_table["Display_Model"] = top_table["Model"].map(_display_model_name)
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = np.arange(len(top_table))
    bars = ax.barh(positions, top_table["Top5_RMSE_RM"], color=colors["orange"])
    ax.set_yticks(positions, top_table["Display_Model"])
    ax.bar_label(bars, labels=[f"{value / 1000:.0f}k" for value in top_table["Top5_RMSE_RM"]], padding=4, fontsize=8)
    ax.margins(x=0.12)
    ax.invert_yaxis()
    ax.set_xlabel("Top-5% RMSE (RM)")
    _style_axis(ax, "Premium-property RMSE comparison", "Premium threshold is the global descriptive 95th percentile")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "05_top5_model_comparison.png", dpi=160)
    plt.close(fig)

    fold_keys = comparison.head(5)["Model_Key"].tolist()
    fold_columns = [f"fold_rmse__{key}" for key in fold_keys]
    fold_colors = ["#2563eb", "#f59e0b", "#6b7280", "#8b5cf6", "#d97706"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for column, color in zip(fold_columns, fold_colors):
        values = oof[column].dropna().to_numpy(float)
        model_key = column.split("__", 1)[1]
        label = comparison.loc[comparison["Model_Key"] == model_key, "Model"].iloc[0]
        ax.plot(np.arange(1, len(values) + 1), values, marker="o", color=color, label=_display_model_name(label))
    ax.set_xlabel("Fold")
    ax.set_ylabel("RMSE (RM)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _style_axis(ax, "Fold RMSE stability", "Same five outer folds for all displayed candidates")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "06_fold_stability.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(residual_corr.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(residual_corr)), residual_corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(residual_corr)), residual_corr.index)
    for row in range(len(residual_corr)):
        for column in range(len(residual_corr)):
            ax.text(column, row, f"{residual_corr.iloc[row, column]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Residual correlation")
    _style_axis(ax, "OOF residual correlation", "Lower off-diagonal values indicate more complementary errors")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "07_residual_correlation_heatmap.png", dpi=160)
    plt.close(fig)

    if quantile_predictions:
        order = np.argsort(actual)
        sample = order[np.linspace(0, len(order) - 1, 180, dtype=int)]
        x = np.arange(len(sample))
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.fill_between(x, quantile_predictions[0.1][sample], quantile_predictions[0.9][sample], color="#dbeafe", label="P10-P90")
        ax.plot(x, quantile_predictions[0.5][sample], color=colors["blue"], linewidth=1.5, label="P50")
        ax.scatter(x, actual[sample], s=10, color=colors["ink"], alpha=0.65, label="Actual")
        ax.set_xlabel("Properties ordered by actual price (sampled)")
        ax.set_ylabel("Price (RM)")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1_000_000:.1f}M"))
        ax.legend(frameon=False)
        _style_axis(ax, "Quantile prediction intervals", "OOF P10, P50, and P90 on the original RM scale")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "08_quantile_prediction_intervals.png", dpi=160)
        plt.close(fig)


def json_default(value):
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)}")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    protected_before = {path.as_posix(): sha256(path) for path in PROTECTED_PATHS}
    data, dataset_profile = load_data()
    X = data[MODEL_FEATURES]
    y = data[TARGET_COLUMN].to_numpy(float)
    folds = shared_folds(len(data))
    fold_id = np.empty(len(data), dtype=int)
    for number, (_, validation_index) in enumerate(folds, start=1):
        fold_id[validation_index] = number

    evaluations: dict[str, dict] = {}
    predictions: dict[str, np.ndarray] = {}
    baseline_eval = evaluate_fixed(
        "Random Forest",
        build_ppsf_estimator("Random Forest"),
        X,
        y,
        folds,
        len(MODEL_FEATURES),
    )
    evaluations["random_forest"] = baseline_eval["result"]
    predictions["random_forest"] = baseline_eval["prediction"]
    baseline_verification = assert_baseline(baseline_eval["result"])
    print("Baseline reproduced exactly; running nested advanced-model searches.", flush=True)

    availability = dependency_status()
    parameters = candidate_parameters()
    for family in ("catboost", "xgboost", "lightgbm"):
        if not availability[family]:
            continue
        evaluated = run_nested_family(family, parameters[family], X, y, folds)
        evaluations[family] = evaluated["result"]
        predictions[family] = evaluated["prediction"]
        print(f"Finished {evaluated['result']['name']}.", flush=True)

    huber_eval = run_nested_family("huber", parameters["huber"], X, y, folds)
    evaluations["huber"] = huber_eval["result"]
    predictions["huber"] = huber_eval["prediction"]
    print("Finished Huber boosting.", flush=True)

    if availability["lightgbm"]:
        tweedie_params = tweedie_candidates()
        tweedie_eval = run_nested_family("lightgbm", tweedie_params, X, y, folds)
        tweedie_eval["result"]["name"] = "Tweedie LightGBM"
        evaluations["tweedie"] = tweedie_eval["result"]
        predictions["tweedie"] = tweedie_eval["prediction"]
        print("Finished Tweedie variance-power search.", flush=True)

    target_transformations = {}
    existing_base = build_existing_base_regressor("Random Forest")
    for strategy, label in (
        ("log_ppsf", "RF Log1p PPSF"),
        ("log_price", "RF Log1p Total Price"),
        ("boxcox_price", "RF Box-Cox Total Price"),
    ):
        evaluated = evaluate_fixed(
            label,
            TargetStrategyRegressor(existing_base, strategy=strategy),
            X,
            y,
            folds,
            len(MODEL_FEATURES),
        )
        key = f"target_{strategy}"
        evaluations[key] = evaluated["result"]
        predictions[key] = evaluated["prediction"]
        target_transformations[strategy] = evaluated["result"]
    target_transformations["raw_ppsf"] = evaluations["random_forest"]
    print("Finished reversible target transformations.", flush=True)

    feature_family = "lightgbm" if availability["lightgbm"] else (
        "xgboost" if availability["xgboost"] else (
            "catboost" if availability["catboost"] else "huber"
        )
    )
    feature_params = parameters[feature_family][1]
    feature_variants = {
        "full": dict(include_micro=True, include_interactions=True),
        "minus_micro_market": dict(include_micro=False, include_interactions=True),
        "minus_interactions": dict(include_micro=True, include_interactions=False),
    }
    feature_results = {}
    feature_schemas = {}
    for variant, options in feature_variants.items():
        estimator, numerical, categorical = build_feature_estimator(
            feature_family, feature_params, **options
        )
        evaluated = evaluate_fixed(
            f"{feature_family.title()} FE {variant}",
            estimator,
            X,
            y,
            folds,
            len(numerical) + len(categorical),
        )
        key = f"feature_{variant}"
        evaluations[key] = evaluated["result"]
        predictions[key] = evaluated["prediction"]
        feature_results[variant] = evaluated["result"]
        feature_schemas[variant] = {"numerical": numerical, "categorical": categorical}

    comparable_key = feature_family
    feature_improved = (
        comparable_key in evaluations
        and feature_results["full"]["metrics"]["RMSE_RM"]
        < evaluations[comparable_key]["metrics"]["RMSE_RM"]
    )
    if feature_improved:
        for variant, flags in {
            "minus_city": dict(remove_city=True),
            "minus_building_developer": dict(remove_building_developer=True),
        }.items():
            estimator, numerical, categorical = build_feature_estimator(
                feature_family,
                feature_params,
                include_micro=True,
                include_interactions=True,
                **flags,
            )
            evaluated = evaluate_fixed(
                f"{feature_family.title()} FE {variant}", estimator, X, y, folds,
                len(numerical) + len(categorical),
            )
            key = f"feature_{variant}"
            evaluations[key] = evaluated["result"]
            predictions[key] = evaluated["prediction"]
            feature_results[variant] = evaluated["result"]
            feature_schemas[variant] = {"numerical": numerical, "categorical": categorical}
    print("Finished micro-market and size feature experiment.", flush=True)

    quantile_results, quantile_predictions = {}, {}
    if availability["lightgbm"]:
        for alpha in (0.1, 0.5, 0.9):
            estimator = build_estimator(
                "lightgbm",
                quantile_parameters(alpha),
                NUMERICAL_FEATURES,
                CATEGORICAL_FEATURES,
                target_strategy="ppsf",
            )
            evaluated = evaluate_fixed(
                f"LightGBM P{int(alpha * 100)}", estimator, X, y, folds,
                len(MODEL_FEATURES),
            )
            quantile_results[f"P{int(alpha * 100)}"] = evaluated["result"]
            quantile_predictions[alpha] = evaluated["prediction"]
        quantile_quality = quantile_diagnostics(y, quantile_predictions)
    else:
        quantile_quality = {"status": "skipped; LightGBM unavailable"}
    print("Finished quantile models.", flush=True)

    ensemble_bases = {"Random Forest": build_ppsf_estimator("Random Forest")}
    for family, label in (
        ("catboost", "CatBoost"),
        ("xgboost", "XGBoost"),
        ("lightgbm", "LightGBM"),
    ):
        if availability[family]:
            ensemble_bases[label] = build_estimator(
                family,
                parameters[family][1],
                NUMERICAL_FEATURES,
                CATEGORICAL_FEATURES,
                target_strategy="ppsf",
            )
    ensembles = evaluate_nested_ensembles(
        ensemble_bases, X, y, folds, len(MODEL_FEATURES)
    )
    for key, evaluated in ensembles.items():
        evaluations[key] = evaluated["result"]
        predictions[key] = evaluated["prediction"]
    print("Finished leakage-safe blending and stacking.", flush=True)

    baseline = evaluations["random_forest"]
    for result in evaluations.values():
        attach_baseline_comparison(result, baseline)
    comparison = pd.DataFrame(
        [comparison_row(key, result) for key, result in evaluations.items()]
    ).sort_values("RMSE_RM", ignore_index=True)

    standard_prediction_keys = [
        key for key in ("random_forest", "catboost", "xgboost", "lightgbm")
        if key in predictions
    ]
    residual_corr = residual_correlations(
        y, {key: predictions[key] for key in standard_prediction_keys}
    )
    residual_corr.to_csv(RESIDUAL_CORRELATIONS_CSV)

    oof = pd.DataFrame(
        {
            "listing_id": data["listing_id"],
            "fold": fold_id,
            "actual_price_RM": y,
            "premium_top_5_percent": y >= np.quantile(y, 0.95),
        }
    )
    for key, prediction in predictions.items():
        oof[f"prediction__{key}"] = prediction
    for alpha, prediction in quantile_predictions.items():
        oof[f"prediction__quantile_p{int(alpha * 100)}"] = prediction
    for key, result in evaluations.items():
        values = {row["fold"]: row["RMSE_RM"] for row in result["folds"]}
        for fold_number, value in values.items():
            oof.loc[fold_number - 1, f"fold_rmse__{key}"] = value
    oof.to_csv(OOF_CSV, index=False)
    comparison.to_csv(COMPARISON_CSV, index=False)

    fold_table = pd.DataFrame(
        [
            {"Model_Key": key, "Model": result["name"], **fold}
            for key, result in evaluations.items()
            for fold in result["folds"]
        ]
    )
    fold_table.to_csv(FOLD_METRICS_CSV, index=False)
    premium_table = pd.DataFrame(
        [
            {
                "Model_Key": key,
                "Model": result["name"],
                **{f"Top5_{name}": value for name, value in result["metrics"]["top_5_percent"].items()},
                **{f"Remaining95_{name}": value for name, value in result["metrics"]["remaining_95_percent"].items()},
            }
            for key, result in evaluations.items()
        ]
    )
    premium_table.to_csv(PREMIUM_METRICS_CSV, index=False)
    if quantile_results:
        pd.DataFrame(
            [
                {"Quantile": key, **result["metrics"]}
                for key, result in quantile_results.items()
            ]
        ).drop(columns=["top_5_percent", "remaining_95_percent"]).to_csv(
            QUANTILE_METRICS_CSV, index=False
        )

    best_key = comparison.iloc[0]["Model_Key"]
    best = evaluations[best_key]
    best_mae_key = comparison.sort_values("MAE_RM").iloc[0]["Model_Key"]
    best_premium_key = comparison.sort_values("Top5_RMSE_RM").iloc[0]["Model_Key"]
    improvement = -best["comparison_vs_random_forest"]["RMSE_percentage_change"]
    generator = np.random.default_rng(42)
    bootstrap_rmse_differences = []
    bootstrap_mae_differences = []
    positions = np.arange(len(y))
    for _ in range(5000):
        sample = generator.choice(positions, len(positions), replace=True)
        bootstrap_rmse_differences.append(
            np.sqrt(np.mean(np.square(y[sample] - predictions[best_key][sample])))
            - np.sqrt(
                np.mean(np.square(y[sample] - predictions["random_forest"][sample]))
            )
        )
        bootstrap_mae_differences.append(
            np.mean(np.abs(y[sample] - predictions[best_key][sample]))
            - np.mean(
                np.abs(y[sample] - predictions["random_forest"][sample])
            )
        )
    rmse_interval = np.quantile(bootstrap_rmse_differences, [0.025, 0.5, 0.975])
    mae_interval = np.quantile(bootstrap_mae_differences, [0.025, 0.5, 0.975])
    best_fold_rmse = {row["fold"]: row["RMSE_RM"] for row in best["folds"]}
    baseline_fold_rmse = {
        row["fold"]: row["RMSE_RM"] for row in baseline["folds"]
    }
    improved_fold_count = sum(
        best_fold_rmse[fold] < baseline_fold_rmse[fold]
        for fold in best_fold_rmse
    )
    uncertainty = {
        "method": "paired 5000-sample row bootstrap on fixed OOF predictions",
        "limitation": "Conditions on fitted CV models; repeated nested CV or a fresh holdout is stronger evidence.",
        "RMSE_difference_RM_95_percent_interval": rmse_interval.tolist(),
        "MAE_difference_RM_95_percent_interval": mae_interval.tolist(),
        "outer_folds_with_lower_RMSE": int(improved_fold_count),
        "outer_fold_count": 5,
    }
    promotion_checks = {
        "overall_RMSE_improves_by_at_least_1_percent": improvement >= 1.0,
        "MAE_improves_or_within_0_5_percent": best["comparison_vs_random_forest"]["MAE_percentage_change"] <= 0.5,
        "R2_stable_or_improves": best["comparison_vs_random_forest"]["R2_difference"] >= -0.001,
        "premium_RMSE_not_over_5_percent_worse": best["comparison_vs_random_forest"]["Top5_RMSE_percentage_change"] <= 5.0,
        "absolute_mean_bias_below_2_percent_mean_price": abs(best["metrics"]["Mean_Error_RM"]) <= y.mean() * 0.02,
        "R2_gap_not_over_0_05_worse_than_baseline": best["generalization_gap"]["R2_gap"] <= baseline["generalization_gap"]["R2_gap"] + 0.05,
        "paired_bootstrap_RMSE_interval_excludes_no_improvement": bool(rmse_interval[2] < 0.0),
    }
    promote = best_key != "random_forest" and all(promotion_checks.values())
    recommendation = (
        f"Promote {best['name']} to a separate production-validation stage; it passed all measured promotion safeguards."
        if promote
        else f"Retain the current Random Forest production candidate. Advance {best['name']} to repeated nested CV or a fresh holdout/shadow evaluation because its point-estimate gain did not exclude no improvement under the paired OOF bootstrap."
    )

    feature_summary = {
        "base_family": feature_family,
        "base_parameters": feature_params,
        "size_features": list(SIZE_NUMERICAL_FEATURES) + ["size_band"],
        "location_and_size_interactions": list(INTERACTION_FEATURES),
        "micro_market_levels": list(MICRO_LEVELS),
        "micro_market_statistics": list(MICRO_STATS),
        "micro_market_features": list(MICRO_FEATURES),
        "training_row_encoding": "5-fold OOF inside each outer training fold",
        "validation_encoding": "outer-training rows only with hierarchical fallback",
        "fallback_order": "building/developer -> city+property_type -> city -> state -> global training statistic",
        "feature_engineering_improved_base_family": feature_improved,
        "schemas": feature_schemas,
        "ablation_RMSE_RM": {
            key: value["metrics"]["RMSE_RM"] for key, value in feature_results.items()
        },
    }
    FEATURE_SUMMARY_PATH.write_text(
        json.dumps(feature_summary, indent=2, default=json_default), encoding="utf-8"
    )

    protected_after = {path.as_posix(): sha256(path) for path in PROTECTED_PATHS}
    if protected_before != protected_after:
        changed = [path for path in protected_before if protected_before[path] != protected_after[path]]
        raise AssertionError(f"Protected files changed during experiment: {changed}")

    create_figures(
        comparison,
        oof,
        f"prediction__{best_key}",
        residual_corr,
        quantile_predictions,
    )
    payload = {
        "dataset": dataset_profile,
        "baseline": {**baseline, "verification": baseline_verification},
        "models": {
            key: evaluations.get(key, {"status": "not run"})
            for key in ("random_forest", "catboost", "xgboost", "lightgbm", "huber", "tweedie")
        },
        "target_transformations": target_transformations,
        "feature_engineering": feature_results,
        "quantile_models": {"models": quantile_results, "quality": quantile_quality},
        "blending": evaluations["best_blend"],
        "stacking": evaluations["best_stack"],
        "premium_segment": {
            "threshold_RM": float(np.quantile(y, 0.95)),
            "count": int(np.sum(y >= np.quantile(y, 0.95))),
            "best_model": best_premium_key,
            "best_metrics": evaluations[best_premium_key]["metrics"]["top_5_percent"],
        },
        "generalization_gap": {
            key: result["generalization_gap"] for key, result in evaluations.items()
        },
        "residual_correlations": residual_corr.to_dict(),
        "best_overall_model": best_key,
        "best_rmse_model": best_key,
        "best_mae_model": best_mae_key,
        "best_premium_model": best_premium_key,
        "promotion_checks": promotion_checks,
        "uncertainty": uncertainty,
        "recommendation": recommendation,
        "leakage_audit": {
            "shared_outer_folds": True,
            "preprocessing_fit_on_training_rows_only": True,
            "native_catboost_categories_not_target_encoded": True,
            "micro_market_aggregates_training_fold_only": True,
            "micro_market_training_rows_cross_fitted": True,
            "boxcox_lambda_outer_training_fold_only": True,
            "blend_weights_inner_oof_training_only": True,
            "stacker_inner_oof_training_only": True,
            "validation_targets_used_for_model_fitting": False,
            "target_winsorization_or_capping": False,
            "premium_rows_deleted": False,
            "arbitrary_prediction_clipping": False,
            "protected_files_unchanged": True,
        },
        "dependency_status": availability,
        "skipped": {
            "log_cosh": "Skipped: no direct verified objective was available in the selected stable APIs; an unverified custom gradient/Hessian was intentionally not introduced."
        },
        "artifacts": [
            path.relative_to(PROJECT_ROOT).as_posix()
            for path in (
                RESULTS_PATH, COMPARISON_CSV, OOF_CSV, FEATURE_SUMMARY_PATH,
                RESIDUAL_CORRELATIONS_CSV, FOLD_METRICS_CSV,
                PREMIUM_METRICS_CSV, QUANTILE_METRICS_CSV,
            )
            if path.exists() or path == RESULTS_PATH
        ],
    }
    RESULTS_PATH.write_text(
        json.dumps(payload, indent=2, default=json_default), encoding="utf-8"
    )

    display = comparison[["Model", "RMSE_RM", "MAE_RM", "R2", "Top5_RMSE_RM", "R2_Gap"]]
    print("\n" + display.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
    print(f"\nOverall best: {best['name']}")
    print(f"Recommendation: {recommendation}")
    print(f"Saved artifacts in: {EXPERIMENT_DIR}")


if __name__ == "__main__":
    main()
