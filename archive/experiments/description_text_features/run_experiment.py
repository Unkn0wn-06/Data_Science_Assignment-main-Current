"""Run the leakage-safe global description text feature experiment."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.description_text_features.evaluation import (
    complete_metrics,
    paired_bootstrap,
    price_band_masks,
    price_band_table,
    regression_summary,
)
from experiments.description_text_features.model_builders import (
    fit_lightgbm_fold,
    fit_text_ridge_fold,
)
from experiments.description_text_features.regex_features import (
    FEATURE_TO_GROUP,
    REGEX_GROUPS,
    extract_regex_features,
    frequency_table,
)
from experiments.description_text_features.text_cleaning import link_descriptions
from experiments.description_text_features.tfidf_features import FoldTextTransformer
from experiments.noncoordinate_target_encoding.target_encoding import (
    DEFAULT_M_VALUES,
    MEstimateTargetEncoder,
)


EXPERIMENT = ROOT / "experiments" / "description_text_features"
FIGURES = EXPERIMENT / "figures"
DATA_PATH = ROOT / "data" / "processed" / "enhanced_city_dataset.csv"
RAW_PATH = ROOT / "data" / "raw" / "houses.csv"
SVD_COUNTS = (10, 20, 30, 50)
PREMIUM_THRESHOLD = 905_000.0
BLUE = "#16697A"; GOLD = "#D6A84B"; NEUTRAL = "#8D99AE"; INK = "#263238"


def _json_default(value):
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Path): return value.as_posix()
    raise TypeError(type(value).__name__)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def _protected_snapshot():
    paths = [
        RAW_PATH, DATA_PATH,
        ROOT / "results" / "enhanced_city" / "model_comparison.json",
        ROOT / "results" / "best_model" / "best_model_summary.json",
        ROOT / "prototype" / "app.py", ROOT / "app.py",
    ]
    for directory in (
        ROOT / "experiments" / "advanced_real_estate_models",
        ROOT / "experiments" / "noncoordinate_target_encoding",
        ROOT / "experiments" / "premium_mixture_of_experts",
    ):
        paths.extend(path for path in directory.rglob("*") if path.is_file())
    return {str(path.relative_to(ROOT)): _sha256(path) for path in sorted(set(paths))}


def _load_references(frame):
    advanced = pd.read_csv(ROOT / "experiments" / "advanced_real_estate_models" / "oof_predictions.csv")
    advanced = advanced.set_index("listing_id").loc[frame["listing_id"].astype(int)].reset_index()
    noncoordinate = pd.read_csv(ROOT / "experiments" / "noncoordinate_target_encoding" / "oof_predictions.csv")
    building = noncoordinate[noncoordinate["variant"] == "building_name_te"].set_index("row_id").loc[frame["listing_id"].astype(int)].reset_index()
    actual = frame["price"].to_numpy(float)
    if not np.array_equal(advanced["actual_price_RM"].to_numpy(float), actual) or not np.array_equal(building["actual_price_RM"].to_numpy(float), actual):
        raise AssertionError("Verified reference OOF rows do not align to canonical data.")
    predictions = {
        "random_forest_reference": advanced["prediction__random_forest"].to_numpy(float),
        "lightgbm_structured_reference": advanced["prediction__feature_minus_micro_market"].to_numpy(float),
        "building_name_te_reference": building["predicted_price_RM"].to_numpy(float),
    }
    return predictions, advanced["fold"].to_numpy(int)


def _record_fold(variant, fold, train_index, validation_index, y, prediction, global_masks):
    metrics = regression_summary(y[validation_index], prediction)
    premium = y[validation_index] >= PREMIUM_THRESHOLD
    top = regression_summary(y[validation_index][premium], prediction[premium])
    p99 = global_masks["P99_P100"][validation_index]
    extreme = regression_summary(y[validation_index][p99], prediction[p99])
    return {"variant": variant, "fold": fold, "training_rows": len(train_index), "validation_rows": len(validation_index), "RMSE_RM": metrics["RMSE_RM"], "MAE_RM": metrics["MAE_RM"], "R2": metrics["R2"], "Top5_RMSE_RM": top["RMSE_RM"], "Top5_MAE_RM": top["MAE_RM"], "P99_RMSE_RM": extreme["RMSE_RM"]}


def _fit_variant(variant, fold, X, y, train_index, validation_index, train_dense, validation_dense, predictions, train_metrics, fold_rows, importance_rows, global_masks, building_te=False):
    output = fit_lightgbm_fold(X.iloc[train_index], y[train_index], X.iloc[validation_index], train_dense, validation_dense, building_te=building_te)
    predictions.setdefault(variant, np.empty(len(y), dtype=float))[validation_index] = output["validation_prediction"]
    train_metrics[variant].append(regression_summary(y[train_index], output["training_prediction"]))
    fold_rows.append(_record_fold(variant, fold, train_index, validation_index, y, output["validation_prediction"], global_masks))
    for feature, importance in output["feature_importance"].items():
        importance_rows.append({"variant": variant, "fold": fold, "feature": feature, "importance": importance, "feature_family": "regex" if feature in FEATURE_TO_GROUP else "svd" if feature.startswith("description_svd_") else "building_te" if feature == "building_name_te" else "structured"})
    return output


def _select_svd_inner(X_outer, y_outer, text_outer):
    inner = list(KFold(n_splits=3, shuffle=True, random_state=42).split(X_outer))
    scores = {}
    for count in SVD_COUNTS:
        inner_prediction = np.empty(len(X_outer), dtype=float)
        for inner_train, inner_validation in inner:
            transformer = FoldTextTransformer(count).fit(text_outer.iloc[inner_train])
            train_dense = transformer.transform(text_outer.iloc[inner_train], X_outer.index[inner_train])
            validation_dense = transformer.transform(text_outer.iloc[inner_validation], X_outer.index[inner_validation])
            output = fit_lightgbm_fold(X_outer.iloc[inner_train], y_outer[inner_train], X_outer.iloc[inner_validation], train_dense, validation_dense)
            inner_prediction[inner_validation] = output["validation_prediction"]
        scores[count] = regression_summary(y_outer, inner_prediction)["RMSE_RM"]
    selected = min(SVD_COUNTS, key=scores.get)
    return selected, {str(key): value for key, value in scores.items()}


def _building_te_features(X_train, y_train, X_validation):
    size = pd.to_numeric(X_train["property_size_sqft"], errors="coerce").to_numpy(float)
    ppsf = np.asarray(y_train, dtype=float) / size
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = {}; encoded_by_m = {}
    for m in DEFAULT_M_VALUES:
        encoded = MEstimateTargetEncoder(("building_name",), m=float(m)).fit_transform_oof(X_train, ppsf, cv)
        scores[float(m)] = float(np.sqrt(np.mean(np.square(ppsf - encoded["building_name_te"].to_numpy(float)))))
        encoded_by_m[float(m)] = encoded
    selected = min(scores, key=scores.get)
    encoder = MEstimateTargetEncoder(("building_name",), m=selected)
    train_encoded = encoder.fit_transform_oof(X_train, ppsf, cv)
    validation_encoded = encoder.transform(X_validation)
    return train_encoded, validation_encoded, selected, scores


def _evaluate_regex_ablation(group_name, eligible_by_group, X, y, folds, regex, predictions, train_metrics, fold_rows, importance_rows, global_masks):
    variant = f"regex_group_{group_name}"
    columns = eligible_by_group[group_name]
    for fold, (train_index, validation_index) in enumerate(folds, 1):
        _fit_variant(variant, fold, X, y, train_index, validation_index, regex.loc[train_index, columns], regex.loc[validation_index, columns], predictions, train_metrics, fold_rows, importance_rows, global_masks)


def _metric_row(variant, metrics, architecture):
    return {"variant": variant, "architecture": architecture, "RMSE_RM": metrics["RMSE_RM"], "MAE_RM": metrics["MAE_RM"], "R2": metrics["R2"], "Adjusted_R2": metrics["Adjusted_R2"], "Median_AE_RM": metrics["Median_AE_RM"], "Mean_Error_RM": metrics["Mean_Error_RM"], "Median_Error_RM": metrics["Median_Error_RM"], "Top5_RMSE_RM": metrics["top_5_percent"]["RMSE_RM"], "Top5_MAE_RM": metrics["top_5_percent"]["MAE_RM"], "Remaining95_RMSE_RM": metrics["remaining_95_percent"]["RMSE_RM"], "Remaining95_MAE_RM": metrics["remaining_95_percent"]["MAE_RM"], "P95_P99_RMSE_RM": metrics["P95_P99"]["RMSE_RM"], "P95_P99_MAE_RM": metrics["P95_P99"]["MAE_RM"], "P99_P100_RMSE_RM": metrics["P99_P100"]["RMSE_RM"], "P99_P100_MAE_RM": metrics["P99_P100"]["MAE_RM"]}


def _predictor_count(variant, eligible, eligible_by_group, selections):
    base = 42
    if variant == "random_forest_reference": return 32
    if variant in {"lightgbm_structured_reference", "building_name_te_reference", "structured_baseline_reproduced", "basic_text_existing_control"}: return 42
    if variant == "regex_expanded": return base + len(eligible)
    if variant.startswith("regex_group_"): return base + len(eligible_by_group[variant.removeprefix("regex_group_")])
    if variant.startswith("tfidf_svd_"): return base + int(variant.rsplit("_", 1)[1])
    if variant == "char_tfidf_svd_20": return base + 20
    if variant in {"regex_svd_nested", "regex_svd_building_te_nested"}: return base + len(eligible) + int(round(np.mean([row["selected_components"] for row in selections])))
    if variant == "text_only_tfidf_ridge": return 5000
    if variant == "structured_text_ridge_blend_80_20": return 5042
    raise KeyError(f"Unknown predictor count for {variant}")


def _title(ax, title, subtitle):
    ax.set_title(title, loc="left", color=INK, fontsize=15, pad=24)
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, color="#5F6B73", fontsize=9, va="bottom")


def _bar(comparison, metric, filename, title, subtitle):
    data = comparison.sort_values(metric).head(14)
    colors = [NEUTRAL if "reference" in name else BLUE for name in data["variant"]]
    fig, ax = plt.subplots(figsize=(13, 7.5)); ax.barh(data["variant"], data[metric], color=colors, edgecolor=INK, linewidth=.5)
    ax.invert_yaxis(); ax.set_xlabel("RM"); ax.set_xlim(left=0); ax.grid(axis="x", alpha=.2); _title(ax, title, subtitle)
    fig.subplots_adjust(left=.38, right=.98, bottom=.10, top=.86); fig.savefig(FIGURES / filename, dpi=160); plt.close(fig)


def _make_figures(comparison, bands, fold_frame, oof_frame, importance, best, importance_variant):
    _bar(comparison, "RMSE_RM", "01_model_rmse_comparison.png", "OOF RMSE comparison", "Original total price in RM; all 3,791 canonical listings")
    _bar(comparison, "MAE_RM", "02_model_mae_comparison.png", "OOF MAE comparison", "Original total price in RM; all 3,791 canonical listings")
    _bar(comparison, "Top5_RMSE_RM", "03_top5_rmse_comparison.png", "Top-5% RMSE comparison", "Actual price at or above RM905,000; n=190")
    _bar(comparison, "P99_P100_RMSE_RM", "04_p99_rmse_comparison.png", "Extreme-tail RMSE comparison", "Actual-price 99th–100th percentile; n=39")
    selected = ["random_forest_reference", "lightgbm_structured_reference", "building_name_te_reference", best]
    for metric, filename, title in (("RMSE_RM", "05_price_band_rmse.png", "RMSE by actual-price band"), ("MAE_RM", "06_price_band_mae.png", "MAE by actual-price band")):
        pivot = bands[bands["variant"].isin(selected)].pivot(index="price_band", columns="variant", values=metric)
        colors = [NEUTRAL, "#AAB4BF", GOLD, BLUE][:len(pivot.columns)]
        ax = pivot.plot.bar(figsize=(12, 7), color=colors, edgecolor=INK, linewidth=.4); ax.set_ylabel("RM"); ax.set_xlabel("Actual-price percentile band"); ax.grid(axis="y", alpha=.2); _title(ax, title, "Shared OOF rows; original total-price RM")
        plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(FIGURES / filename, dpi=160); plt.close()
    stability = fold_frame[fold_frame["variant"].isin(selected)]
    fig, ax = plt.subplots(figsize=(10, 6)); style_map = {
        "building_name_te_reference": ("-", "o", NEUTRAL),
        "lightgbm_structured_reference": ("--", "s", GOLD),
        "random_forest_reference": (":", "^", "#B8C0C8"),
        best: ("-", "D", BLUE),
    }
    for variant, group in stability.groupby("variant"):
        style, marker, color = style_map[variant]
        ax.plot(group["fold"], group["RMSE_RM"], linestyle=style, marker=marker, color=color, label=variant)
    ax.set(xlabel="Outer fold", ylabel="RMSE (RM)"); ax.set_xticks(range(1, 6)); ax.grid(alpha=.2); ax.legend(fontsize=8); _title(ax, "Fold RMSE stability", "Five shared shuffled KFold partitions; random_state=42")
    fig.tight_layout(); fig.savefig(FIGURES / "07_fold_rmse_stability.png", dpi=160); plt.close(fig)
    best_rows = oof_frame[oof_frame["variant"] == best].sort_values("row_index")
    fig, ax = plt.subplots(figsize=(7, 7)); ax.scatter(best_rows["actual_price_RM"], best_rows["predicted_price_RM"], s=11, alpha=.45, color=BLUE)
    limits = [min(best_rows["actual_price_RM"].min(), best_rows["predicted_price_RM"].min()), max(best_rows["actual_price_RM"].max(), best_rows["predicted_price_RM"].max())]; ax.plot(limits, limits, "--", color=INK)
    ax.set(xlabel="Actual price (RM)", ylabel="OOF predicted price (RM)"); _title(ax, "Actual versus predicted price", f"Best balanced new text candidate: {best}; n=3,791")
    fig.tight_layout(); fig.savefig(FIGURES / "08_actual_vs_predicted_best.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 6)); ax.scatter(best_rows["actual_price_RM"], best_rows["residual_RM"], s=11, alpha=.45, color=BLUE); ax.axhline(0, linestyle="--", color=INK)
    ax.set(xlabel="Actual price (RM)", ylabel="Prediction − actual (RM)"); _title(ax, "Residuals versus actual price", f"Best balanced new text candidate: {best}; n=3,791")
    fig.tight_layout(); fig.savefig(FIGURES / "09_residuals_vs_actual_best.png", dpi=160); plt.close(fig)
    averaged = importance[importance["variant"] == importance_variant].groupby(["feature", "feature_family"], as_index=False)["importance"].mean()
    regex_top = averaged[averaged["feature_family"] == "regex"].nlargest(10, "importance").sort_values("importance")
    svd_top = averaged[averaged["feature_family"] == "svd"].nlargest(10, "importance").sort_values("importance")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, data, color, label in ((axes[0], regex_top, GOLD, "Regex indicators"), (axes[1], svd_top, BLUE, "SVD components")):
        ax.barh(data["feature"], data["importance"], color=color, edgecolor=INK, linewidth=.4)
        ax.set_xlabel("Mean LightGBM split importance"); ax.set_xlim(left=0); ax.grid(axis="x", alpha=.2); ax.set_title(label, loc="left", color=INK)
    fig.suptitle("Regex and SVD feature importance", x=.08, ha="left", fontsize=15, color=INK)
    fig.text(.08, .925, f"{importance_variant}; SVD components are not assigned semantic labels", color="#5F6B73", fontsize=9)
    fig.subplots_adjust(left=.16, right=.98, bottom=.10, top=.84, wspace=.42); fig.savefig(FIGURES / "10_regex_feature_importance.png", dpi=160); plt.close(fig)


def main():
    started = time.perf_counter(); FIGURES.mkdir(parents=True, exist_ok=True)
    before = _protected_snapshot()
    frame = pd.read_csv(DATA_PATH)
    if len(frame) != 3791 or frame["listing_id"].nunique() != 3791: raise AssertionError("Canonical row grain changed.")
    descriptions, linkage = link_descriptions(RAW_PATH, frame["listing_id"])
    regex = extract_regex_features(descriptions); frequencies = frequency_table(regex, minimum_count=10)
    eligible = frequencies.loc[frequencies["modeled"], "feature"].tolist()
    eligible_by_group = {group: [name for name in names if name in eligible] for group, names in ((group, list(features)) for group, features in REGEX_GROUPS.items())}
    X = frame.drop(columns=["price"]); y = frame["price"].to_numpy(float)
    folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(X))
    generated_fold = np.empty(len(y), dtype=int)
    for fold, (_, validation_index) in enumerate(folds, 1): generated_fold[validation_index] = fold
    references, reference_fold = _load_references(frame)
    if not np.array_equal(generated_fold, reference_fold): raise AssertionError("Outer folds do not match verified references.")
    global_masks, boundaries = price_band_masks(y)

    predictions = {}; train_metrics = defaultdict(list); fold_rows = []; importance_rows = []; svd_rows = []; selections = []; te_records = []; blend_training_pairs = []
    for fold, (train_index, validation_index) in enumerate(folds, 1):
        X_train = X.iloc[train_index]; X_validation = X.iloc[validation_index]
        text_train = descriptions.iloc[train_index]; text_validation = descriptions.iloc[validation_index]
        baseline = _fit_variant("structured_baseline_reproduced", fold, X, y, train_index, validation_index, None, None, predictions, train_metrics, fold_rows, importance_rows, global_masks)

        regex_train = regex.loc[train_index, eligible]; regex_validation = regex.loc[validation_index, eligible]
        _fit_variant("regex_expanded", fold, X, y, train_index, validation_index, regex_train, regex_validation, predictions, train_metrics, fold_rows, importance_rows, global_masks)
        cache = {}
        for count in SVD_COUNTS:
            transformer = FoldTextTransformer(count).fit(text_train)
            train_dense = transformer.transform(text_train, X_train.index); validation_dense = transformer.transform(text_validation, X_validation.index)
            cache[count] = (train_dense, validation_dense, transformer)
            _fit_variant(f"tfidf_svd_{count}", fold, X, y, train_index, validation_index, train_dense, validation_dense, predictions, train_metrics, fold_rows, importance_rows, global_masks)
            cumulative = np.cumsum(transformer.explained_variance_ratio_)
            for component, ratio in enumerate(transformer.explained_variance_ratio_, 1): svd_rows.append({"fold": fold, "analyzer": "word", "n_components": count, "component": f"description_svd_{component:02d}", "explained_variance_ratio": ratio, "cumulative_explained_variance_ratio": cumulative[component - 1], "vocabulary_size": transformer.vocabulary_size_})

        selected, inner_scores = _select_svd_inner(X_train, y[train_index], text_train.reset_index(drop=True))
        selections.append({"fold": fold, "selected_components": selected, "inner_RMSE_RM": inner_scores})
        selected_train, selected_validation, _ = cache[selected]
        combined_train = regex_train.join(selected_train); combined_validation = regex_validation.join(selected_validation)
        _fit_variant("regex_svd_nested", fold, X, y, train_index, validation_index, combined_train, combined_validation, predictions, train_metrics, fold_rows, importance_rows, global_masks)

        te_train, te_validation, selected_m, te_scores = _building_te_features(X_train, y[train_index], X_validation)
        te_records.append({"fold": fold, "selected_m": selected_m, "training_only_proxy_scores": {str(key): value for key, value in te_scores.items()}})
        combined_te_train = combined_train.join(te_train); combined_te_validation = combined_validation.join(te_validation)
        _fit_variant("regex_svd_building_te_nested", fold, X, y, train_index, validation_index, combined_te_train, combined_te_validation, predictions, train_metrics, fold_rows, importance_rows, global_masks, building_te=True)

        ridge_transformer = cache[50][2]
        ridge_train = ridge_transformer.vectorizer_.transform(text_train); ridge_validation = ridge_transformer.vectorizer_.transform(text_validation)
        ridge = fit_text_ridge_fold(ridge_train, y[train_index], X_train["property_size_sqft"], ridge_validation, X_validation["property_size_sqft"])
        predictions.setdefault("text_only_tfidf_ridge", np.empty(len(y)))[validation_index] = ridge["validation_prediction"]
        train_metrics["text_only_tfidf_ridge"].append(regression_summary(y[train_index], ridge["training_prediction"]))
        fold_rows.append(_record_fold("text_only_tfidf_ridge", fold, train_index, validation_index, y, ridge["validation_prediction"], global_masks))
        blend_training_pairs.append((train_index, baseline["training_prediction"], ridge["training_prediction"]))
        print(f"Completed primary outer fold {fold}/5; nested SVD selected {selected} components.", flush=True)

    # The existing basic text columns are already in the structured schema by design.
    predictions["basic_text_existing_control"] = predictions["structured_baseline_reproduced"].copy()
    train_metrics["basic_text_existing_control"] = list(train_metrics["structured_baseline_reproduced"])
    for row in [item.copy() for item in fold_rows if item["variant"] == "structured_baseline_reproduced"]:
        row["variant"] = "basic_text_existing_control"; fold_rows.append(row)

    preliminary = {name: complete_metrics(y, predicted, predictors=X.shape[1] + 60) for name, predicted in predictions.items()}
    ridge_corr_gate = float(np.corrcoef(predictions["text_only_tfidf_ridge"] - y, predictions["structured_baseline_reproduced"] - y)[0, 1])
    blend_eligible = ridge_corr_gate < 0.80 and preliminary["text_only_tfidf_ridge"]["RMSE_RM"] < preliminary["structured_baseline_reproduced"]["RMSE_RM"] * 1.5
    if blend_eligible:
        blend_name = "structured_text_ridge_blend_80_20"
        predictions[blend_name] = 0.8 * predictions["structured_baseline_reproduced"] + 0.2 * predictions["text_only_tfidf_ridge"]
        for fold, ((train_index, baseline_train, ridge_train), (_, validation_index)) in enumerate(zip(blend_training_pairs, folds), 1):
            train_metrics[blend_name].append(regression_summary(y[train_index], 0.8 * baseline_train + 0.2 * ridge_train))
            fold_rows.append(_record_fold(blend_name, fold, train_index, validation_index, y, predictions[blend_name][validation_index], global_masks))
        print("Text-only residuals met the evidence gate; completed the fixed 80/20 diagnostic blend.", flush=True)
    regex_helped = preliminary["regex_expanded"]["RMSE_RM"] < preliminary["structured_baseline_reproduced"]["RMSE_RM"]
    if regex_helped:
        for group in REGEX_GROUPS:
            if eligible_by_group[group]: _evaluate_regex_ablation(group, eligible_by_group, X, y, folds, regex, predictions, train_metrics, fold_rows, importance_rows, global_masks)
        print("Regex improved OOF RMSE; completed group ablations.", flush=True)

    word_helped = min(preliminary[f"tfidf_svd_{count}"]["RMSE_RM"] for count in SVD_COUNTS) < preliminary["structured_baseline_reproduced"]["RMSE_RM"]
    char_tested = not word_helped
    if char_tested:
        for fold, (train_index, validation_index) in enumerate(folds, 1):
            transformer = FoldTextTransformer(20, analyzer="char").fit(descriptions.iloc[train_index])
            train_dense = transformer.transform(descriptions.iloc[train_index], X.index[train_index]); validation_dense = transformer.transform(descriptions.iloc[validation_index], X.index[validation_index])
            _fit_variant("char_tfidf_svd_20", fold, X, y, train_index, validation_index, train_dense, validation_dense, predictions, train_metrics, fold_rows, importance_rows, global_masks)
            cumulative = np.cumsum(transformer.explained_variance_ratio_)
            for component, ratio in enumerate(transformer.explained_variance_ratio_, 1): svd_rows.append({"fold": fold, "analyzer": "char_wb", "n_components": 20, "component": f"description_svd_{component:02d}", "explained_variance_ratio": ratio, "cumulative_explained_variance_ratio": cumulative[component - 1], "vocabulary_size": transformer.vocabulary_size_})
        print("Word TF-IDF did not improve RMSE; completed controlled character ablation.", flush=True)

    all_predictions = {**references, **predictions}
    metrics = {name: complete_metrics(y, predicted, predictors=_predictor_count(name, eligible, eligible_by_group, selections)) for name, predicted in all_predictions.items()}
    architectures = {name: "reference" for name in references}
    for name in predictions:
        architectures[name] = "control" if name in {"structured_baseline_reproduced", "basic_text_existing_control"} else "ridge_text_only" if name == "text_only_tfidf_ridge" else "exploratory_fixed_blend" if name == "structured_text_ridge_blend_80_20" else "text_enhanced_lightgbm"
    comparison = pd.DataFrame([_metric_row(name, bundle, architectures[name]) for name, bundle in metrics.items()]).sort_values("RMSE_RM")
    new_candidates = [name for name in predictions if name not in {"structured_baseline_reproduced", "basic_text_existing_control", "text_only_tfidf_ridge", "structured_text_ridge_blend_80_20"}]
    best_rmse = min(new_candidates, key=lambda name: metrics[name]["RMSE_RM"]); best_mae = min(new_candidates, key=lambda name: metrics[name]["MAE_RM"]); best_tail = min(new_candidates, key=lambda name: metrics[name]["P99_P100"]["RMSE_RM"])
    best_balanced = min(new_candidates, key=lambda name: metrics[name]["RMSE_RM"] / 120201.22 + metrics[name]["MAE_RM"] / 61217.27 + metrics[name]["top_5_percent"]["RMSE_RM"] / 418749.39)

    for name, predicted in references.items():
        for fold, (_, validation_index) in enumerate(folds, 1): fold_rows.append(_record_fold(name, fold, np.setdiff1d(np.arange(len(y)), validation_index), validation_index, y, predicted[validation_index], global_masks))
    fold_frame = pd.DataFrame(fold_rows).sort_values(["variant", "fold"])
    band_frame = price_band_table(y, all_predictions)
    band_label = np.empty(len(y), dtype=object)
    for name, mask in global_masks.items(): band_label[mask] = name
    oof_parts = []
    for name, predicted in all_predictions.items():
        oof_parts.append(pd.DataFrame({"row_id": frame["listing_id"].astype(int), "row_index": np.arange(len(y)), "fold": generated_fold, "actual_price_RM": y, "predicted_price_RM": predicted, "residual_RM": predicted - y, "absolute_error_RM": np.abs(predicted - y), "premium_flag": y >= PREMIUM_THRESHOLD, "price_band": band_label, "variant": name}))
    oof_frame = pd.concat(oof_parts, ignore_index=True)
    importance = pd.DataFrame(importance_rows)

    generalization = {}
    for name in predictions:
        train_rmse = float(np.mean([row["RMSE_RM"] for row in train_metrics[name]])); train_mae = float(np.mean([row["MAE_RM"] for row in train_metrics[name]])); train_r2 = float(np.mean([row["R2"] for row in train_metrics[name]]))
        generalization[name] = {"Training_RMSE_RM": train_rmse, "OOF_RMSE_RM": metrics[name]["RMSE_RM"], "RMSE_gap_RM": metrics[name]["RMSE_RM"] - train_rmse, "Training_MAE_RM": train_mae, "OOF_MAE_RM": metrics[name]["MAE_RM"], "MAE_gap_RM": metrics[name]["MAE_RM"] - train_mae, "Training_R2": train_r2, "OOF_R2": metrics[name]["R2"], "R2_gap": train_r2 - metrics[name]["R2"]}

    residuals = pd.DataFrame({name: predicted - y for name, predicted in all_predictions.items()})
    residual_correlations = residuals.corr()
    bootstrap_tables = []
    for reference in references:
        table = paired_bootstrap(y, predictions[best_balanced], references[reference], y >= PREMIUM_THRESHOLD, draws=5000)
        table.insert(0, "candidate", best_balanced); table.insert(1, "reference", reference); bootstrap_tables.append(table)
    bootstrap = pd.concat(bootstrap_tables, ignore_index=True)
    wins = {}
    for reference in references:
        selected = fold_frame[fold_frame["variant"].isin([best_balanced, reference])]
        pivot_rmse = selected.pivot(index="fold", columns="variant", values="RMSE_RM"); pivot_mae = selected.pivot(index="fold", columns="variant", values="MAE_RM")
        wins[reference] = {"RMSE_folds_won": int((pivot_rmse[best_balanced] < pivot_rmse[reference]).sum()), "MAE_folds_won": int((pivot_mae[best_balanced] < pivot_mae[reference]).sum())}

    verified = metrics["lightgbm_structured_reference"]; reproduced = metrics["structured_baseline_reproduced"]
    reproduction = {key: reproduced[key] - verified[key] for key in ("RMSE_RM", "MAE_RM", "R2")}
    if any(abs(value) > 1e-8 for value in reproduction.values()): raise AssertionError(f"Structured baseline reproduction mismatch: {reproduction}")
    ridge_corr = float(residual_correlations.loc["text_only_tfidf_ridge", "structured_baseline_reproduced"])
    blend_tested = blend_eligible

    best_importance_variant = min([name for name in new_candidates if "regex_svd" in name], key=lambda name: metrics[name]["RMSE_RM"])
    top_importance = {}
    averaged = importance[importance["variant"] == best_importance_variant].groupby(["feature", "feature_family"], as_index=False)["importance"].mean()
    for family in ("structured", "regex", "svd", "building_te"):
        top_importance[family] = averaged[averaged["feature_family"] == family].nlargest(15, "importance").to_dict("records")

    _make_figures(comparison, band_frame, fold_frame, oof_frame, importance, best_balanced, best_importance_variant)
    frequencies.to_csv(EXPERIMENT / "regex_feature_frequencies.csv", index=False)
    svd_frame = pd.DataFrame(svd_rows); svd_frame.to_csv(EXPERIMENT / "svd_component_summary.csv", index=False)
    comparison.to_csv(EXPERIMENT / "model_comparison.csv", index=False); fold_frame.to_csv(EXPERIMENT / "fold_metrics.csv", index=False); oof_frame.to_csv(EXPERIMENT / "oof_predictions.csv", index=False)

    feature_summary = {"regex_indicator_count": len(regex.columns), "modeling_minimum_count": 10, "regex_retained_count": len(eligible), "retained_regex_features": eligible, "groups": {group: {"defined": list(features), "retained": eligible_by_group[group]} for group, features in REGEX_GROUPS.items()}, "tfidf_word_parameters": {"lowercase": True, "strip_accents": "unicode", "ngram_range": [1, 2], "min_df": 5, "max_df": 0.95, "max_features": 5000, "sublinear_tf": True}, "svd_counts_tested": list(SVD_COUNTS), "nested_component_selection": selections, "feature_importance_variant": best_importance_variant, "feature_importance": top_importance}
    with (EXPERIMENT / "feature_summary.json").open("w", encoding="utf-8") as handle: json.dump(feature_summary, handle, indent=2, default=_json_default)

    results = {
        "dataset": {"path": DATA_PATH.relative_to(ROOT).as_posix(), "sha256": _sha256(DATA_PATH), "rows": len(frame), "unique_listing_ids": frame["listing_id"].nunique(), "headline_rows_removed": 0, "premium_rows_removed": 0, "target_capped_or_winsorized": False},
        "description_linkage": linkage,
        "references": {name: metrics[name] for name in references},
        "baseline_reproduction": {"variant": "structured_baseline_reproduced", "metrics": reproduced, "differences_vs_verified": reproduction, "matched_at_1e-8": True, "basic_text_fields_already_present": ["description_length", "is_furnished", "is_renovated"]},
        "regex_features": {"defined": len(regex.columns), "retained": len(eligible), "minimum_count": 10, "frequencies": frequencies.to_dict("records"), "regex_helped_rmse": regex_helped, "group_ablation_run": regex_helped},
        "tfidf": {"word_parameters": {"ngram_range": [1, 2], "min_df": 5, "max_df": .95, "max_features": 5000, "sublinear_tf": True}, "vocabulary_sizes_by_fold": svd_frame[svd_frame["analyzer"] == "word"].groupby("fold")["vocabulary_size"].first().astype(int).to_dict(), "validation_text_influenced_fit": False, "character_ablation_tested": char_tested},
        "svd": {"counts_tested": list(SVD_COUNTS), "nested_selection": selections, "outer_fold_components_and_explained_variance_saved": True},
        "building_te_integration": {"included": True, "method": "outer-training-only inner-OOF M-estimate PPSF encoding", "per_fold": te_records, "historical_global_oof_used_as_training_feature": False},
        "variants": {name: metrics[name] for name in predictions},
        "price_band_performance": band_frame.to_dict("records"),
        "premium_performance": {name: {"top_5_percent": bundle["top_5_percent"], "remaining_95_percent": bundle["remaining_95_percent"], "underprediction": bundle["premium_underprediction"]} for name, bundle in metrics.items()},
        "extreme_tail_performance": {name: {"P95_P99": bundle["P95_P99"], "P99_P100": bundle["P99_P100"], "P99_underprediction": bundle["P99_underprediction"]} for name, bundle in metrics.items()},
        "generalization": generalization,
        "residual_correlations": residual_correlations.to_dict(),
        "text_only_ridge_diagnostic": {"metrics": metrics["text_only_tfidf_ridge"], "residual_correlation_with_structured": ridge_corr, "blend_eligibility_rule": "correlation < 0.80 and RMSE < 1.5x structured", "blend_tested": blend_tested, "blend_variant": "structured_text_ridge_blend_80_20" if blend_tested else None, "blend_metrics": metrics.get("structured_text_ridge_blend_80_20"), "blend_is_exploratory": True, "blend_skipped_reason": None if blend_tested else "Text-only Ridge did not meet the predefined complementary-and-competitive evidence gate."},
        "feature_importance": {"variant": best_importance_variant, "groups": top_importance, "svd_semantics_claimed": False},
        "bootstrap": {"candidate": best_balanced, "difference_definition": "candidate - reference; negative is better", "comparisons": bootstrap.to_dict("records"), "fixed_oof_limitation": "Does not include refitting or candidate-selection uncertainty."},
        "fold_wins": wins,
        "best_rmse_variant": best_rmse, "best_mae_variant": best_mae, "best_tail_variant": best_tail, "best_balanced_variant": best_balanced,
        "success_thresholds": {"RMSE_below_120201_22": metrics[best_balanced]["RMSE_RM"] < 120201.22, "MAE_below_61217_27": metrics[best_balanced]["MAE_RM"] < 61217.27, "Top5_RMSE_below_418749_39": metrics[best_balanced]["top_5_percent"]["RMSE_RM"] < 418749.39, "both_RMSE_and_MAE": metrics[best_balanced]["RMSE_RM"] < 120201.22 and metrics[best_balanced]["MAE_RM"] < 61217.27},
        "recommendation": "Do not promote automatically. Decide from balanced OOF metrics, fold wins, tail behavior, generalization gaps, and bootstrap intervals.",
        "leakage_audit": {"tfidf_vocabulary_training_fold_only": True, "idf_training_fold_only": True, "svd_training_fold_only": True, "validation_descriptions_in_transformer_fit": False, "regex_keywords_target_selected": False, "building_te_inner_oof": True, "validation_price_used_for_feature_engineering": False, "validation_price_used_for_model_tuning": False, "duplicate_canonical_identifiers_introduced": False, "many_to_many_raw_join": False, "all_3791_rows_evaluated_per_variant": bool((oof_frame.groupby("variant")["row_id"].nunique() == 3791).all())},
        "runtime_seconds": time.perf_counter() - started,
    }
    after = _protected_snapshot(); results["production_safety"] = {"protected_file_count": len(before), "all_sha256_unchanged": before == after, "before_sha256": before, "after_sha256": after}; results["leakage_audit"]["protected_files_unchanged"] = before == after
    results["artifacts"] = [str(path.relative_to(ROOT)).replace("\\", "/") for path in [EXPERIMENT / "results.json", EXPERIMENT / "model_comparison.csv", EXPERIMENT / "fold_metrics.csv", EXPERIMENT / "oof_predictions.csv", EXPERIMENT / "feature_summary.json", EXPERIMENT / "regex_feature_frequencies.csv", EXPERIMENT / "svd_component_summary.csv", *sorted(FIGURES.glob("*.png"))]]
    with (EXPERIMENT / "results.json").open("w", encoding="utf-8") as handle: json.dump(results, handle, indent=2, default=_json_default)
    print(f"Completed description experiment in {results['runtime_seconds']:.1f}s; best balanced={best_balanced}.", flush=True)


if __name__ == "__main__": main()
