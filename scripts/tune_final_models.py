"""Tune the current four submitted models on frozen Scenario B full-market folds."""

from __future__ import annotations

if __package__:
    from scripts import _bootstrap  # noqa: F401
else:
    import _bootstrap  # noqa: F401

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import ParameterSampler

from src.cleaning.pipeline import PROJECT_ROOT
from src.models.common.features import MODEL_FEATURES
from src.models.final.description_linkage import link_descriptions
from src.models.final.final_evaluation import (
    DATA_PATH,
    EXPECTED_ROWS,
    FINAL_MODELS,
    FOLD_PATH,
    PREDICTOR_COUNTS,
    RAW_PATH,
    load_scenario_b,
    metrics,
)
from src.models.final.model_builders import (
    FINAL_TUNED_PARAMS_PATH,
    MODEL_SCALING_POLICY,
    build_standard_ppsf_estimator,
    fit_position_fold,
    get_final_model_parameters,
)
from src.models.final.position_regex_lightgbm import FINAL_MODEL_NAME
from src.models.final.regex_features import extract_position_features


RANDOM_STATE = 42
RESULTS_DIR = PROJECT_ROOT / "results" / "tuning"
PRE_TUNING_PATH = RESULTS_DIR / "pre_tuning_model_comparison.csv"
OFFICIAL_COMPARISON_PATH = PROJECT_ROOT / "results" / "final_models" / "model_comparison.csv"

SEARCH_SPACES = {
    "Ridge Regression": {
        "alpha": np.logspace(-3, 3, 31).tolist(),
    },
    "Random Forest": {
        "n_estimators": [300, 500, 700, 900, 1200],
        "max_depth": [None, 12, 16, 20, 24, 30],
        "min_samples_split": [2, 4, 6, 8, 10],
        "min_samples_leaf": [1, 2, 3, 4, 5],
        "max_features": [0.5, 0.7, 0.8, 0.9, 1.0, "sqrt"],
        "criterion": ["squared_error", "poisson"],
    },
    "Gradient Boosting": {
        "n_estimators": [100, 150, 200, 250, 300, 400],
        "learning_rate": [0.02, 0.03, 0.05, 0.07, 0.1],
        "max_depth": [2, 3, 4, 5, 6],
        "min_samples_split": [2, 4, 6, 8],
        "min_samples_leaf": [1, 2, 3, 5],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "max_features": [None, 0.7, 0.9, "sqrt"],
        "loss": ["squared_error", "huber"],
    },
    FINAL_MODEL_NAME: {
        "n_estimators": [500, 800, 1000, 1200, 1500, 2000],
        "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
        "num_leaves": [15, 31, 47, 63, 95],
        "max_depth": [-1, 6, 8, 10, 12],
        "min_child_samples": [10, 20, 30, 40, 60],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0.0, 0.05, 0.1, 0.5, 1.0],
        "reg_lambda": [0.0, 0.5, 1.0, 2.0, 5.0],
    },
}
CANDIDATE_COUNTS = {
    "Ridge Regression": 31,
    "Random Forest": 50,
    "Gradient Boosting": 60,
    FINAL_MODEL_NAME: 80,
}
FIXED_PARAMETERS = {
    "Ridge Regression": {},
    "Random Forest": {"bootstrap": True, "random_state": 42, "n_jobs": -1},
    "Gradient Boosting": {"random_state": 42},
    FINAL_MODEL_NAME: {
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
        "objective": "regression",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write_text(path: Path, value: str) -> None:
    """Replace one text artifact only after its complete temporary file is written."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Replace one CSV artifact only after its complete temporary file is written."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _parameter_key(parameters: dict) -> str:
    return json.dumps(_json_ready(parameters), sort_keys=True)


def candidate_parameters(model_name: str) -> list[dict]:
    """Return a deterministic bounded search that includes the current baseline."""
    baseline = get_final_model_parameters(model_name)
    if model_name == "Ridge Regression":
        candidates = [
            {"alpha": float(alpha)}
            for alpha in SEARCH_SPACES[model_name]["alpha"]
        ]
    else:
        requested = CANDIDATE_COUNTS[model_name]
        sampled = list(
            ParameterSampler(
                SEARCH_SPACES[model_name],
                n_iter=requested * 2,
                random_state=RANDOM_STATE,
            )
        )
        candidates = [baseline, *sampled]

    unique: list[dict] = []
    seen: set[str] = set()
    for candidate in candidates:
        complete = {**candidate, **FIXED_PARAMETERS[model_name]}
        if model_name == FINAL_MODEL_NAME:
            complete["subsample_freq"] = (
                1 if float(complete.get("subsample", 1.0)) < 1.0 else 0
            )
        key = _parameter_key(complete)
        if key not in seen:
            seen.add(key)
            unique.append(_json_ready(complete))
        if len(unique) == CANDIDATE_COUNTS[model_name]:
            break
    if len(unique) != CANDIDATE_COUNTS[model_name]:
        raise AssertionError(f"Could not generate enough unique candidates for {model_name}.")
    return unique


def evaluate_candidate(
    model_name: str,
    parameters: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    position: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    p95: float,
) -> dict:
    prediction = np.full(len(y), np.nan)
    coverage = np.zeros(len(y), dtype=int)
    for training, validation in folds:
        if model_name == FINAL_MODEL_NAME:
            fold_prediction = fit_position_fold(
                X.iloc[training],
                y[training],
                X.iloc[validation],
                position.iloc[training],
                position.iloc[validation],
                parameters,
            )
        else:
            fitted = clone(
                build_standard_ppsf_estimator(model_name, parameters)
            ).fit(X.iloc[training], y[training])
            fold_prediction = np.asarray(fitted.predict(X.iloc[validation]), float)
        prediction[validation] = fold_prediction
        coverage[validation] += 1
    if not np.all(coverage == 1) or not np.isfinite(prediction).all():
        raise AssertionError(f"Incomplete tuning OOF coverage for {model_name}.")
    return metrics(y, prediction, PREDICTOR_COUNTS[model_name], p95)


def _comparison_summary(before: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    baseline = before.set_index("Model")
    tuned = selected.set_index("Model")
    rows = []
    for model_name in FINAL_MODELS:
        old = baseline.loc[model_name]
        new = tuned.loc[model_name]
        rows.append(
            {
                "Model": model_name,
                "Pre_Tuning_RMSE_RM": old["RMSE_RM"],
                "Tuned_RMSE_RM": new["CV_RMSE_RM"],
                "RMSE_Improvement_RM": old["RMSE_RM"] - new["CV_RMSE_RM"],
                "RMSE_Improvement_Pct": 100.0 * (old["RMSE_RM"] - new["CV_RMSE_RM"]) / old["RMSE_RM"],
                "Pre_Tuning_MAE_RM": old["MAE_RM"],
                "Tuned_MAE_RM": new["CV_MAE_RM"],
                "MAE_Improvement_RM": old["MAE_RM"] - new["CV_MAE_RM"],
                "MAE_Improvement_Pct": 100.0 * (old["MAE_RM"] - new["CV_MAE_RM"]) / old["MAE_RM"],
                "Pre_Tuning_R2": old["R2"],
                "Tuned_R2": new["CV_R2"],
                "R2_Change": new["CV_R2"] - old["R2"],
                "Pre_Tuning_Adjusted_R2": old["Adjusted_R2"],
                "Tuned_Adjusted_R2": new["CV_Adjusted_R2"],
                "Adjusted_R2_Change": new["CV_Adjusted_R2"] - old["Adjusted_R2"],
            }
        )
    return pd.DataFrame(rows)


def tune_models() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not PRE_TUNING_PATH.is_file():
        atomic_write_text(
            PRE_TUNING_PATH,
            OFFICIAL_COMPARISON_PATH.read_text(encoding="utf-8"),
        )
    before = pd.read_csv(PRE_TUNING_PATH)
    if set(before["Model"]) != set(FINAL_MODELS):
        raise ValueError("Pre-tuning comparison does not contain the current four models.")

    data = pd.read_csv(DATA_PATH).reset_index(drop=True)
    if len(data) != EXPECTED_ROWS or data["listing_id"].nunique() != EXPECTED_ROWS:
        raise AssertionError("Current tuning requires 3,791 unique canonical listings.")
    assignments, folds = load_scenario_b(data)
    descriptions, _ = link_descriptions(RAW_PATH, data["listing_id"])
    position = extract_position_features(descriptions)
    X = data[MODEL_FEATURES]
    y = data["price"].to_numpy(float)
    p95 = float(np.quantile(y, 0.95))
    candidate_rows: list[dict] = []

    for model_name in FINAL_MODELS:
        candidates = candidate_parameters(model_name)
        for number, parameters in enumerate(candidates, 1):
            result = evaluate_candidate(model_name, parameters, X, y, position, folds, p95)
            candidate_rows.append(
                {
                    "Model": model_name,
                    "Candidate_ID": number,
                    "Parameters_JSON": _parameter_key(parameters),
                    "CV_RMSE_RM": result["RMSE_RM"],
                    "CV_MAE_RM": result["MAE_RM"],
                    "CV_R2": result["R2"],
                    "CV_Adjusted_R2": result["Adjusted_R2"],
                }
            )
            print(
                f"Tuned {model_name}: candidate {number}/{len(candidates)} "
                f"RMSE=RM {result['RMSE_RM']:,.2f}",
                flush=True,
            )

    candidates_frame = pd.DataFrame(candidate_rows)
    ranked_parts = []
    selected_parts = []
    for model_name in FINAL_MODELS:
        rows = candidates_frame[candidates_frame["Model"].eq(model_name)].copy()
        rows = rows.sort_values(
            ["CV_RMSE_RM", "CV_MAE_RM", "CV_R2"],
            ascending=[True, True, False],
            kind="stable",
        ).reset_index(drop=True)
        rows["Candidate_Rank"] = np.arange(1, len(rows) + 1)
        rows["Selected"] = rows["Candidate_Rank"].eq(1)
        ranked_parts.append(rows)
        selected_parts.append(rows.iloc[[0]])
    candidates_frame = pd.concat(ranked_parts, ignore_index=True)
    selected = pd.concat(selected_parts, ignore_index=True)

    generated_at = datetime.now(timezone.utc).isoformat()
    config = {
        "metadata": {
            "generated_at": generated_at,
            "random_state": RANDOM_STATE,
            "validation": "Scenario B group-safe 5-fold cross-validation",
            "scoring": "Total-price RMSE primary; MAE secondary; R2 supporting evidence",
            "target_strategy": "Fit PPSF and reconstruct total property price before scoring",
            "dataset_rows": EXPECTED_ROWS,
            "dataset_path": DATA_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "dataset_sha256": sha256(DATA_PATH),
            "fold_assignments_path": FOLD_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "fold_assignments_sha256": sha256(FOLD_PATH),
            "scaling_policy": MODEL_SCALING_POLICY,
        },
        "models": {},
    }
    for _, row in selected.iterrows():
        model_name = str(row["Model"])
        config["models"][model_name] = {
            "parameters": json.loads(row["Parameters_JSON"]),
            "scaling": MODEL_SCALING_POLICY[model_name],
        }
    atomic_write_text(
        FINAL_TUNED_PARAMS_PATH,
        json.dumps(_json_ready(config), indent=2),
    )

    atomic_write_csv(RESULTS_DIR / "tuning_candidates.csv", candidates_frame)
    atomic_write_csv(RESULTS_DIR / "tuned_cv_results.csv", selected)
    summary = _comparison_summary(before, selected)
    atomic_write_csv(RESULTS_DIR / "tuning_summary.csv", summary)
    metadata = {
        "generated_at": generated_at,
        "status": "complete",
        "search_method": {
            "Ridge Regression": "31-value logarithmic grid",
            "Random Forest": "Bounded ParameterSampler",
            "Gradient Boosting": "Bounded ParameterSampler",
            FINAL_MODEL_NAME: "Bounded ParameterSampler",
        },
        "candidate_counts": CANDIDATE_COUNTS,
        "search_spaces": _json_ready(SEARCH_SPACES),
        "validation": "Scenario B group-safe cross-validated tuning results",
        "folds": 5,
        "random_state": RANDOM_STATE,
        "primary_metric": "total property price RMSE in RM",
        "secondary_metric": "total property price MAE in RM",
        "adjusted_r2_optimised": False,
        "dataset_path": DATA_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "dataset_sha256": sha256(DATA_PATH),
        "dataset_rows": EXPECTED_ROWS,
        "fold_assignments_path": FOLD_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "fold_assignments_sha256": sha256(FOLD_PATH),
        "repeat_groups_crossing_folds": int(
            assignments[assignments["is_grouped_repeat"]]
            .groupby("repeat_group_id")["fold"]
            .nunique()
            .gt(1)
            .sum()
        ),
        "scaling_policy": MODEL_SCALING_POLICY,
        "selected_parameters": {
            name: config["models"][name]["parameters"] for name in FINAL_MODELS
        },
        "frozen_configuration_sha256": sha256(FINAL_TUNED_PARAMS_PATH),
        "artifacts": {
            "pre_tuning_model_comparison": PRE_TUNING_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "tuning_candidates": "results/tuning/tuning_candidates.csv",
            "tuned_cv_results": "results/tuning/tuned_cv_results.csv",
            "tuning_summary": "results/tuning/tuning_summary.csv",
            "frozen_configuration": FINAL_TUNED_PARAMS_PATH.relative_to(PROJECT_ROOT).as_posix(),
        },
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
        },
    }
    atomic_write_text(
        RESULTS_DIR / "metadata.json",
        json.dumps(_json_ready(metadata), indent=2),
    )
    return candidates_frame, summary, metadata


def main() -> None:
    _, summary, metadata = tune_models()
    print("\nCURRENT FORMAL TUNING — BEFORE VS AFTER")
    print(summary.to_string(index=False))
    print("\nSELECTED PARAMETERS")
    for model_name, parameters in metadata["selected_parameters"].items():
        print(f"{model_name}: {json.dumps(parameters, sort_keys=True)}")


if __name__ == "__main__":
    main()
