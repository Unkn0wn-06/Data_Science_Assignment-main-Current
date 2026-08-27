"""Export verified trimming artifacts for Streamlit without retraining models."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "experiments" / "upper_tail_trimming"
OUTPUT_DIR = PROJECT_ROOT / "results" / "outlier_trimming"
CANONICAL_PATH = PROJECT_ROOT / "data" / "processed" / "enhanced_city_dataset.csv"
FOLD_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "repeat_group_sensitivity"
    / "scenario_b_fold_assignments.csv"
)
EXPECTED_ROWS = 3_791
TRIM_LEVELS = [0.0, 0.5, 1.0, 2.5, 5.0, 10.0]
PRESENTATION_FILES = (
    "training_only_comparison.csv",
    "trimmed_population_comparison.csv",
    "segment_metrics.csv",
    "distribution_shift.csv",
    "bootstrap_results.csv",
)
RETAINED_CV_FILENAME = "retained_cv_summary.csv"
REQUIRED_COLUMNS = {
    "training_only_comparison.csv": {
        "Model",
        "Trim_Level",
        "Removal_Percent",
        "OOF_Rows",
        "RMSE_RM",
        "MAE_RM",
        "R2",
        "Adjusted_R2",
        "Top5_RMSE_RM",
        "P99_100_RMSE_RM",
        "Top5_Underprediction_Pct",
    },
    "trimmed_population_comparison.csv": {
        "Trim_Level",
        "Removal_Percent",
        "Retained_OOF_Rows",
        "Removed_Evaluation_Rows",
        "Matched_Original_RMSE_RM",
        "Matched_Retrained_RMSE_RM",
        "Matched_RMSE_Gain_RM",
        "Matched_Original_MAE_RM",
        "Matched_Retrained_MAE_RM",
        "Matched_MAE_Gain_RM",
    },
    "segment_metrics.csv": {
        "Model",
        "Trim_Level",
        "Removal_Percent",
        "Segment",
        "RMSE_RM",
        "MAE_RM",
        "Underprediction_Pct",
    },
    "distribution_shift.csv": {
        "Trim_Level",
        "Removal_Percent",
        "After_Row_Count",
        "After_Mean_Price_RM",
        "After_Maximum_Price_RM",
        "After_Skewness",
    },
    "bootstrap_results.csv": {
        "Model",
        "Trim_Level",
        "Removal_Percent",
        "Bootstrap_Samples",
        "RMSE_Difference_RM",
        "RMSE_CI95_Lower_RM",
        "RMSE_CI95_Upper_RM",
        "MAE_Difference_RM",
        "MAE_CI95_Lower_RM",
        "MAE_CI95_Upper_RM",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_source() -> tuple[dict, dict[str, str]]:
    """Validate completed outputs and return metadata plus source hashes."""
    missing_files = [
        name
        for name in (*PRESENTATION_FILES, "oof_predictions.csv", "results.json")
        if not (SOURCE_DIR / name).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(f"Missing trimming experiment outputs: {missing_files}")

    source_result = json.loads(
        (SOURCE_DIR / "results.json").read_text(encoding="utf-8")
    )
    if source_result["canonical_dataset"]["rows"] != EXPECTED_ROWS:
        raise ValueError("Trimming experiment does not use 3,791 canonical rows.")
    if source_result["validation"]["scenario"] != "B":
        raise ValueError("Trimming experiment does not use Scenario B.")
    if source_result["recommended_trim_level"] != "A":
        raise ValueError("Verified trimming recommendation is not the 0% baseline.")
    if source_result["production_model_changed"]:
        raise ValueError("Source metadata unexpectedly reports a production-model change.")
    if not source_result["production_safety"]["all_protected_files_unchanged"]:
        raise ValueError("Source experiment did not preserve protected production files.")

    for filename, required in REQUIRED_COLUMNS.items():
        frame = pd.read_csv(SOURCE_DIR / filename)
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{filename} is missing required columns: {missing}")
        observed_levels = sorted(frame["Removal_Percent"].astype(float).unique())
        if observed_levels != TRIM_LEVELS:
            raise ValueError(
                f"{filename} has unexpected trimming levels: {observed_levels}"
            )

    training = pd.read_csv(SOURCE_DIR / "training_only_comparison.csv")
    primary = training[training["Model"] == "LightGBM + Position Features"]
    if len(primary) != len(TRIM_LEVELS):
        raise ValueError("Primary LightGBM trimming results are incomplete.")
    if not (primary["OOF_Rows"].astype(int) == EXPECTED_ROWS).all():
        raise ValueError("Training-only results do not retain all validation rows.")
    nonzero = primary[primary["Removal_Percent"] > 0]
    if not (
        (nonzero["RMSE_Change_vs_0_RM"] > 0)
        & (nonzero["MAE_Change_vs_0_RM"] > 0)
    ).all():
        raise ValueError("Saved results do not support the verified 0% decision.")

    hashes = {
        name: sha256(SOURCE_DIR / name)
        for name in (*PRESENTATION_FILES, "oof_predictions.csv", "results.json")
    }
    return source_result, hashes


def build_retained_cv_summary() -> pd.DataFrame:
    """Derive retained-population fold counts from IDs and saved Scenario B folds."""
    canonical = pd.read_csv(CANONICAL_PATH, usecols=["listing_id", "price"])
    assignments = pd.read_csv(FOLD_PATH)
    distribution = pd.read_csv(SOURCE_DIR / "distribution_shift.csv")
    experiment_oof = pd.read_csv(
        SOURCE_DIR / "oof_predictions.csv",
        usecols=[
            "Experiment_Type",
            "Model",
            "Trim_Level",
            "listing_id",
            "scenario_b_fold",
        ],
    )

    if len(canonical) != EXPECTED_ROWS or canonical["listing_id"].nunique() != EXPECTED_ROWS:
        raise ValueError("Canonical listing IDs are not unique and complete.")
    if len(assignments) != EXPECTED_ROWS or assignments["listing_id"].nunique() != EXPECTED_ROWS:
        raise ValueError("Scenario B assignments are not unique and complete.")
    if set(assignments["listing_id"]) != set(canonical["listing_id"]):
        raise ValueError("Scenario B assignments do not match canonical listing IDs.")
    if set(assignments["fold"].astype(int)) != set(range(1, 6)):
        raise ValueError("Scenario B assignments must contain folds 1 through 5.")
    repeated = assignments[assignments["is_grouped_repeat"].astype(bool)]
    if repeated.groupby("repeat_group_id")["fold"].nunique().gt(1).any():
        raise ValueError("A Scenario B repeat group crosses validation folds.")

    joined = canonical.merge(
        assignments[["listing_id", "fold"]],
        on="listing_id",
        how="left",
        validate="one_to_one",
    )
    trimmed_oof = experiment_oof[
        experiment_oof["Experiment_Type"].eq("trimmed_population")
        & experiment_oof["Model"].eq("LightGBM + Position Features")
    ]
    records: list[dict] = []
    for _, row in distribution.sort_values("Removal_Percent").iterrows():
        trim_level = str(row["Trim_Level"])
        removal_percent = float(row["Removal_Percent"])
        cutoff = row["Full_Population_Cutoff_RM"]
        retained = joined if removal_percent == 0 else joined[joined["price"] <= cutoff]
        retained_ids = set(retained["listing_id"].astype(int))
        saved_rows = trimmed_oof[trimmed_oof["Trim_Level"].eq(trim_level)]
        if retained_ids != set(saved_rows["listing_id"].astype(int)):
            raise ValueError(
                f"Derived retained IDs differ from saved OOF IDs at {removal_percent:g}%."
            )
        saved_fold_by_id = saved_rows.set_index("listing_id")["scenario_b_fold"]
        assigned_fold_by_id = retained.set_index("listing_id")["fold"]
        if not saved_fold_by_id.sort_index().equals(assigned_fold_by_id.sort_index()):
            raise ValueError(
                f"Saved OOF folds differ from Scenario B assignments at {removal_percent:g}%."
            )

        retained_rows = len(retained)
        if retained_rows != int(row["After_Row_Count"]):
            raise ValueError(f"Retained-row count mismatch at {removal_percent:g}%.")
        for fold in range(1, 6):
            validation_ids = set(
                retained.loc[retained["fold"].astype(int).eq(fold), "listing_id"].astype(int)
            )
            training_ids = retained_ids.difference(validation_ids)
            if training_ids.intersection(validation_ids):
                raise AssertionError("A retained listing appears in both fold partitions.")
            if training_ids.union(validation_ids) != retained_ids:
                raise AssertionError("Fold partitions do not cover the retained population.")
            records.append(
                {
                    "trim_level": f"{removal_percent:g}%",
                    "fold": fold,
                    "original_rows": EXPECTED_ROWS,
                    "retained_rows": retained_rows,
                    "removed_rows": EXPECTED_ROWS - retained_rows,
                    "training_rows": len(training_ids),
                    "validation_rows": len(validation_ids),
                    "retention_percentage": 100.0 * retained_rows / EXPECTED_ROWS,
                }
            )

    summary = pd.DataFrame.from_records(records)
    if len(summary) != len(TRIM_LEVELS) * 5:
        raise AssertionError("Retained CV summary must contain 30 trim/fold rows.")
    if not (
        summary.groupby("trim_level")["validation_rows"].sum()
        == summary.groupby("trim_level")["retained_rows"].first()
    ).all():
        raise AssertionError("Each retained listing must be validated exactly once.")
    if not (
        summary.groupby("trim_level")["training_rows"].sum()
        == 4 * summary.groupby("trim_level")["retained_rows"].first()
    ).all():
        raise AssertionError("Each retained listing must train in exactly four folds.")
    return summary


def build_results() -> dict:
    """Copy only presentation artifacts and write deterministic metadata."""
    source_result, source_hashes = validate_source()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename in PRESENTATION_FILES:
        shutil.copyfile(SOURCE_DIR / filename, OUTPUT_DIR / filename)

    retained_cv = build_retained_cv_summary()
    retained_cv.to_csv(OUTPUT_DIR / RETAINED_CV_FILENAME, index=False)

    copied_hashes = {
        filename: sha256(OUTPUT_DIR / filename) for filename in PRESENTATION_FILES
    }
    if copied_hashes != {
        name: source_hashes[name] for name in PRESENTATION_FILES
    }:
        raise AssertionError("An exported trimming artifact differs from its source.")
    output_hashes = {
        **copied_hashes,
        RETAINED_CV_FILENAME: sha256(OUTPUT_DIR / RETAINED_CV_FILENAME),
    }

    metadata = {
        "canonical_rows": EXPECTED_ROWS,
        "canonical_dataset": CANONICAL_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "canonical_dataset_sha256": sha256(CANONICAL_PATH),
        "validation_method": "Scenario B group-safe 5-fold CV",
        "scenario_b_fold_assignments": FOLD_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "scenario_b_fold_assignments_sha256": sha256(FOLD_PATH),
        "trim_levels_percent": TRIM_LEVELS,
        "recommended_trimming": "0%",
        "recommended_trim_level": "A",
        "decision": "Do not adopt upper-tail trimming",
        "production_model": "LightGBM + Position Features",
        "production_model_changed": False,
        "target": "PPSF",
        "evaluation_unit": "reconstructed total RM price",
        "bootstrap_samples": int(source_result["bootstrap_samples"]),
        "training_only_validation_rows_removed": 0,
        "retained_cv_folds": 5,
        "retained_cv_summary": RETAINED_CV_FILENAME,
        "retained_cv_summary_rows": len(retained_cv),
        "streamlit_retraining": False,
        "source_experiment": "experiments/upper_tail_trimming",
        "source_results_sha256": source_hashes["results.json"],
        "source_artifact_sha256": source_hashes,
        "exported_artifact_sha256": output_hashes,
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    metadata = build_results()
    print(
        "Exported verified outlier/trimming presentation artifacts to "
        f"{OUTPUT_DIR.relative_to(PROJECT_ROOT).as_posix()}."
    )
    print(f"Decision: {metadata['decision']} ({metadata['recommended_trimming']}).")


if __name__ == "__main__":
    main()
