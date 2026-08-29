"""Validate and print the synchronized tuned-model and trimming evidence."""

from __future__ import annotations

if __package__:
    from scripts import _bootstrap  # noqa: F401
else:
    import _bootstrap  # noqa: F401

import json

import numpy as np
import pandas as pd

from src.cleaning.pipeline import PROJECT_ROOT
from src.models.final.final_evaluation import FINAL_MODELS, PREDICTOR_COUNTS
from src.models.final.model_builders import final_tuned_params_sha256


FINAL_DIR = PROJECT_ROOT / "results" / "final_models"
TRIM_DIR = PROJECT_ROOT / "results" / "outlier_trimming"
TUNING_DIR = PROJECT_ROOT / "results" / "tuning"
METRICS = ["RMSE_RM", "MAE_RM", "R2", "Adjusted_R2"]
LEVELS = ["0%", "0.5%", "1%", "2.5%", "5%", "10%"]


def validate() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raise on any mismatch and return the two requested reporting tables."""
    official = pd.read_csv(FINAL_DIR / "model_comparison.csv").set_index("Model")
    summary = pd.read_csv(
        TRIM_DIR / "all_models_trimmed_market_summary.csv"
    )
    oof = pd.read_csv(TRIM_DIR / "all_models_trimmed_market_oof.csv")
    tuning = pd.read_csv(TUNING_DIR / "tuning_summary.csv")

    if set(official.index) != set(FINAL_MODELS):
        raise AssertionError("Official results do not contain exactly the final four models.")
    if len(summary) != 24 or summary.duplicated(["Model", "Trim_Level"]).any():
        raise AssertionError("Trimming summary must contain 24 unique model/scope rows.")
    for model_name in FINAL_MODELS:
        observed = summary.loc[summary["Model"].eq(model_name), "Trim_Level"].tolist()
        if observed != LEVELS:
            raise AssertionError(f"Trimming levels are incomplete for {model_name}.")

    zero = summary.loc[summary["Trim_Level"].eq("0%")].set_index("Model")
    for metric, tolerance in zip(METRICS, (1e-6, 1e-6, 1e-12, 1e-12)):
        np.testing.assert_allclose(
            zero.loc[list(FINAL_MODELS), metric],
            official.loc[list(FINAL_MODELS), metric],
            rtol=1e-10,
            atol=tolerance,
        )

    if oof.duplicated(["Model", "Trim_Level", "listing_id"]).any():
        raise AssertionError("Restricted-market OOF predictions contain duplicate rows.")
    if not np.isfinite(oof[["actual_price_RM", "predicted_price_RM"]]).all().all():
        raise AssertionError("Restricted-market OOF predictions contain NaN or infinity.")
    indexed_summary = summary.set_index(["Model", "Trim_Level"])
    for key, rows in oof.groupby(["Model", "Trim_Level"], sort=False):
        saved = indexed_summary.loc[key]
        actual = rows["actual_price_RM"].to_numpy(float)
        predicted = rows["predicted_price_RM"].to_numpy(float)
        residual = predicted - actual
        r2 = 1.0 - np.square(residual).sum() / np.square(actual - actual.mean()).sum()
        adjusted = 1.0 - (1.0 - r2) * (len(rows) - 1) / (
            len(rows) - PREDICTOR_COUNTS[key[0]] - 1
        )
        recomputed = [
            np.sqrt(np.mean(np.square(residual))),
            np.mean(np.abs(residual)),
            r2,
            adjusted,
        ]
        for observed, expected, tolerance in zip(
            recomputed,
            saved[METRICS].to_numpy(float),
            (1e-6, 1e-6, 1e-12, 1e-12),
        ):
            np.testing.assert_allclose(
                observed,
                expected,
                rtol=1e-10,
                atol=tolerance,
                err_msg=f"Metric mismatch for {key}",
            )
        if len(rows) != int(saved["Retained_Rows"]):
            raise AssertionError(f"OOF row-count mismatch for {key}.")

    for path in (
        FINAL_DIR / "metadata.json",
        TRIM_DIR / "all_models_trimmed_market_metadata.json",
    ):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        saved_hash = metadata.get(
            "tuned_configuration_sha256", metadata.get("tuned_config_sha256")
        )
        if saved_hash != final_tuned_params_sha256():
            raise AssertionError(f"Tuned configuration fingerprint mismatch in {path}.")

    return tuning, summary[
        ["Model", "Trim_Level", "Retained_Rows", *METRICS]
    ]


def main() -> None:
    tuning, trimming = validate()
    print("\nFINAL BEFORE/AFTER TUNING")
    print(tuning.to_string(index=False))
    print("\nNEW FROZEN-CONFIGURATION TRIMMING RESULTS")
    print(trimming.to_string(index=False))
    print("\nValidation passed: official, trimming, OOF, and config hashes agree.")


if __name__ == "__main__":
    main()
