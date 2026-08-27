"""Compatibility helper for loading the active final model comparison."""

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_COMPARISON_PATH = PROJECT_ROOT / "results" / "final_models" / "model_comparison.json"


def load_enhanced_comparison_table(
    path: Path = FINAL_COMPARISON_PATH,
) -> pd.DataFrame:
    """Return the four final Scenario B model metrics in RMSE order."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (
        pd.DataFrame(payload["models"])[
            ["Model", "R2", "Adjusted_R2", "MAE_RM", "RMSE_RM"]
        ]
        .rename(
            columns={
                "R2": "R²",
                "Adjusted_R2": "Adjusted R²",
                "MAE_RM": "MAE (RM)",
                "RMSE_RM": "RMSE (RM)",
            }
        )
        .sort_values("RMSE (RM)")
        .reset_index(drop=True)
    )
