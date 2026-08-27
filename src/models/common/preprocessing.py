"""Leakage-safe production and enhanced feature preprocessors."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
    TargetEncoder,
)

from src.models.common.features import (
    PRODUCTION_CATEGORICAL_FEATURES,
    PRODUCTION_NUMERICAL_FEATURES,
)


def finite_numeric_values(values):
    """Coerce numeric inputs to float64 and convert infinities to missing values."""
    numeric = pd.DataFrame(values).apply(pd.to_numeric, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)


def make_preprocessor(
    scale_numeric: bool,
    numerical_features: list[str] = PRODUCTION_NUMERICAL_FEATURES,
    categorical_features: list[str] = PRODUCTION_CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    """Build the exact imputation, optional scaling, and one-hot workflow."""
    numeric_steps = [
        (
            "finite_values",
            FunctionTransformer(
                finite_numeric_values,
                validate=False,
                feature_names_out="one-to-one",
            ),
        ),
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numerical_features),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def make_target_encoding_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """Build the established fold-fitted target encoder for enhanced experiments."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numerical_features),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "target_encoder",
                            # sklearn 1.9 accepts the splitter object and therefore
                            # preserves the frozen shuffled, seeded inner folds.
                            TargetEncoder(
                                target_type="continuous",
                                smooth="auto",
                                cv=KFold(n_splits=5, shuffle=True, random_state=42),
                            ),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
