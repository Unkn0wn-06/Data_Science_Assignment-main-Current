"""Inner-OOF M-estimate PPSF encoding required by the final Building Name model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


MISSING_TOKEN = "__MISSING__"
DEFAULT_M_VALUES = (5.0, 10.0, 20.0, 50.0, 100.0)


def normalize_category(values: pd.Series, missing_token: str = MISSING_TOKEN) -> pd.Series:
    return values.astype("string").fillna(missing_token)


class MEstimateTargetEncoder(TransformerMixin, BaseEstimator):
    """Learn smoothed mappings only from the training rows supplied to fit."""

    def __init__(
        self,
        columns: tuple[str, ...],
        m: float = 20.0,
        missing_token: str = MISSING_TOKEN,
    ):
        self.columns = columns
        self.m = m
        self.missing_token = missing_token

    def fit(self, X: pd.DataFrame, y):
        target = np.asarray(y, dtype=float)
        if len(X) != len(target):
            raise ValueError("X and y must have identical row counts.")
        if not np.all(np.isfinite(target)):
            raise ValueError("Target encoding requires finite targets.")
        self.global_mean_ = float(np.mean(target))
        self.mappings_ = {}
        for column in self.columns:
            category = normalize_category(X[column], self.missing_token)
            grouped = pd.DataFrame(
                {"category": category.to_numpy(), "target": target}
            ).groupby("category")["target"].agg(["mean", "count"])
            smoothed = (
                grouped["count"] * grouped["mean"]
                + float(self.m) * self.global_mean_
            ) / (grouped["count"] + float(self.m))
            self.mappings_[column] = smoothed.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "mappings_"):
            raise ValueError("Encoder must be fitted before transform.")
        result = pd.DataFrame(index=X.index)
        for column in self.columns:
            category = normalize_category(X[column], self.missing_token)
            result[f"{column}_te"] = (
                category.map(self.mappings_[column])
                .fillna(self.global_mean_)
                .astype(float)
            )
        return result

    def fit_transform_oof(self, X: pd.DataFrame, y, cv) -> pd.DataFrame:
        """Cross-fit so a row's own target never enters its encoded value."""
        target = np.asarray(y, dtype=float)
        result = pd.DataFrame(
            index=X.index,
            columns=[f"{column}_te" for column in self.columns],
            dtype=float,
        )
        positions = np.arange(len(X))
        for train_position, validation_position in cv.split(positions):
            encoder = MEstimateTargetEncoder(
                self.columns, m=self.m, missing_token=self.missing_token
            ).fit(X.iloc[train_position], target[train_position])
            transformed = encoder.transform(X.iloc[validation_position])
            result.loc[X.index[validation_position], :] = transformed.to_numpy()
        if result.isna().any().any():
            raise AssertionError("OOF target encoding produced missing values.")
        self.fit(X, target)
        return result.astype(float)
