"""Explicit inner-OOF M-estimate PPSF and training-only count encoders."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


MISSING_TOKEN = "__MISSING__"
DEFAULT_M_VALUES = (5.0, 10.0, 20.0, 50.0, 100.0)
COUNT_NAMES = {
    "building_name": "building_count",
    "developer": "developer_count",
    "city": "city_count",
}


def normalize_category(values: pd.Series, missing_token: str = MISSING_TOKEN) -> pd.Series:
    return values.astype("string").fillna(missing_token)


class MEstimateTargetEncoder(TransformerMixin, BaseEstimator):
    """Learn smoothed mappings only from the X and PPSF supplied to fit."""

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


class CategoryCountEncoder(TransformerMixin, BaseEstimator):
    """Training-partition category counts with zero fallback for unseen values."""

    def __init__(
        self,
        columns: tuple[str, ...],
        add_log_count: bool = True,
        missing_token: str = MISSING_TOKEN,
    ):
        self.columns = columns
        self.add_log_count = add_log_count
        self.missing_token = missing_token

    def fit(self, X: pd.DataFrame, y=None):
        self.count_mappings_ = {
            column: normalize_category(X[column], self.missing_token)
            .value_counts(dropna=False)
            .astype(float)
            .to_dict()
            for column in self.columns
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "count_mappings_"):
            raise ValueError("Encoder must be fitted before transform.")
        result = pd.DataFrame(index=X.index)
        for column in self.columns:
            name = COUNT_NAMES.get(column, f"{column}_count")
            count = (
                normalize_category(X[column], self.missing_token)
                .map(self.count_mappings_[column])
                .fillna(0.0)
                .astype(float)
            )
            result[name] = count
            if self.add_log_count:
                result[f"log_{name}"] = np.log1p(count)
        return result
