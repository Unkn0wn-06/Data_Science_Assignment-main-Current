"""Reusable target-free geometry and leakage-safe spatial PPSF utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


EARTH_RADIUS_KM = 6371.0088
LATITUDE_ALIASES = ("latitude", "lat")
LONGITUDE_ALIASES = ("longitude", "lon", "lng")
GEOMETRY_K = (3, 5, 10, 20)
RADIUS_KM = (1.0, 3.0, 5.0, 10.0)


def find_coordinate_columns(columns) -> tuple[str | None, str | None]:
    """Find explicit coordinate columns without inferring or geocoding them."""
    lookup = {str(column).lower(): str(column) for column in columns}
    latitude = next((lookup[name] for name in LATITUDE_ALIASES if name in lookup), None)
    longitude = next(
        (lookup[name] for name in LONGITUDE_ALIASES if name in lookup), None
    )
    return latitude, longitude


def validate_coordinates(
    frame: pd.DataFrame, latitude_column: str, longitude_column: str
) -> dict:
    """Report mathematical validity and suspicious non-Malaysian positions."""
    latitude = pd.to_numeric(frame[latitude_column], errors="coerce")
    longitude = pd.to_numeric(frame[longitude_column], errors="coerce")
    missing = latitude.isna() | longitude.isna()
    valid = (
        ~missing
        & latitude.between(-90.0, 90.0)
        & longitude.between(-180.0, 180.0)
    )
    suspicious_malaysia = valid & ~(
        latitude.between(0.5, 7.5) & longitude.between(99.0, 120.0)
    )
    return {
        "valid_mask": valid.to_numpy(bool),
        "missing_mask": missing.to_numpy(bool),
        "invalid_mask": (~missing & ~valid).to_numpy(bool),
        "suspicious_malaysia_mask": suspicious_malaysia.to_numpy(bool),
        "valid_count": int(valid.sum()),
        "missing_count": int(missing.sum()),
        "invalid_count": int((~missing & ~valid).sum()),
        "suspicious_malaysia_count": int(suspicious_malaysia.sum()),
        "coverage_percent": float(valid.mean() * 100.0),
    }


def haversine_km(lat1, lon1, lat2, lon2):
    """Return great-circle distance in kilometers for scalar or array inputs."""
    lat1, lon1, lat2, lon2 = np.broadcast_arrays(
        np.asarray(lat1, dtype=float),
        np.asarray(lon1, dtype=float),
        np.asarray(lat2, dtype=float),
        np.asarray(lon2, dtype=float),
    )
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(
        np.radians, (lat1, lon1, lat2, lon2)
    )
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    a = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2.0) ** 2
    )
    return EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _coordinate_array(frame, latitude_column, longitude_column):
    latitude = pd.to_numeric(frame[latitude_column], errors="coerce").to_numpy(float)
    longitude = pd.to_numeric(frame[longitude_column], errors="coerce").to_numpy(float)
    valid = (
        np.isfinite(latitude)
        & np.isfinite(longitude)
        & (latitude >= -90)
        & (latitude <= 90)
        & (longitude >= -180)
        & (longitude <= 180)
    )
    coordinates = np.column_stack([latitude, longitude])
    return coordinates, valid


class SpatialGeometryFeatures:
    """Create target-free BallTree geometry features from training coordinates."""

    def __init__(self, latitude_column="latitude", longitude_column="longitude"):
        self.latitude_column = latitude_column
        self.longitude_column = longitude_column

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "nearest_property_distance_km",
            *[f"knn_{k}_mean_distance_km" for k in GEOMETRY_K],
            "knn_5_max_distance_km",
            "knn_10_max_distance_km",
            *[f"properties_within_{int(radius)}km" for radius in RADIUS_KM],
        ]

    def fit(self, X: pd.DataFrame):
        coordinates, valid = _coordinate_array(
            X, self.latitude_column, self.longitude_column
        )
        self.training_coordinates_ = coordinates[valid]
        self.training_valid_positions_ = np.flatnonzero(valid)
        self.tree_ = (
            None
            if len(self.training_coordinates_) == 0
            else BallTree(np.radians(self.training_coordinates_), metric="haversine")
        )
        return self

    def _row_features(self, distances_km: np.ndarray) -> list[float]:
        if len(distances_km) == 0:
            return [np.nan] * 7 + [0.0] * len(RADIUS_KM)
        values = [float(distances_km[0])]
        for k in GEOMETRY_K:
            values.append(float(np.mean(distances_km[: min(k, len(distances_km))])))
        for k in (5, 10):
            values.append(float(np.max(distances_km[: min(k, len(distances_km))])))
        values.extend(float(np.sum(distances_km <= radius)) for radius in RADIUS_KM)
        return values

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        coordinates, valid = _coordinate_array(
            X, self.latitude_column, self.longitude_column
        )
        output = pd.DataFrame(index=X.index, columns=self.feature_names(), dtype=float)
        if self.tree_ is None:
            return output
        query_count = min(max(GEOMETRY_K), len(self.training_coordinates_))
        for position in np.flatnonzero(valid):
            angular, _ = self.tree_.query(
                np.radians(coordinates[[position]]), k=query_count
            )
            output.iloc[position] = self._row_features(angular[0] * EARTH_RADIUS_KM)
        return output

    def fit_transform_training(self, X: pd.DataFrame) -> pd.DataFrame:
        """Exclude the exact training row while retaining co-located properties."""
        self.fit(X)
        output = pd.DataFrame(index=X.index, columns=self.feature_names(), dtype=float)
        if self.tree_ is None:
            return output
        query_count = min(max(GEOMETRY_K) + 1, len(self.training_coordinates_))
        for tree_position, original_position in enumerate(self.training_valid_positions_):
            angular, indices = self.tree_.query(
                np.radians(self.training_coordinates_[[tree_position]]), k=query_count
            )
            keep = indices[0] != tree_position
            distances = angular[0][keep] * EARTH_RADIUS_KM
            output.iloc[original_position] = self._row_features(distances)
        return output


class SpatialPPSFNeighborEncoder:
    """Build priced-neighbor features from supplied training targets only."""

    def __init__(
        self,
        latitude_column="latitude",
        longitude_column="longitude",
        epsilon_km=1e-3,
    ):
        self.latitude_column = latitude_column
        self.longitude_column = longitude_column
        self.epsilon_km = epsilon_km

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "knn_5_median_ppsf",
            "knn_10_median_ppsf",
            "knn_20_median_ppsf",
            "knn_5_mean_ppsf",
            "knn_10_mean_ppsf",
            "knn_10_weighted_ppsf",
        ]

    def fit(self, X: pd.DataFrame, y):
        coordinates, valid = _coordinate_array(
            X, self.latitude_column, self.longitude_column
        )
        total_price = np.asarray(y, dtype=float)
        size = pd.to_numeric(X["property_size_sqft"], errors="coerce").to_numpy(float)
        ppsf = total_price / size
        valid &= np.isfinite(ppsf) & (ppsf > 0)
        usable_ppsf = np.isfinite(ppsf) & (ppsf > 0)
        if not np.any(usable_ppsf):
            raise ValueError("At least one finite, positive training PPSF is required.")
        self.global_training_median_ppsf_ = float(np.median(ppsf[usable_ppsf]))
        self.training_coordinates_ = coordinates[valid]
        self.training_ppsf_ = ppsf[valid]
        self.training_valid_positions_ = np.flatnonzero(valid)
        self.tree_ = (
            None
            if len(self.training_coordinates_) == 0
            else BallTree(np.radians(self.training_coordinates_), metric="haversine")
        )
        return self

    def _features(self, distances, indices) -> list[float]:
        if len(indices) == 0:
            return [self.global_training_median_ppsf_] * len(self.feature_names())
        values = self.training_ppsf_[indices]
        result = []
        for k in (5, 10, 20):
            result.append(float(np.median(values[: min(k, len(values))])))
        for k in (5, 10):
            result.append(float(np.mean(values[: min(k, len(values))])))
        count = min(10, len(values))
        weights = 1.0 / (distances[:count] + float(self.epsilon_km))
        result.append(float(np.sum(weights * values[:count]) / np.sum(weights)))
        return result

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        coordinates, valid = _coordinate_array(
            X, self.latitude_column, self.longitude_column
        )
        output = pd.DataFrame(index=X.index, columns=self.feature_names(), dtype=float)
        fallback = np.full(len(self.feature_names()), self.global_training_median_ppsf_)
        fallback_count = 0
        for position in range(len(X)):
            if not valid[position] or self.tree_ is None:
                output.iloc[position] = fallback
                fallback_count += 1
                continue
            count = min(20, len(self.training_coordinates_))
            angular, indices = self.tree_.query(
                np.radians(coordinates[[position]]), k=count
            )
            output.iloc[position] = self._features(
                angular[0] * EARTH_RADIUS_KM, indices[0]
            )
        self.last_fallback_count_ = fallback_count
        return output

    def fit_transform_oof(self, X: pd.DataFrame, y, cv) -> pd.DataFrame:
        output = pd.DataFrame(index=X.index, columns=self.feature_names(), dtype=float)
        target = np.asarray(y, dtype=float)
        positions = np.arange(len(X))
        for train_position, validation_position in cv.split(positions):
            encoder = SpatialPPSFNeighborEncoder(
                self.latitude_column, self.longitude_column, self.epsilon_km
            ).fit(X.iloc[train_position], target[train_position])
            transformed = encoder.transform(X.iloc[validation_position])
            output.loc[X.index[validation_position], :] = transformed.to_numpy()
        return output.astype(float)

    def transform_training_excluding_self(self, X: pd.DataFrame) -> pd.DataFrame:
        """Diagnostic helper proving a fitted training row cannot price itself."""
        coordinates, valid = _coordinate_array(
            X, self.latitude_column, self.longitude_column
        )
        output = pd.DataFrame(index=X.index, columns=self.feature_names(), dtype=float)
        fallback = np.full(len(self.feature_names()), self.global_training_median_ppsf_)
        tree_position_by_original = {
            original: tree for tree, original in enumerate(self.training_valid_positions_)
        }
        for original_position in range(len(X)):
            if original_position not in tree_position_by_original or self.tree_ is None:
                output.iloc[original_position] = fallback
                continue
            self_tree_position = tree_position_by_original[original_position]
            count = min(21, len(self.training_coordinates_))
            angular, indices = self.tree_.query(
                np.radians(coordinates[[original_position]]), k=count
            )
            keep = indices[0] != self_tree_position
            kept_indices = indices[0][keep][:20]
            distances = angular[0][keep][:20] * EARTH_RADIUS_KM
            output.iloc[original_position] = self._features(distances, kept_indices)
        return output
