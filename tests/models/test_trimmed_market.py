"""Validate experimental deployment trimming without changing final artifacts."""

from __future__ import annotations

import hashlib
import inspect
import json
import unittest

import numpy as np
import pandas as pd

from prototype.app import (
    _load_trimmed_deployment_model_cached,
    load_trimmed_deployment_model,
)
from src.cleaning.pipeline import PROJECT_ROOT
from src.models.common.features import MODEL_FEATURES
from src.models.final.model_builders import (
    FINAL_TUNED_PARAMS_PATH,
    build_position_lightgbm,
    final_tuned_params_sha256,
)
from src.models.final.position_regex_lightgbm import predict_total_price
from src.models.final.trimmed_market import (
    fit_trimmed_market_model,
    get_trim_market_metadata,
    get_trimmed_population,
)


DATA_PATH = PROJECT_ROOT / "data" / "processed" / "enhanced_city_dataset.csv"
OOF_PATH = PROJECT_ROOT / "results" / "final_models" / "oof_predictions.csv"
COMPARISON_PATH = PROJECT_ROOT / "results" / "final_models" / "model_comparison.csv"


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TrimmedMarketDeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(DATA_PATH).reset_index(drop=True)
        cls.data_hash = sha256(DATA_PATH)
        cls.trimmed_model = fit_trimmed_market_model(5.0)

    def test_official_artifacts_remain_exact(self):
        self.assertEqual(
            "4a295007fc5fdf6def33a612797606dfb60d811e7b19ade930466033a1fd66cf",
            sha256(DATA_PATH),
        )
        metadata = json.loads(
            (PROJECT_ROOT / "results" / "final_models" / "metadata.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(final_tuned_params_sha256(), metadata["tuned_configuration_sha256"])
        self.assertTrue(FINAL_TUNED_PARAMS_PATH.is_file())
        self.assertGreater(OOF_PATH.stat().st_size, 0)
        self.assertGreater(COMPARISON_PATH.stat().st_size, 0)

    def test_five_percent_population_matches_saved_experiment_without_mutation(self):
        original = self.data.copy(deep=True)
        retained = get_trimmed_population(self.data, 5.0)
        metadata = get_trim_market_metadata(5.0)
        self.assertEqual(3_791, len(self.data))
        self.assertEqual(3_601, len(retained))
        self.assertEqual(3_601, metadata["retained_rows"])
        self.assertEqual(190, metadata["removed_rows"])
        pd.testing.assert_frame_equal(original, self.data)
        self.assertEqual(self.data_hash, sha256(DATA_PATH))

    def test_trimmed_fit_uses_only_retained_rows_and_same_architecture(self):
        self.assertEqual(3_601, self.trimmed_model.training_rows_)
        self.assertEqual(3_791, self.trimmed_model.original_training_rows_)
        self.assertEqual(5.0, self.trimmed_model.trim_level_)
        estimator, numerical, categorical = build_position_lightgbm()
        expected_params = estimator.named_steps["model"].get_params()
        actual_params = self.trimmed_model.estimator_.named_steps["model"].get_params()
        self.assertEqual(expected_params, actual_params)
        self.assertEqual(list(numerical) + list(categorical), self.trimmed_model.feature_names_)

    def test_new_property_is_predicted_not_trimmed(self):
        values = self.data.iloc[0][MODEL_FEATURES].to_dict()
        values["property_size_sqft"] = 20_000.0
        result = predict_total_price(
            self.trimmed_model,
            values,
            "Premium top floor condominium with a large balcony",
            self.trimmed_model.description_length_median_,
        )
        self.assertTrue(np.isfinite(result["total_price_RM"]))
        self.assertGreater(result["total_price_RM"], 0)

    def test_trimming_uses_known_training_price_only(self):
        self.assertEqual(
            ["data", "trim_level", "distribution_path"],
            list(inspect.signature(get_trimmed_population).parameters),
        )
        source = inspect.getsource(get_trimmed_population)
        self.assertIn('["price"]', source)
        self.assertNotIn("predict", source)

    def test_streamlit_cache_key_includes_tuned_configuration(self):
        self.assertEqual(
            ["trim_level"],
            list(inspect.signature(load_trimmed_deployment_model).parameters),
        )
        source = inspect.getsource(load_trimmed_deployment_model)
        self.assertIn("final_tuned_params_sha256", source)
        cached_source = inspect.getsource(_load_trimmed_deployment_model_cached)
        self.assertIn("@st.cache_resource", cached_source)
        self.assertEqual(
            ["trim_level", "model_config_hash"],
            list(inspect.signature(_load_trimmed_deployment_model_cached).parameters),
        )

    def test_active_python_imports_do_not_reference_archive(self):
        active_roots = (PROJECT_ROOT / "src", PROJECT_ROOT / "prototype")
        for root in active_roots:
            for path in root.rglob("*.py"):
                source = path.read_text(encoding="utf-8")
                self.assertNotIn("archive/", source, str(path))
                self.assertNotIn("archive\\", source, str(path))


if __name__ == "__main__":
    unittest.main()
