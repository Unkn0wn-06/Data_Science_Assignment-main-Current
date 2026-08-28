"""Validate the source-backed dropdown EDA registry and chart builders."""

from __future__ import annotations

import hashlib
import inspect
import unittest

import numpy as np

from prototype import eda_page
from prototype.eda_page import (
    CANONICAL_PATH,
    EDA_VISUALIZATIONS,
    PREPARED_PATH,
    RAW_PATH,
    dataset_overview_frame,
    descriptive_statistics_frame,
    load_eda_sources,
    render_eda_page,
)


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExploratoryDataAnalysisPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.prepared, cls.canonical = load_eda_sources()

    def test_registry_contains_six_categories_and_twenty_real_visualizations(self):
        self.assertEqual(
            [
                "Price & Location",
                "Property Characteristics",
                "Size & Price",
                "Tenure & Development",
                "Price per Square Foot",
                "Data Quality & Preparation",
            ],
            list(EDA_VISUALIZATIONS),
        )
        self.assertEqual(
            [3, 5, 2, 4, 2, 4],
            [len(visualizations) for visualizations in EDA_VISUALIZATIONS.values()],
        )
        self.assertEqual(20, sum(map(len, EDA_VISUALIZATIONS.values())))

    def test_overview_and_descriptive_tables_are_source_backed(self):
        overview = dataset_overview_frame(self.raw, self.prepared, self.canonical)
        self.assertEqual(3, len(overview))
        self.assertEqual([4_000, 3_791, 3_791], overview["Rows"].tolist())
        self.assertEqual([32, 22, 34], overview["Columns"].tolist())
        statistics = descriptive_statistics_frame(self.canonical)
        self.assertEqual(6, len(statistics))
        self.assertTrue(
            np.isfinite(
                statistics.drop(columns="Feature").to_numpy(dtype=float)
            ).all()
        )

    def test_every_registered_builder_returns_its_named_plotly_chart(self):
        before = {path: sha256(path) for path in (RAW_PATH, PREPARED_PATH, CANONICAL_PATH)}
        for category, visualizations in EDA_VISUALIZATIONS.items():
            for chart_name, builder in visualizations.items():
                with self.subTest(category=category, chart=chart_name):
                    figure = builder(self.raw, self.prepared, self.canonical)
                    self.assertEqual(chart_name, figure.layout.title.text)
                    self.assertGreaterEqual(int(figure.layout.height), 550)
                    self.assertGreater(len(figure.data), 0)
        after = {path: sha256(path) for path in (RAW_PATH, PREPARED_PATH, CANONICAL_PATH)}
        self.assertEqual(before, after)

    def test_eda_page_has_unique_dropdown_keys_and_no_fit_or_write_path(self):
        renderer_source = inspect.getsource(render_eda_page)
        module_source = inspect.getsource(eda_page)
        self.assertIn('key="eda_category"', renderer_source)
        self.assertIn('key="eda_visualization"', renderer_source)
        self.assertIn("EDA_VISUALIZATIONS[category][chart_name]", renderer_source)
        self.assertEqual(1, renderer_source.count("st.plotly_chart"))
        for forbidden in (".fit(", "fit_", "to_csv", "write_text", "run_experiment"):
            self.assertNotIn(forbidden, module_source)


if __name__ == "__main__":
    unittest.main()
