"""Validate the source-backed, report-faithful dropdown EDA registry."""

from __future__ import annotations

import hashlib
import inspect
import unittest

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from prototype import eda_page
from prototype.eda_page import (
    BEDROOM_SIZE_LABELS,
    CANONICAL_PATH,
    COMPLETION_LABELS,
    CONDO_SIZE_LABELS,
    CURRENT_PREPARED_PATH,
    EDA_VISUALIZATIONS,
    MAJOR_PROPERTY_TYPES,
    RAW_PATH,
    dataset_overview_frame,
    descriptive_statistics_frame,
    load_eda_sources,
    render_eda_page,
)


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def displayed_title(figure: Figure) -> str:
    if figure._suptitle is not None:
        return figure._suptitle.get_text()
    return next(axis.get_title() for axis in figure.axes if axis.get_title())


REPORT_TITLES = [
    "Average Property Listing Price by State",
    "Mean and Median Condominium Price by State",
    "Listing Price Distribution by Property Type",
    "Median Condominium Price by Bedroom Count",
    "Median Condominium Price by Bedroom Count and Property Size",
    "Condominium Price Distribution by Bathroom Count",
    "Property Size vs Listing Price",
    "Median Condominium Price Across Property Size Groups",
    "Cumulative Distribution of Listing Prices by Tenure",
    "Median Condo Price by State and Tenure Type",
    "Median Condominium Price by Completion Year",
    "Median Condominium Price by State and Completion Period",
    "Listing Price Distribution by Land Title",
    "Condominium Price by Floor Range",
    "Condominium Price Density by Parking Allocation",
    "Median Price per Square Foot by State and Property Type",
    "Price per Square Foot Distribution by Property Type",
]

REMOVED_PREPARATION_TITLES = {
    "Missing Values by Feature Before Data Preparation",
    "Property Size Distribution Before Outlier Treatment",
    "Property Size Distribution After Outlier Treatment",
    "Dataset Size Across Data Preparation Stages",
}


class ExploratoryDataAnalysisPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.prepared, cls.canonical = load_eda_sources()

    def build(self, title: str) -> Figure:
        for visualizations in EDA_VISUALIZATIONS.values():
            if title in visualizations:
                figure = visualizations[title](self.raw, self.prepared, self.canonical)
                self.addCleanup(plt.close, figure)
                return figure
        self.fail(f"Unknown visualization: {title}")

    def test_registry_matches_final_seventeen_report_visualizations(self):
        self.assertEqual(
            [
                "Price & Location",
                "Property Characteristics",
                "Size & Price",
                "Tenure & Development",
                "Price per Square Foot",
            ],
            list(EDA_VISUALIZATIONS),
        )
        self.assertEqual(
            [3, 6, 2, 4, 2],
            [len(group) for group in EDA_VISUALIZATIONS.values()],
        )
        registered = [
            title for group in EDA_VISUALIZATIONS.values() for title in group
        ]
        self.assertEqual(17, len(registered))
        self.assertCountEqual(REPORT_TITLES, registered)
        self.assertIn(
            "Listing Price Distribution by Land Title",
            EDA_VISUALIZATIONS["Property Characteristics"],
        )
        self.assertTrue(REMOVED_PREPARATION_TITLES.isdisjoint(registered))

    def test_overview_and_descriptive_tables_are_source_backed(self):
        overview = dataset_overview_frame(self.raw, self.prepared, self.canonical)
        self.assertEqual([4_000, 3_791, 3_791], overview["Rows"].tolist())
        self.assertEqual([32, 22, 34], overview["Columns"].tolist())
        self.assertEqual(
            [
                "data/raw/houses.csv",
                "data/processed/production_prepared_dataset.csv",
                "data/processed/enhanced_city_dataset.csv",
            ],
            overview["Source"].tolist(),
        )
        statistics = descriptive_statistics_frame(self.canonical)
        self.assertEqual(6, len(statistics))
        self.assertTrue(
            np.isfinite(statistics.drop(columns="Feature").to_numpy(dtype=float)).all()
        )

    def test_every_builder_returns_a_large_matplotlib_figure_with_report_title(self):
        protected = (RAW_PATH, CURRENT_PREPARED_PATH, CANONICAL_PATH)
        before = {path: sha256(path) for path in protected}
        for category, visualizations in EDA_VISUALIZATIONS.items():
            for chart_name, builder in visualizations.items():
                with self.subTest(category=category, chart=chart_name):
                    figure = builder(self.raw, self.prepared, self.canonical)
                    try:
                        self.assertIsInstance(figure, Figure)
                        self.assertGreaterEqual(figure.get_figwidth(), 9)
                        title = displayed_title(figure)
                        if chart_name == "Property Size vs Listing Price":
                            self.assertRegex(
                                title,
                                r"^Property Size vs Listing Price \(r = -?\d+\.\d{2}\)$",
                            )
                        else:
                            self.assertEqual(chart_name, title)
                    finally:
                        plt.close(figure)
        after = {path: sha256(path) for path in protected}
        self.assertEqual(before, after)

    def test_special_report_chart_types_and_scales_are_preserved(self):
        expectations = {
            "Listing Price Distribution by Property Type": "faceted_histogram",
            "Median Condominium Price by Bedroom Count and Property Size": "heatmap",
            "Property Size vs Listing Price": "hexbin",
            "Cumulative Distribution of Listing Prices by Tenure": "ecdf",
            "Median Condominium Price by State and Completion Period": "heatmap",
            "Listing Price Distribution by Land Title": "boxenplot",
            "Condominium Price Density by Parking Allocation": "kde",
            "Median Price per Square Foot by State and Property Type": "heatmap",
            "Price per Square Foot Distribution by Property Type": "violin",
        }
        for title, chart_type in expectations.items():
            with self.subTest(title=title):
                self.assertEqual(
                    chart_type, self.build(title)._eda_metadata["chart_type"]
                )

        facets = self.build("Listing Price Distribution by Property Type")
        self.assertEqual(MAJOR_PROPERTY_TYPES, facets._eda_metadata["property_types"])
        self.assertEqual(50, facets._eda_metadata["bins"])
        self.assertTrue(facets._eda_metadata["kde"])
        self.assertTrue(all(axis.get_xscale() == "log" for axis in facets.axes))
        tenure = self.build("Cumulative Distribution of Listing Prices by Tenure")
        self.assertEqual("log", tenure.axes[0].get_xscale())

        land_title = self.build("Listing Price Distribution by Land Title")
        self.assertEqual(
            ["Bumi Lot", "Non Bumi Lot"], land_title._eda_metadata["land_titles"]
        )
        expected_land_rows = self.canonical["land_title"].isin(
            ["Bumi Lot", "Non Bumi Lot"]
        ).sum()
        self.assertEqual(expected_land_rows, land_title._eda_metadata["rows_used"])
        self.assertEqual("Land Title", land_title.axes[0].get_xlabel())
        self.assertEqual("Listing Price (RM)", land_title.axes[0].get_ylabel())

    def test_correlation_filters_bins_and_groupings_match_the_report(self):
        size_price = self.build("Property Size vs Listing Price")
        metadata = size_price._eda_metadata
        expected = self.canonical[self.canonical["property_size_sqft"].between(300, 5000)]
        expected_r = expected[["property_size_sqft", "price"]].corr().iloc[0, 1]
        self.assertAlmostEqual(float(expected_r), metadata["correlation"])
        self.assertEqual("pearson", metadata["correlation_method"])
        self.assertEqual((35, 1), (metadata["gridsize"], metadata["mincnt"]))

        bedroom = self.build(
            "Median Condominium Price by Bedroom Count and Property Size"
        )._eda_metadata
        self.assertEqual(BEDROOM_SIZE_LABELS, bedroom["size_bins"])
        self.assertEqual((1, 5), bedroom["bedroom_range"])
        self.assertEqual((300, 2000), bedroom["size_range"])
        size_groups = self.build(
            "Median Condominium Price Across Property Size Groups"
        )
        self.assertEqual(CONDO_SIZE_LABELS, size_groups._eda_metadata["size_bins"])
        completion = self.build(
            "Median Condominium Price by State and Completion Period"
        )
        self.assertEqual(COMPLETION_LABELS, completion._eda_metadata["completion_periods"])
        ppsf = self.build(
            "Median Price per Square Foot by State and Property Type"
        )
        self.assertEqual(MAJOR_PROPERTY_TYPES, ppsf._eda_metadata["property_types"])

    def test_boxplots_hide_only_displayed_fliers_without_removing_rows(self):
        condos = self.canonical[self.canonical["property_type"].eq("Condominium")]
        bathroom_expected = condos[condos["bathroom"].between(1, 5)]
        bathroom = self.build("Condominium Price Distribution by Bathroom Count")
        self.assertFalse(bathroom._eda_metadata["showfliers"])
        self.assertEqual(len(bathroom_expected), bathroom._eda_metadata["rows_used"])

        floor_order = ["Low", "Medium", "High"]
        floor_expected = condos[condos["floor_range"].isin(floor_order)]
        floor = self.build("Condominium Price by Floor Range")
        self.assertFalse(floor._eda_metadata["showfliers"])
        self.assertEqual(floor_order, floor._eda_metadata["floor_order"])
        self.assertEqual(len(floor_expected), floor._eda_metadata["rows_used"])

    def test_renderer_has_dropdowns_and_one_matplotlib_render_path(self):
        renderer_source = inspect.getsource(render_eda_page)
        self.assertIn('key="eda_category"', renderer_source)
        self.assertIn('key="eda_visualization"', renderer_source)
        self.assertIn("EDA_VISUALIZATIONS[category][chart_name]", renderer_source)
        self.assertEqual(1, renderer_source.count("st.pyplot"))
        self.assertIn('width="stretch"', renderer_source)
        self.assertNotIn("st.plotly_chart", renderer_source)
        self.assertIn("plt.close(figure)", renderer_source)

    def test_no_old_prepared_dependency_or_mutating_code_path_exists(self):
        module_source = inspect.getsource(eda_page).replace("\\", "/")
        for obsolete in (
            "data/processed/prepared_dataset.csv",
            "data/processed/final_prepared_dataset.csv",
        ):
            self.assertNotIn(obsolete, module_source)
        for forbidden in (".fit(", "fit_", "to_csv", "write_text", "run_experiment"):
            self.assertNotIn(forbidden, module_source)


if __name__ == "__main__":
    unittest.main()
