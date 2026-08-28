"""Report-faithful dropdown EDA backed by current repository datasets."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "houses.csv"
CURRENT_PREPARED_PATH = (
    PROJECT_ROOT / "data" / "processed" / "production_prepared_dataset.csv"
)
CANONICAL_PATH = PROJECT_ROOT / "data" / "processed" / "enhanced_city_dataset.csv"
EXPECTED_SHAPES = {
    "Raw source": (4_000, 32),
    "Production prepared": (3_791, 22),
    "Enhanced canonical": (3_791, 34),
}
MAJOR_PROPERTY_TYPES = ["Condominium", "Apartment", "Service Residence", "Flat"]
BEDROOM_SIZE_LABELS = [
    "300-700",
    "701-900",
    "901-1100",
    "1101-1300",
    "1301-1600",
    "1601-2000",
]
BEDROOM_SIZE_BINS = [300, 701, 901, 1101, 1301, 1601, 2001]
CONDO_SIZE_LABELS = [*BEDROOM_SIZE_LABELS, "2001-5000"]
CONDO_SIZE_BINS = [*BEDROOM_SIZE_BINS, 5001]
COMPLETION_LABELS = ["<=2000", "2001-2010", "2011-2015", "2016-2020", "2021-2026"]
COMPLETION_BINS = [0, 2001, 2011, 2016, 2021, 2027]
REPORT_PALETTE = ["#2563eb", "#d97706", "#db2777", "#4d7c0f"]


@st.cache_data
def load_eda_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read current immutable raw, prepared-stage, and canonical datasets."""
    raw = pd.read_csv(RAW_PATH)
    prepared = pd.read_csv(CURRENT_PREPARED_PATH)
    canonical = pd.read_csv(CANONICAL_PATH)
    frames = {
        "Raw source": raw,
        "Production prepared": prepared,
        "Enhanced canonical": canonical,
    }
    for name, frame in frames.items():
        if frame.shape != EXPECTED_SHAPES[name]:
            raise ValueError(
                f"{name} has shape {frame.shape}; expected {EXPECTED_SHAPES[name]}."
            )
    required = {
        "listing_id",
        "price",
        "property_size_sqft",
        "bedroom",
        "bathroom",
        "parking_lot",
        "completion_year",
        "property_type",
        "tenure_type",
        "land_title",
        "floor_range",
        "state",
    }
    missing = sorted(required.difference(canonical.columns))
    if missing:
        raise ValueError(f"Canonical EDA data is missing fields: {missing}")
    return raw, prepared, canonical


def _raw_with_standard_missing(raw: pd.DataFrame) -> pd.DataFrame:
    return raw.replace(r"^\s*-\s*$", np.nan, regex=True)


def _parse_raw_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.replace(",", "", regex=False)
    extracted = cleaned.str.extract(r"([-+]?\d*\.?\d+)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def _condominiums(canonical: pd.DataFrame) -> pd.DataFrame:
    return canonical[canonical["property_type"].eq("Condominium")].copy()


def _with_ppsf(canonical: pd.DataFrame) -> pd.DataFrame:
    frame = canonical.copy()
    frame["price_psf"] = frame["price"] / frame["property_size_sqft"]
    return frame[np.isfinite(frame["price_psf"]) & frame["price_psf"].gt(0)]


def _new_figure(figsize: tuple[float, float]) -> tuple[Figure, plt.Axes]:
    sns.set_theme(style="whitegrid")
    return plt.subplots(figsize=figsize)


def _finish(fig: Figure) -> Figure:
    fig.tight_layout()
    return fig


def _record(fig: Figure, **metadata) -> Figure:
    """Attach analytical-contract metadata for regression tests and maintenance."""
    fig._eda_metadata = metadata  # type: ignore[attr-defined]
    return fig


def dataset_overview_frame(
    raw: pd.DataFrame,
    prepared: pd.DataFrame,
    canonical: pd.DataFrame,
) -> pd.DataFrame:
    normalized_raw = _raw_with_standard_missing(raw)
    rows = []
    for stage, source, frame, missing_cells in (
        ("Raw source", "data/raw/houses.csv", raw, int(normalized_raw.isna().sum().sum())),
        (
            "Production prepared",
            "data/processed/production_prepared_dataset.csv",
            prepared,
            int(prepared.isna().sum().sum()),
        ),
        (
            "Enhanced canonical",
            "data/processed/enhanced_city_dataset.csv",
            canonical,
            int(canonical.isna().sum().sum()),
        ),
    ):
        rows.append(
            {
                "Dataset Stage": stage,
                "Source": source,
                "Rows": len(frame),
                "Columns": len(frame.columns),
                "Missing Cells": missing_cells,
                "Duplicate Rows": int(frame.duplicated().sum()),
            }
        )
    return pd.DataFrame(rows)


def descriptive_statistics_frame(canonical: pd.DataFrame) -> pd.DataFrame:
    labels = {
        "price": "Listing Price (RM)",
        "property_size_sqft": "Property Size (sq.ft.)",
        "bedroom": "Bedrooms",
        "bathroom": "Bathrooms",
        "parking_lot": "Parking Lots",
        "completion_year": "Completion Year",
    }
    return (
        canonical[list(labels)]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .T.rename(index=labels)
        .reset_index(names="Feature")
        .rename(
            columns={
                "count": "Count",
                "mean": "Mean",
                "std": "Std. Dev.",
                "min": "Minimum",
                "25%": "25th Percentile",
                "50%": "Median",
                "75%": "75th Percentile",
                "max": "Maximum",
            }
        )
    )


def build_average_price_by_state(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> Figure:
    grouped = (
        canonical.groupby("state", as_index=False, observed=True)
        .agg(Mean_Price=("price", "mean"), Count=("listing_id", "size"))
        .loc[lambda frame: frame["Count"].ge(10)]
        .sort_values("Mean_Price")
    )
    fig, ax = _new_figure((10, 7))
    sns.barplot(
        data=grouped,
        x="Mean_Price",
        y="state",
        hue="state",
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.set_title("Average Property Listing Price by State")
    ax.set_xlabel("Average Listing Price (RM)")
    ax.set_ylabel("State")
    ax.ticklabel_format(style="plain", axis="x")
    _record(fig, chart_type="horizontal_bar", minimum_state_count=10, states=grouped["state"].tolist())
    return _finish(fig)


def build_condo_mean_median_by_state(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> Figure:
    grouped = (
        _condominiums(canonical)
        .groupby("state", as_index=False, observed=True)
        .agg(Count=("listing_id", "size"), Mean=("price", "mean"), Median=("price", "median"))
        .loc[lambda frame: frame["Count"].ge(10)]
        .sort_values("Median", ascending=False)
    )
    long = grouped.melt(
        id_vars=["state", "Count"],
        value_vars=["Mean", "Median"],
        var_name="Statistic",
        value_name="Listing Price (RM)",
    )
    fig, ax = _new_figure((11, 6))
    sns.barplot(
        data=long,
        x="state",
        y="Listing Price (RM)",
        hue="Statistic",
        order=grouped["state"].tolist(),
        palette=[REPORT_PALETTE[0], REPORT_PALETTE[1]],
        ax=ax,
    )
    ax.set_title("Mean and Median Condominium Price by State")
    ax.set_xlabel("State")
    ax.set_ylabel("Listing Price (RM)")
    ax.tick_params(axis="x", rotation=45)
    ax.ticklabel_format(style="plain", axis="y")
    _record(fig, chart_type="grouped_bar", minimum_state_count=10, statistics=["Mean", "Median"])
    return _finish(fig)


def build_property_type_price_distribution(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> Figure:
    frame = canonical[canonical["property_type"].isin(MAJOR_PROPERTY_TYPES)].copy()
    sns.set_theme(style="whitegrid")
    grid = sns.FacetGrid(
        frame,
        row="property_type",
        row_order=MAJOR_PROPERTY_TYPES,
        height=3,
        aspect=3,
        sharex=False,
    )
    grid.map_dataframe(sns.histplot, x="price", bins=50, kde=True, color=REPORT_PALETTE[0])
    grid.set_axis_labels("Listing Price (RM, log scale)", "Count / Density")
    grid.set_titles(row_template="{row_name}")
    for ax in grid.axes.flat:
        ax.set_xscale("log")
    grid.figure.subplots_adjust(top=0.94, hspace=0.5)
    grid.figure.suptitle("Listing Price Distribution by Property Type")
    _record(
        grid.figure,
        chart_type="faceted_histogram",
        property_types=MAJOR_PROPERTY_TYPES,
        bins=50,
        kde=True,
        xscale="log",
    )
    return grid.figure


def build_condo_price_by_bedrooms(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> Figure:
    frame = _condominiums(canonical)
    frame = frame[frame["bedroom"].between(1, 5)]
    grouped = frame.groupby("bedroom", as_index=False, observed=True)["price"].median()
    fig, ax = _new_figure((9, 6))
    sns.lineplot(data=grouped, x="bedroom", y="price", marker="o", color=REPORT_PALETTE[0], ax=ax)
    ax.set_title("Median Condominium Price by Bedroom Count")
    ax.set_xlabel("Number of Bedrooms")
    ax.set_ylabel("Median Listing Price (RM)")
    ax.set_xticks(grouped["bedroom"].astype(int).tolist())
    ax.ticklabel_format(style="plain", axis="y")
    _record(fig, chart_type="line", marker="o", bedroom_range=(1, 5))
    return _finish(fig)


def build_bedroom_size_heatmap(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> Figure:
    frame = _condominiums(canonical)
    frame = frame[
        frame["bedroom"].between(1, 5)
        & frame["property_size_sqft"].between(300, 2000)
    ].copy()
    frame["Size Group"] = pd.cut(
        frame["property_size_sqft"],
        bins=BEDROOM_SIZE_BINS,
        labels=BEDROOM_SIZE_LABELS,
        right=False,
    )
    pivot = frame.pivot_table(
        index="bedroom",
        columns="Size Group",
        values="price",
        aggfunc="median",
        observed=False,
    ).reindex(index=range(1, 6), columns=BEDROOM_SIZE_LABELS)
    fig, ax = _new_figure((11, 6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=0.5, ax=ax)
    ax.set_title("Median Condominium Price by Bedroom Count and Property Size")
    ax.set_xlabel("Property Size (sq ft)")
    ax.set_ylabel("Number of Bedrooms")
    _record(
        fig,
        chart_type="heatmap",
        annotations=True,
        bedroom_range=(1, 5),
        size_range=(300, 2000),
        size_bins=BEDROOM_SIZE_LABELS,
    )
    return _finish(fig)


def build_condo_price_by_bathrooms(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> Figure:
    frame = _condominiums(canonical)
    frame = frame[frame["bathroom"].between(1, 5)].copy()
    fig, ax = _new_figure((9, 6))
    sns.boxplot(data=frame, x="bathroom", y="price", showfliers=False, color="#93c5fd", ax=ax)
    ax.set_title("Condominium Price Distribution by Bathroom Count")
    ax.set_xlabel("Number of Bathrooms")
    ax.set_ylabel("Listing Price (RM)")
    ax.ticklabel_format(style="plain", axis="y")
    _record(
        fig,
        chart_type="boxplot",
        showfliers=False,
        bathroom_range=(1, 5),
        rows_used=len(frame),
    )
    return _finish(fig)


def build_size_vs_price(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> Figure:
    frame = canonical[canonical["property_size_sqft"].between(300, 5000)].copy()
    correlation = float(frame[["property_size_sqft", "price"]].corr(method="pearson").iloc[0, 1])
    fig, ax = _new_figure((10, 7))
    density = ax.hexbin(
        frame["property_size_sqft"],
        frame["price"],
        gridsize=35,
        mincnt=1,
        cmap="viridis",
    )
    colorbar = fig.colorbar(density, ax=ax)
    colorbar.set_label("Number of Properties")
    ax.set_title(f"Property Size vs Listing Price (r = {correlation:.2f})")
    ax.set_xlabel("Property Size (sq ft)")
    ax.set_ylabel("Listing Price (RM)")
    ax.ticklabel_format(style="plain", axis="both")
    _record(
        fig,
        chart_type="hexbin",
        correlation=correlation,
        correlation_method="pearson",
        gridsize=35,
        mincnt=1,
        size_range=(300, 5000),
    )
    return _finish(fig)


def build_condo_price_by_size_group(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> Figure:
    frame = _condominiums(canonical)
    frame = frame[frame["property_size_sqft"].between(300, 5000)].copy()
    frame["Size Group"] = pd.cut(
        frame["property_size_sqft"],
        bins=CONDO_SIZE_BINS,
        labels=CONDO_SIZE_LABELS,
        right=False,
    )
    grouped = (
        frame.groupby("Size Group", as_index=False, observed=False)["price"]
        .median()
        .rename(columns={"price": "Median Price"})
    )
    fig, ax = _new_figure((10, 6))
    sns.lineplot(
        data=grouped,
        x="Size Group",
        y="Median Price",
        marker="o",
        color=REPORT_PALETTE[0],
        sort=False,
        ax=ax,
    )
    ax.set_title("Median Condominium Price Across Property Size Groups")
    ax.set_xlabel("Property Size Group (sq ft)")
    ax.set_ylabel("Median Listing Price (RM)")
    ax.tick_params(axis="x", rotation=35)
    ax.ticklabel_format(style="plain", axis="y")
    _record(fig, chart_type="line", marker="o", size_range=(300, 5000), size_bins=CONDO_SIZE_LABELS)
    return _finish(fig)


def build_price_cdf_by_tenure(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> Figure:
    frame = canonical[canonical["tenure_type"].isin(["Freehold", "Leasehold"])].copy()
    fig, ax = _new_figure((10, 7))
    sns.ecdfplot(data=frame, x="price", hue="tenure_type", palette=REPORT_PALETTE[:2], ax=ax)
    ax.set_xscale("log")
    ax.set_title("Cumulative Distribution of Listing Prices by Tenure")
    ax.set_xlabel("Listing Price (RM, log scale)")
    ax.set_ylabel("Cumulative Proportion")
    _record(fig, chart_type="ecdf", tenures=["Freehold", "Leasehold"], xscale="log")
    return _finish(fig)


def build_condo_price_state_tenure(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> Figure:
    frame = _condominiums(canonical)
    frame = frame[frame["tenure_type"].isin(["Freehold", "Leasehold"])].copy()
    state_counts = frame.groupby("state", observed=True).size()
    valid_states = state_counts[state_counts.ge(20)].index.tolist()
    frame = frame[frame["state"].isin(valid_states)]
    fig, ax = _new_figure((12, 7))
    sns.pointplot(
        data=frame,
        x="state",
        y="price",
        hue="tenure_type",
        order=valid_states,
        estimator=np.median,
        errorbar=None,
        dodge=0.3,
        markers=["o", "s"],
        palette=REPORT_PALETTE[:2],
        ax=ax,
    )
    ax.set_title("Median Condo Price by State and Tenure Type")
    ax.set_xlabel("State")
    ax.set_ylabel("Median Listing Price (RM)")
    ax.tick_params(axis="x", rotation=45)
    ax.ticklabel_format(style="plain", axis="y")
    _record(
        fig,
        chart_type="pointplot",
        statistic="median",
        minimum_state_count=20,
        tenures=["Freehold", "Leasehold"],
    )
    return _finish(fig)


def build_condo_price_by_completion_year(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> Figure:
    frame = _condominiums(canonical)
    frame = frame[frame["completion_year"].between(1980, 2026)]
    grouped = (
        frame.groupby("completion_year", as_index=False, observed=True)
        .agg(Median=("price", "median"), Count=("listing_id", "size"))
        .loc[lambda values: values["Count"].ge(5)]
        .sort_values("completion_year")
    )
    fig, ax = _new_figure((12, 6))
    sns.lineplot(
        data=grouped,
        x="completion_year",
        y="Median",
        marker="o",
        color=REPORT_PALETTE[0],
        ax=ax,
    )
    ax.set_title("Median Condominium Price by Completion Year")
    ax.set_xlabel("Completion Year")
    ax.set_ylabel("Median Listing Price (RM)")
    ax.ticklabel_format(style="plain", axis="y")
    _record(
        fig,
        chart_type="line",
        marker="o",
        completion_year_range=(1980, 2026),
        minimum_year_count=5,
    )
    return _finish(fig)


def build_condo_price_state_completion_period(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> Figure:
    frame = _condominiums(canonical)
    state_counts = frame.groupby("state", observed=True).size()
    valid_states = state_counts[state_counts.ge(20)].index.tolist()
    frame = frame[
        frame["state"].isin(valid_states)
        & frame["completion_year"].between(1900, 2026)
    ].copy()
    frame["Completion Period"] = pd.cut(
        frame["completion_year"],
        bins=COMPLETION_BINS,
        labels=COMPLETION_LABELS,
        right=False,
    )
    pivot = frame.pivot_table(
        index="state",
        columns="Completion Period",
        values="price",
        aggfunc="median",
        observed=False,
    ).reindex(index=valid_states, columns=COMPLETION_LABELS)
    fig, ax = _new_figure((12, 7))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=0.5, ax=ax)
    ax.set_title("Median Condominium Price by State and Completion Period")
    ax.set_xlabel("Completion Period")
    ax.set_ylabel("State")
    _record(
        fig,
        chart_type="heatmap",
        annotations=True,
        completion_periods=COMPLETION_LABELS,
        minimum_state_count=20,
    )
    return _finish(fig)


def build_price_distribution_by_land_title(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> Figure:
    order = ["Bumi Lot", "Non Bumi Lot"]
    frame = canonical[canonical["land_title"].isin(order)].copy()
    fig, ax = _new_figure((9, 6))
    sns.boxenplot(
        data=frame,
        x="land_title",
        y="price",
        order=order,
        hue="land_title",
        palette=REPORT_PALETTE[:2],
        legend=False,
        ax=ax,
    )
    ax.set_title("Listing Price Distribution by Land Title")
    ax.set_xlabel("Land Title")
    ax.set_ylabel("Listing Price (RM)")
    ax.ticklabel_format(style="plain", axis="y")
    _record(
        fig,
        chart_type="boxenplot",
        land_titles=order,
        rows_used=len(frame),
    )
    return _finish(fig)


def build_condo_price_by_floor_range(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> Figure:
    order = ["Low", "Medium", "High"]
    frame = _condominiums(canonical)
    frame = frame[frame["floor_range"].isin(order)].copy()
    fig, ax = _new_figure((9, 6))
    sns.boxplot(
        data=frame,
        x="floor_range",
        y="price",
        order=order,
        showfliers=False,
        color="#93c5fd",
        ax=ax,
    )
    ax.set_title("Condominium Price by Floor Range")
    ax.set_xlabel("Floor Range")
    ax.set_ylabel("Listing Price (RM)")
    ax.ticklabel_format(style="plain", axis="y")
    _record(fig, chart_type="boxplot", showfliers=False, floor_order=order, rows_used=len(frame))
    return _finish(fig)


def build_condo_price_density_by_parking(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> Figure:
    frame = _condominiums(canonical)
    frame = frame[frame["parking_lot"].between(1, 4)].copy()
    frame["Parking Allocation"] = frame["parking_lot"].astype(int).astype(str)
    fig, ax = _new_figure((10, 7))
    sns.kdeplot(
        data=frame,
        x="price",
        hue="Parking Allocation",
        hue_order=["1", "2", "3", "4"],
        fill=True,
        common_norm=False,
        palette=REPORT_PALETTE,
        ax=ax,
    )
    ax.set_title("Condominium Price Density by Parking Allocation")
    ax.set_xlabel("Listing Price (RM)")
    ax.set_ylabel("Density")
    ax.ticklabel_format(style="plain", axis="x")
    _record(
        fig,
        chart_type="kde",
        parking_range=(1, 4),
        fill=True,
        common_norm=False,
    )
    return _finish(fig)


def build_median_ppsf_state_property_type(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> Figure:
    frame = _with_ppsf(canonical)
    frame = frame[frame["property_type"].isin(MAJOR_PROPERTY_TYPES)].copy()
    state_counts = frame.groupby("state", observed=True).size()
    valid_states = state_counts[state_counts.ge(20)].index.tolist()
    frame = frame[frame["state"].isin(valid_states)]
    pivot = frame.pivot_table(
        index="state",
        columns="property_type",
        values="price_psf",
        aggfunc="median",
        observed=True,
    ).reindex(index=valid_states, columns=MAJOR_PROPERTY_TYPES)
    fig, ax = _new_figure((12, 8))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=0.5, ax=ax)
    ax.set_title("Median Price per Square Foot by State and Property Type")
    ax.set_xlabel("Property Type")
    ax.set_ylabel("State")
    _record(
        fig,
        chart_type="heatmap",
        annotations=True,
        property_types=MAJOR_PROPERTY_TYPES,
        minimum_state_count=20,
        ppsf_formula="price / property_size_sqft",
    )
    return _finish(fig)


def build_ppsf_distribution_property_type(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> Figure:
    frame = _with_ppsf(canonical)
    frame = frame[frame["property_type"].isin(MAJOR_PROPERTY_TYPES)].copy()
    fig, ax = _new_figure((11, 7))
    sns.violinplot(
        data=frame,
        x="property_type",
        y="price_psf",
        order=MAJOR_PROPERTY_TYPES,
        inner="quartile",
        cut=0,
        palette=REPORT_PALETTE,
        hue="property_type",
        legend=False,
        ax=ax,
    )
    ax.set_title("Price per Square Foot Distribution by Property Type")
    ax.set_xlabel("Property Type")
    ax.set_ylabel("Price per Square Foot (RM)")
    ax.ticklabel_format(style="plain", axis="y")
    _record(
        fig,
        chart_type="violin",
        property_types=MAJOR_PROPERTY_TYPES,
        inner="quartile",
        cut=0,
    )
    return _finish(fig)


def build_missing_values_before_preparation(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> Figure:
    missing_pct = _raw_with_standard_missing(raw).isna().mean().mul(100)
    missing_pct = missing_pct[missing_pct.gt(0)].sort_values()
    fig, ax = _new_figure((10, 8))
    ax.barh(missing_pct.index, missing_pct.values, color=REPORT_PALETTE[1])
    ax.set_title("Missing Values by Feature Before Data Preparation")
    ax.set_xlabel("Missing Values (%)")
    ax.set_ylabel("Feature")
    _record(fig, chart_type="horizontal_bar", missing_markers=["NaN", "-"], positive_only=True)
    return _finish(fig)


def build_property_size_before_preparation(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> Figure:
    sizes = _parse_raw_numeric(raw["Property Size"]).dropna()
    fig, ax = _new_figure((10, 5))
    sns.boxplot(x=sizes, color="#fbbf24", ax=ax)
    ax.set_xscale("log")
    ax.set_title("Property Size Distribution Before Outlier Treatment")
    ax.set_xlabel("Property Size (sq.ft., log scale)")
    ax.set_ylabel("")
    _record(fig, chart_type="horizontal_boxplot", xscale="log", rows_used=len(sizes))
    return _finish(fig)


def build_property_size_after_preparation(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> Figure:
    fig, ax = _new_figure((10, 5))
    sns.boxplot(x=canonical["property_size_sqft"], color="#93c5fd", ax=ax)
    ax.set_title("Property Size Distribution After Outlier Treatment")
    ax.set_xlabel("Property Size (sq.ft.)")
    ax.set_ylabel("")
    _record(
        fig,
        chart_type="horizontal_boxplot",
        showfliers=True,
        rows_used=len(canonical),
    )
    return _finish(fig)


def build_dataset_size_stages(
    raw: pd.DataFrame, prepared: pd.DataFrame, canonical: pd.DataFrame
) -> Figure:
    stages = ["Raw Dataset", "After Data Preparation", "Final Enhanced Dataset"]
    counts = [len(raw), len(prepared), len(canonical)]
    fig, ax = _new_figure((10, 6))
    bars = ax.bar(stages, counts, color=[REPORT_PALETTE[1], REPORT_PALETTE[0], REPORT_PALETTE[3]])
    ax.bar_label(bars, labels=[f"{count:,}" for count in counts], padding=3)
    ax.set_title("Dataset Size Across Data Preparation Stages")
    ax.set_xlabel("Data Preparation Stage")
    ax.set_ylabel("Number of Records")
    ax.tick_params(axis="x", rotation=15)
    _record(fig, chart_type="bar", stages=stages, counts=counts)
    return _finish(fig)


ChartBuilder = Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], Figure]
EDA_VISUALIZATIONS: dict[str, dict[str, ChartBuilder]] = {
    "Price & Location": {
        "Average Property Listing Price by State": build_average_price_by_state,
        "Mean and Median Condominium Price by State": build_condo_mean_median_by_state,
        "Listing Price Distribution by Property Type": build_property_type_price_distribution,
    },
    "Property Characteristics": {
        "Median Condominium Price by Bedroom Count": build_condo_price_by_bedrooms,
        "Median Condominium Price by Bedroom Count and Property Size": build_bedroom_size_heatmap,
        "Condominium Price Distribution by Bathroom Count": build_condo_price_by_bathrooms,
        "Listing Price Distribution by Land Title": build_price_distribution_by_land_title,
        "Condominium Price by Floor Range": build_condo_price_by_floor_range,
        "Condominium Price Density by Parking Allocation": build_condo_price_density_by_parking,
    },
    "Size & Price": {
        "Property Size vs Listing Price": build_size_vs_price,
        "Median Condominium Price Across Property Size Groups": build_condo_price_by_size_group,
    },
    "Tenure & Development": {
        "Cumulative Distribution of Listing Prices by Tenure": build_price_cdf_by_tenure,
        "Median Condo Price by State and Tenure Type": build_condo_price_state_tenure,
        "Median Condominium Price by Completion Year": build_condo_price_by_completion_year,
        "Median Condominium Price by State and Completion Period": build_condo_price_state_completion_period,
    },
    "Price per Square Foot": {
        "Median Price per Square Foot by State and Property Type": build_median_ppsf_state_property_type,
        "Price per Square Foot Distribution by Property Type": build_ppsf_distribution_property_type,
    },
}


def render_eda_page() -> None:
    """Render overview tables and exactly one selected report-style chart."""
    st.subheader("Exploratory Data Analysis")
    raw, prepared, canonical = load_eda_sources()

    st.write("### Dataset Overview")
    overview = dataset_overview_frame(raw, prepared, canonical)
    st.dataframe(
        overview.style.format(
            {
                "Rows": "{:,}",
                "Columns": "{:,}",
                "Missing Cells": "{:,}",
                "Duplicate Rows": "{:,}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.write("### Descriptive Statistics")
    statistics = descriptive_statistics_frame(canonical)
    st.dataframe(
        statistics.style.format(
            {column: "{:,.2f}" for column in statistics.columns if column != "Feature"}
        ),
        width="stretch",
        hide_index=True,
    )

    st.write("### Visual Analysis")
    category = st.selectbox(
        "EDA Category",
        list(EDA_VISUALIZATIONS),
        key="eda_category",
    )
    chart_name = st.selectbox(
        "Select Visualization",
        list(EDA_VISUALIZATIONS[category]),
        key="eda_visualization",
    )
    figure = EDA_VISUALIZATIONS[category][chart_name](raw, prepared, canonical)
    st.pyplot(figure, width="stretch")
    plt.close(figure)
