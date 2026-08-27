"""Validity-first audit and isolated Position-regex LightGBM comparison.

The audit never uses price or PPSF percentiles as deletion rules. Percentile
bands are diagnostics only. Rows are removed only when a deterministic rule
identifies an unusable single-property observation or a critical invalid value.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.description_text_features.model_builders import fit_lightgbm_fold
from experiments.description_text_features.regex_features import (
    REGEX_GROUPS,
    extract_regex_features,
)
from experiments.description_text_features.text_cleaning import link_descriptions


EXPERIMENT = ROOT / "experiments" / "data_validity_audit"
DATA_PATH = ROOT / "data" / "processed" / "enhanced_city_dataset.csv"
RAW_PATH = ROOT / "data" / "raw" / "houses.csv"
REFERENCE_OOF_PATH = ROOT / "experiments" / "description_text_features" / "oof_predictions.csv"
REFERENCE_VARIANT = "regex_group_position"
POSITION_FEATURES = list(REGEX_GROUPS["position"])
PREDICTOR_COUNT = 47
EXPECTED_ROWS = 3_791
EXPECTED_REFERENCE = {
    "rmse_rm": 118_750.19350875785,
    "mae_rm": 60_967.10968555279,
    "r2": 0.8702674858843791,
    "top5_rmse_rm": 412_319.68890075386,
}
AUDIT_COLUMNS = [
    "row_id",
    "price",
    "property_size_sqft",
    "ppsf",
    "property_type",
    "building_name",
    "developer",
    "city",
    "state",
    "bedroom",
    "bathroom",
    "parking_lot",
    "completion_year",
    "number_of_floors",
    "total_units",
    "validity_status",
    "flag_reason",
    "recommended_action",
]
OPTIONAL_FIELDS = [
    "completion_year",
    "property_age",
    "number_of_floors",
    "total_units",
    "developer",
    "building_name",
    "floor_range",
    "parking_lot",
]
MISSING_TEXT = {"", "-", "--", "n/a", "na", "nan", "none", "null", "unknown"}
PRICE_BANDS = (
    ("P00_P95", 0.00, 0.95),
    ("P95_P99", 0.95, 0.99),
    ("P99_P100", 0.99, 1.00),
)


RULES = {
    "EXACT_DUPLICATE_CANONICAL_ROW": {
        "classification": "CLEARLY_INVALID",
        "criterion": "A later row is byte-for-value identical across every canonical column, including listing_id.",
        "rationale": "It is the same canonical observation repeated; keeping one preserves information without double counting.",
        "deletion_rule": True,
    },
    "CRITICAL_INVALID_PRICE": {
        "classification": "CLEARLY_INVALID",
        "criterion": "price is missing, non-finite, or <= 0",
        "rationale": "A positive finite target is required for supervised PPSF modelling.",
        "deletion_rule": True,
    },
    "CRITICAL_INVALID_PROPERTY_SIZE": {
        "classification": "CLEARLY_INVALID",
        "criterion": "property_size_sqft is missing, non-finite, or <= 0",
        "rationale": "A positive finite denominator is required to define PPSF.",
        "deletion_rule": True,
    },
    "IMPOSSIBLE_CORE_COUNT": {
        "classification": "CLEARLY_INVALID",
        "criterion": "bedroom, bathroom, or parking is non-integer, negative, or above the repository's broad hard ceiling of 20",
        "rationale": "These are discrete property counts; values outside the broad source-validity domain indicate parsing or entry corruption.",
        "deletion_rule": True,
    },
    "OUTSIDE_BROAD_SIZE_DOMAIN": {
        "classification": "CLEARLY_INVALID",
        "criterion": "property_size_sqft is outside the repository's established 100-20,000 sq-ft hard validity domain",
        "rationale": "The existing source cleaner defines this broad range as the parse-valid domain; it is not a percentile rule.",
        "deletion_rule": True,
    },
    "MULTI_UNIT_BUNDLE_NOT_SINGLE_OBSERVATION": {
        "classification": "CLEARLY_INVALID",
        "criterion": "description explicitly sells two units together and lists multiple unit sizes and component prices",
        "rationale": "One row cannot map a bundle price to one property size, so its target and PPSF are not a single-property observation.",
        "deletion_rule": True,
    },
    "BEDROOM_COUNT_DESCRIPTION_CONTRADICTION": {
        "classification": "CLEARLY_INVALID",
        "criterion": "structured bedroom >= 8 but the description explicitly identifies three bedrooms and does not identify eight bedrooms",
        "rationale": "The structured core count conflicts with direct listing evidence; the raw description instead itemizes non-bedroom rooms.",
        "deletion_rule": True,
    },
    "BATHROOM_COUNT_CONFIGURATION_CONTRADICTION": {
        "classification": "CLEARLY_INVALID",
        "criterion": "bathroom >= 6 in <600 sq ft, exceeds bedroom by >3, and description advertises two-bathroom configurations rather than the encoded count",
        "rationale": "The structured count is physically implausible for the stated size and contradicted by the listing's configurations.",
        "deletion_rule": True,
    },
    "POSSIBLE_REPEAT_DIFFERENT_LISTING_ID": {
        "classification": "SUSPICIOUS",
        "criterion": "different listing_ids have identical canonical fields excluding listing_id",
        "rationale": "May be a relisted ad or may be distinct units sharing attributes; it must not be auto-deleted.",
        "deletion_rule": False,
    },
    "ISOLATED_FLOOR_COUNT_NEEDS_REVIEW": {
        "classification": "SUSPICIOUS",
        "criterion": "number_of_floors >= 80 (the dataset has one isolated value at 135; the next-highest observed value is 63)",
        "rationale": "The value is not impossible globally, but it is unsupported and isolated enough to require source verification.",
        "deletion_rule": False,
    },
    "STRATA_TOTAL_UNITS_NEEDS_REVIEW": {
        "classification": "SUSPICIOUS",
        "criterion": "a flat/apartment/condominium/service residence reports <=3 or >=5,000 total units",
        "rationale": "This may refer to a block, phase, or parsing artifact rather than the development; the optional field can be imputed, so retain the row.",
        "deletion_rule": False,
    },
    "UNSUPPORTED_HIGH_PARKING_COUNT": {
        "classification": "SUSPICIOUS",
        "criterion": "parking_lot >=5 without the same count explicitly supported in the description",
        "rationale": "Large parking allocations exist, but unsupported counts in small/ordinary units need source review and are not deletion grounds.",
        "deletion_rule": False,
    },
    "SINGLE_DESCRIPTION_SIZE_MISMATCH": {
        "classification": "SUSPICIOUS",
        "criterion": "description contains one plausible explicit sq-ft value differing by >10% from the structured size",
        "rationale": "May reflect garden/balcony area, a stale template, or a parse error; retain pending verification.",
        "deletion_rule": False,
    },
    "OPTIONAL_COMPLETION_YEAR_NEEDS_REVIEW": {
        "classification": "SUSPICIOUS",
        "criterion": "non-missing completion_year is non-integer or outside the repository's broad 1900-2030 source-validity domain",
        "rationale": "The optional year should be verified or set missing; it does not make the target/size observation unusable.",
        "deletion_rule": False,
    },
    "OPTIONAL_FLOOR_OR_UNIT_DOMAIN_NEEDS_REVIEW": {
        "classification": "SUSPICIOUS",
        "criterion": "non-missing floors is non-integer/outside 1-200, or total_units is non-integer/outside 1-20,000",
        "rationale": "These optional development attributes should be verified or imputed rather than causing automatic row deletion.",
        "deletion_rule": False,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protected_snapshot() -> dict[str, str]:
    roots = [
        ROOT / "data",
        ROOT / "results",
        ROOT / "prototype",
        ROOT / "src",
        ROOT / "configs",
        ROOT / "scripts",
    ]
    files = [ROOT / "app.py", ROOT / "README.md", ROOT / "requirements.txt"]
    for directory in roots:
        if directory.exists():
            files.extend(path for path in directory.rglob("*") if path.is_file())
    for directory in (ROOT / "experiments").iterdir():
        if directory.is_dir() and directory.name not in {"data_validity_audit", "__pycache__"}:
            files.extend(path for path in directory.rglob("*") if path.is_file())
    return {
        path.relative_to(ROOT).as_posix(): _sha256(path)
        for path in sorted(set(files))
        if path.exists()
    }


def _manifest_digest(snapshot: dict[str, str]) -> str:
    payload = "\n".join(f"{path}:{digest}" for path, digest in sorted(snapshot.items()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_clean(value):
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_clean(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if value is pd.NA:
        return None
    return value


def _is_missing(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.isna() | ~np.isfinite(numeric.astype(float))
    text = series.astype("string").str.strip().str.lower()
    return text.isna() | text.isin(MISSING_TEXT)


def _explicit_sizes(text: str) -> list[float]:
    matches = re.findall(
        r"(?<!\d)(\d{3,5}(?:\.\d+)?)\s*(?:sq\.?\s*ft|sqft|sft|sf)\b",
        text.replace(",", ""),
        flags=re.IGNORECASE,
    )
    return sorted({float(value) for value in matches if 100 <= float(value) <= 20_000})


def _explicit_rm_prices(text: str) -> list[float]:
    values = []
    for number, suffix in re.findall(
        r"\brm\s*([0-9]+(?:\.[0-9]+)?)\s*([km]?)\b", text.replace(",", ""), flags=re.IGNORECASE
    ):
        value = float(number)
        if suffix.lower() == "k":
            value *= 1_000
        elif suffix.lower() == "m":
            value *= 1_000_000
        if value >= 30_000:
            values.append(value)
    return sorted(set(values))


def _description_supports_count(text: str, value: float, noun_pattern: str) -> bool:
    integer = int(value)
    patterns = [
        rf"\b{integer}\s*[- ]?{noun_pattern}\b",
        rf"\b{noun_pattern}\s*[:\-]?\s*{integer}\b",
    ]
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _add_reason(reasons: list[list[str]], evidence: list[list[str]], mask, reason: str, detail: str) -> None:
    positions = np.flatnonzero(np.asarray(mask, dtype=bool))
    for position in positions:
        if reason not in reasons[position]:
            reasons[position].append(reason)
            evidence[position].append(detail)


def build_audit(
    frame: pd.DataFrame, descriptions: pd.Series, regex: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    count = len(frame)
    reasons: list[list[str]] = [[] for _ in range(count)]
    evidence: list[list[str]] = [[] for _ in range(count)]
    clear = np.zeros(count, dtype=bool)
    suspicious = np.zeros(count, dtype=bool)
    ppsf = frame["price"].astype(float) / frame["property_size_sqft"].astype(float)

    exact_all = frame.duplicated(keep=False)
    exact_later = frame.duplicated(keep="first")
    _add_reason(reasons, evidence, exact_later, "EXACT_DUPLICATE_CANONICAL_ROW", "Identical across all 34 canonical columns; later copy only.")
    clear |= exact_later.to_numpy()

    price = pd.to_numeric(frame["price"], errors="coerce")
    size = pd.to_numeric(frame["property_size_sqft"], errors="coerce")
    invalid_price = price.isna() | ~np.isfinite(price.astype(float)) | price.le(0)
    invalid_size = size.isna() | ~np.isfinite(size.astype(float)) | size.le(0)
    _add_reason(reasons, evidence, invalid_price, "CRITICAL_INVALID_PRICE", "Price must be finite and strictly positive.")
    _add_reason(reasons, evidence, invalid_size, "CRITICAL_INVALID_PROPERTY_SIZE", "PPSF denominator must be finite and strictly positive.")
    clear |= invalid_price.to_numpy() | invalid_size.to_numpy()

    impossible_core = np.zeros(count, dtype=bool)
    core_details = []
    for column in ("bedroom", "bathroom", "parking_lot"):
        values = pd.to_numeric(frame[column], errors="coerce")
        mask = values.notna() & ((values < 0) | (values > 20) | ((values % 1) != 0))
        impossible_core |= mask.to_numpy()
        if mask.any():
            core_details.append(f"{column}: {int(mask.sum())}")
    _add_reason(
        reasons,
        evidence,
        impossible_core,
        "IMPOSSIBLE_CORE_COUNT",
        "Discrete core count is outside the broad 0-20 integer domain.",
    )
    clear |= impossible_core

    outside_size_domain = size.notna() & ~size.between(100, 20_000)
    _add_reason(
        reasons,
        evidence,
        outside_size_domain,
        "OUTSIDE_BROAD_SIZE_DOMAIN",
        "Size is outside the repository's established non-percentile parse-valid range.",
    )
    clear |= outside_size_domain.to_numpy()

    normalized_description = descriptions.fillna("").astype(str).str.lower()
    explicit_sizes = normalized_description.map(_explicit_sizes)
    explicit_prices = normalized_description.map(_explicit_rm_prices)
    multi_unit_phrase = normalized_description.str.contains(
        r"\bselling\s+(?:two|2)\s+units?\b", regex=True, na=False
    )
    multi_unit_bundle = multi_unit_phrase & explicit_sizes.map(len).ge(2) & explicit_prices.map(len).ge(2)
    _add_reason(
        reasons,
        evidence,
        multi_unit_bundle,
        "MULTI_UNIT_BUNDLE_NOT_SINGLE_OBSERVATION",
        "Description supplies two unit sizes and component prices while the row contains one size and one bundle price.",
    )
    clear |= multi_unit_bundle.to_numpy()

    explicit_three_bed = normalized_description.str.contains(
        r"\b3\s*[- ]?bedrooms?\b", regex=True, na=False
    )
    explicit_eight_bed = normalized_description.str.contains(
        r"\b8\s*[- ]?bedrooms?\b", regex=True, na=False
    )
    bedroom_contradiction = frame["bedroom"].ge(8) & explicit_three_bed & ~explicit_eight_bed
    _add_reason(
        reasons,
        evidence,
        bedroom_contradiction,
        "BEDROOM_COUNT_DESCRIPTION_CONTRADICTION",
        "Structured count is >=8; description explicitly says three bedrooms and itemizes other room types.",
    )
    clear |= bedroom_contradiction.to_numpy()

    explicit_two_bath = normalized_description.str.contains(
        r"\b2\s*[- ]?(?:bathrooms?|baths?)\b", regex=True, na=False
    )
    bathroom_contradiction = (
        frame["bathroom"].ge(6)
        & frame["property_size_sqft"].lt(600)
        & frame["bathroom"].gt(frame["bedroom"] + 3)
        & explicit_two_bath
    )
    _add_reason(
        reasons,
        evidence,
        bathroom_contradiction,
        "BATHROOM_COUNT_CONFIGURATION_CONTRADICTION",
        "Encoded bathrooms greatly exceed bedrooms in <600 sq ft; description offers two-bathroom layouts, not the encoded count.",
    )
    clear |= bathroom_contradiction.to_numpy()

    canonical_without_id = frame.drop(columns=["listing_id"])
    possible_repeat = canonical_without_id.duplicated(keep=False)
    repeat_hash = pd.util.hash_pandas_object(canonical_without_id, index=False).astype(str)
    repeat_group = pd.Series(pd.NA, index=frame.index, dtype="string")
    for sequence, (_, positions) in enumerate(frame.loc[possible_repeat].groupby(repeat_hash[possible_repeat]).groups.items(), 1):
        repeat_group.loc[list(positions)] = f"PR{sequence:03d}"
    same_description = frame.assign(_description=normalized_description).drop(columns=["listing_id"]).duplicated(keep=False)
    _add_reason(
        reasons,
        evidence,
        possible_repeat,
        "POSSIBLE_REPEAT_DIFFERENT_LISTING_ID",
        "Different listing_id with identical canonical fields; exact description match="
        + "recorded separately.",
    )
    suspicious |= possible_repeat.to_numpy()

    isolated_floor = frame["number_of_floors"].ge(80)
    _add_reason(
        reasons,
        evidence,
        isolated_floor,
        "ISOLATED_FLOOR_COUNT_NEEDS_REVIEW",
        "Observed floor count is >=80; dataset maximum is 135 while the next-highest is 63.",
    )
    suspicious |= isolated_floor.to_numpy()

    strata = frame["property_type"].isin(["Flat", "Apartment", "Condominium", "Service Residence"])
    unit_count_review = strata & (
        frame["total_units"].le(3) | frame["total_units"].ge(5_000)
    )
    _add_reason(
        reasons,
        evidence,
        unit_count_review,
        "STRATA_TOTAL_UNITS_NEEDS_REVIEW",
        "Optional total_units is <=3 or >=5,000 for a strata development; may refer to block/phase or be a source entry error.",
    )
    suspicious |= unit_count_review.to_numpy()

    supported_parking = pd.Series(
        [
            _description_supports_count(text, value, r"(?:car\s*parks?|parking(?:\s*lots?)?)")
            if pd.notna(value)
            else False
            for text, value in zip(normalized_description, frame["parking_lot"])
        ],
        index=frame.index,
    )
    parking_review = frame["parking_lot"].ge(5) & ~supported_parking
    _add_reason(
        reasons,
        evidence,
        parking_review,
        "UNSUPPORTED_HIGH_PARKING_COUNT",
        "Structured parking is >=5 and the description does not explicitly support the same count.",
    )
    suspicious |= parking_review.to_numpy()

    single_size_mismatch = np.zeros(count, dtype=bool)
    for position, values in enumerate(explicit_sizes):
        if len(values) == 1:
            relative = abs(values[0] - float(size.iloc[position])) / float(size.iloc[position])
            if relative > 0.10:
                single_size_mismatch[position] = True
    _add_reason(
        reasons,
        evidence,
        single_size_mismatch,
        "SINGLE_DESCRIPTION_SIZE_MISMATCH",
        "Only explicit description size differs from structured property_size_sqft by more than 10%.",
    )
    suspicious |= single_size_mismatch

    completion = pd.to_numeric(frame["completion_year"], errors="coerce")
    completion_review = completion.notna() & (
        ~completion.between(1900, 2030) | ((completion % 1) != 0)
    )
    _add_reason(
        reasons,
        evidence,
        completion_review,
        "OPTIONAL_COMPLETION_YEAR_NEEDS_REVIEW",
        "Optional completion year is outside the broad source-validity domain or non-integer; retain and verify/impute.",
    )
    suspicious |= completion_review.to_numpy()

    floors = pd.to_numeric(frame["number_of_floors"], errors="coerce")
    units = pd.to_numeric(frame["total_units"], errors="coerce")
    optional_domain_review = (
        (floors.notna() & (~floors.between(1, 200) | ((floors % 1) != 0)))
        | (units.notna() & (~units.between(1, 20_000) | ((units % 1) != 0)))
    )
    _add_reason(
        reasons,
        evidence,
        optional_domain_review,
        "OPTIONAL_FLOOR_OR_UNIT_DOMAIN_NEEDS_REVIEW",
        "Optional floor/unit count is outside the broad source-validity domain or non-integer; retain and verify/impute.",
    )
    suspicious |= optional_domain_review.to_numpy()

    # Clear corruption takes precedence over review flags.
    suspicious &= ~clear

    ppsf_lower = float(ppsf.quantile(0.005))
    ppsf_upper = float(ppsf.quantile(0.995))
    ppsf_extreme = ppsf.le(ppsf_lower) | ppsf.ge(ppsf_upper)
    premium_850 = frame["price"].gt(850_000)
    top5_threshold = float(frame["price"].quantile(0.95))
    top1_threshold = float(frame["price"].quantile(0.99))
    top5 = frame["price"].ge(top5_threshold)
    top1 = frame["price"].ge(top1_threshold)

    premium_regex_columns = [
        feature
        for group in ("layout", "private_luxury", "views", "position", "renovation_furnishing")
        for feature in REGEX_GROUPS[group]
    ]
    premium_text_count = regex[premium_regex_columns].sum(axis=1).astype(int)
    known_context = ~(
        _is_missing(frame["building_name"])
        & _is_missing(frame["city"])
        & _is_missing(frame["state"])
    )
    premium_context = (
        (frame["property_size_sqft"] >= frame["property_size_sqft"].quantile(0.90))
        | premium_text_count.gt(0)
        | known_context
    )
    physical_extreme = (
        frame["bedroom"].ge(6)
        | frame["bathroom"].ge(6)
        | frame["parking_lot"].ge(5)
        | frame["property_size_sqft"].lt(400)
        | frame["property_size_sqft"].gt(5_000)
        | frame["number_of_floors"].ge(60)
        | frame["total_units"].ge(3_600)
    )

    optional_missing_count = pd.DataFrame(
        {column: _is_missing(frame[column]) for column in OPTIONAL_FIELDS}
    ).sum(axis=1)
    validity_status = np.where(clear, "CLEARLY_INVALID", np.where(suspicious, "SUSPICIOUS", "VALID"))
    recommended_action = np.where(
        clear,
        "REMOVE_FROM_EXPERIMENT_ONLY",
        np.where(suspicious, "RETAIN_PENDING_REVIEW", "RETAIN"),
    )
    for position in range(count):
        if ppsf_extreme.iloc[position]:
            evidence[position].append(
                f"PPSF_DIAGNOSTIC_ONLY={ppsf.iloc[position]:.2f}; classified from independent evidence, never PPSF alone."
            )
            if not clear[position] and not suspicious[position]:
                reasons[position].append("VALID_BUT_EXTREME_PPSF_REVIEWED")
        if physical_extreme.iloc[position] and not clear[position] and not suspicious[position]:
            reasons[position].append("VALID_BUT_EXTREME_PHYSICAL_ATTRIBUTES_REVIEWED")
            evidence[position].append(
                "PHYSICAL_EXTREME_REVIEW: "
                f"size={size.iloc[position]:.2f}; bedrooms={frame['bedroom'].iloc[position]}; "
                f"bathrooms={frame['bathroom'].iloc[position]}; parking={frame['parking_lot'].iloc[position]}; "
                f"floors={frame['number_of_floors'].iloc[position]}; units={frame['total_units'].iloc[position]}; "
                "retained because no independent contradiction/corruption rule fired."
            )
        if premium_850.iloc[position]:
            evidence[position].append(
                "PREMIUM_REVIEW: price>RM850k; "
                f"top5={bool(top5.iloc[position])}; top1={bool(top1.iloc[position])}; "
                f"size={size.iloc[position]:.0f}; ppsf={ppsf.iloc[position]:.2f}; "
                f"premium_text_features={premium_text_count.iloc[position]}; known_location_or_building={bool(known_context.iloc[position])}."
            )
            if not clear[position] and not suspicious[position]:
                reasons[position].append("PREMIUM_PROPERTY_CONTEXT_REVIEWED_AND_RETAINED")
        if optional_missing_count.iloc[position] > 0:
            evidence[position].append(
                f"OPTIONAL_MISSING_COUNT={int(optional_missing_count.iloc[position])}; retained with existing fold-safe handling unless another independent rule applies."
            )
        if not reasons[position]:
            reasons[position].append("NO_INVALIDITY_OR_REVIEW_FLAG")
            evidence[position].append("Passed critical, physical-domain, contradiction, repeat, and parsing checks.")

    audit = pd.DataFrame(
        {
            "row_id": frame["listing_id"].astype(int),
            "price": frame["price"],
            "property_size_sqft": frame["property_size_sqft"],
            "ppsf": ppsf,
            "property_type": frame["property_type"],
            "building_name": frame["building_name"],
            "developer": frame["developer"],
            "city": frame["city"],
            "state": frame["state"],
            "bedroom": frame["bedroom"],
            "bathroom": frame["bathroom"],
            "parking_lot": frame["parking_lot"],
            "completion_year": frame["completion_year"],
            "number_of_floors": frame["number_of_floors"],
            "total_units": frame["total_units"],
            "validity_status": validity_status,
            "flag_reason": [";".join(items) for items in reasons],
            "recommended_action": recommended_action,
            "audit_evidence": [" | ".join(items) for items in evidence],
            "description": descriptions,
            "exact_duplicate_flag": exact_all.to_numpy(),
            "possible_repeat_flag": possible_repeat.to_numpy(),
            "possible_repeat_group": repeat_group,
            "possible_repeat_identical_description": same_description.to_numpy(),
            "premium_over_850k": premium_850.to_numpy(),
            "canonical_top5_flag": top5.to_numpy(),
            "canonical_top1_flag": top1.to_numpy(),
            "ppsf_extreme_diagnostic_flag": ppsf_extreme.to_numpy(),
            "valid_but_extreme_flag": (
                (validity_status == "VALID") & (ppsf_extreme.to_numpy() | physical_extreme.to_numpy())
            ),
            "premium_text_feature_count": premium_text_count.to_numpy(),
            "premium_context_supported": premium_context.to_numpy(),
            "optional_missing_count": optional_missing_count.to_numpy(int),
            "explicit_description_sizes": explicit_sizes.map(lambda values: json.dumps(values)),
            "explicit_description_prices": explicit_prices.map(lambda values: json.dumps(values)),
        }
    )
    diagnostics = {
        "exact_duplicate_rows": int(exact_all.sum()),
        "exact_duplicate_rows_removed": int(exact_later.sum()),
        "possible_repeat_rows": int(possible_repeat.sum()),
        "possible_repeat_groups": int(repeat_group.dropna().nunique()),
        "possible_repeat_identical_description_rows": int(same_description.sum()),
        "critical_invalid_price_rows": int(invalid_price.sum()),
        "critical_invalid_size_rows": int(invalid_size.sum()),
        "ppsf_lower_diagnostic_threshold": ppsf_lower,
        "ppsf_upper_diagnostic_threshold": ppsf_upper,
        "ppsf_extreme_rows": int(ppsf_extreme.sum()),
        "valid_but_extreme_rows": int(audit["valid_but_extreme_flag"].sum()),
        "premium_over_850k_rows": int(premium_850.sum()),
        "top5_threshold_rm": top5_threshold,
        "top5_rows": int(top5.sum()),
        "top1_threshold_rm": top1_threshold,
        "top1_rows": int(top1.sum()),
        "status_counts": audit["validity_status"].value_counts().to_dict(),
        "rule_counts": {
            rule: int(audit["flag_reason"].str.split(";").map(lambda values: rule in values).sum())
            for rule in RULES
        },
        "physical_profile": {
            column: {
                "missing": int(frame[column].isna().sum()),
                "minimum": float(frame[column].min()) if frame[column].notna().any() else None,
                "maximum": float(frame[column].max()) if frame[column].notna().any() else None,
                "non_integer": int(((frame[column].dropna() % 1) != 0).sum()),
            }
            for column in [
                "bedroom", "bathroom", "parking_lot", "property_size_sqft",
                "completion_year", "number_of_floors", "total_units",
            ]
        },
    }
    return audit, diagnostics


def metric_bundle(actual, predicted, top5_mask, p95_p99_mask, top1_mask) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    top5_mask = np.asarray(top5_mask, dtype=bool)
    p95_p99_mask = np.asarray(p95_p99_mask, dtype=bool)
    top1_mask = np.asarray(top1_mask, dtype=bool)
    error = predicted - actual

    def subset(mask):
        if not mask.any():
            return {"count": 0, "rmse_rm": None, "mae_rm": None}
        return {
            "count": int(mask.sum()),
            "rmse_rm": float(np.sqrt(np.mean(np.square(error[mask])))),
            "mae_rm": float(np.mean(np.abs(error[mask]))),
        }

    r2 = float(r2_score(actual, predicted))
    adjusted_r2 = 1.0 - (1.0 - r2) * (len(actual) - 1) / (len(actual) - PREDICTOR_COUNT - 1)
    return {
        "count": int(len(actual)),
        "rmse_rm": float(mean_squared_error(actual, predicted) ** 0.5),
        "mae_rm": float(mean_absolute_error(actual, predicted)),
        "r2": r2,
        "adjusted_r2": float(adjusted_r2),
        "median_ae_rm": float(np.median(np.abs(error))),
        "top5": subset(top5_mask),
        "p95_p99": subset(p95_p99_mask),
        "p99_p100": subset(top1_mask),
    }


def load_reference(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    reference = pd.read_csv(REFERENCE_OOF_PATH)
    reference = reference.loc[reference["variant"].eq(REFERENCE_VARIANT)].copy()
    reference = reference.set_index("row_id").loc[frame["listing_id"].astype(int)].reset_index()
    if len(reference) != len(frame) or reference["row_id"].nunique() != len(frame):
        raise AssertionError("Historical Position-regex OOF rows do not match canonical grain.")
    if not np.array_equal(reference["actual_price_RM"].to_numpy(float), frame["price"].to_numpy(float)):
        raise AssertionError("Historical OOF actual prices do not align to canonical rows.")
    return reference["predicted_price_RM"].to_numpy(float), reference["fold"].to_numpy(int)


def fit_cleaned(
    frame: pd.DataFrame,
    regex: pd.DataFrame,
    top5_threshold: float,
    top1_threshold: float,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    X = frame.drop(columns=["price"]).reset_index(drop=True)
    y = frame["price"].to_numpy(float)
    dense = regex[POSITION_FEATURES].reset_index(drop=True)
    folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(X))
    predictions = np.empty(len(frame), dtype=float)
    fold_assignment = np.empty(len(frame), dtype=int)
    rows = []
    for fold, (train_index, validation_index) in enumerate(folds, 1):
        output = fit_lightgbm_fold(
            X.iloc[train_index],
            y[train_index],
            X.iloc[validation_index],
            dense.loc[train_index],
            dense.loc[validation_index],
        )
        prediction = output["validation_prediction"]
        predictions[validation_index] = prediction
        fold_assignment[validation_index] = fold
        actual = y[validation_index]
        metrics = metric_bundle(
            actual,
            prediction,
            actual >= top5_threshold,
            (actual >= top5_threshold) & (actual < top1_threshold),
            actual >= top1_threshold,
        )
        rows.append({"variant": "B_clearly_invalid_removed", "fold": fold, "train_rows": len(train_index), "validation_rows": len(validation_index), **_flatten_metrics(metrics)})
    return predictions, fold_assignment, rows


def _flatten_metrics(metrics: dict) -> dict:
    return {
        "rmse_rm": metrics["rmse_rm"],
        "mae_rm": metrics["mae_rm"],
        "r2": metrics["r2"],
        "adjusted_r2": metrics["adjusted_r2"],
        "median_ae_rm": metrics["median_ae_rm"],
        "top5_count": metrics["top5"]["count"],
        "top5_rmse_rm": metrics["top5"]["rmse_rm"],
        "top5_mae_rm": metrics["top5"]["mae_rm"],
        "p95_p99_count": metrics["p95_p99"]["count"],
        "p95_p99_rmse_rm": metrics["p95_p99"]["rmse_rm"],
        "p99_p100_count": metrics["p99_p100"]["count"],
        "p99_p100_rmse_rm": metrics["p99_p100"]["rmse_rm"],
    }


def paired_bootstrap(actual, candidate, reference, draws=5_000, seed=42) -> dict:
    actual = np.asarray(actual, float)
    candidate_error = np.asarray(candidate, float) - actual
    reference_error = np.asarray(reference, float) - actual
    rng = np.random.default_rng(seed)
    rmse_difference = np.empty(draws)
    mae_difference = np.empty(draws)
    for draw in range(draws):
        selected = rng.integers(0, len(actual), size=len(actual))
        rmse_difference[draw] = (
            np.sqrt(np.mean(np.square(candidate_error[selected])))
            - np.sqrt(np.mean(np.square(reference_error[selected])))
        )
        mae_difference[draw] = (
            np.mean(np.abs(candidate_error[selected]))
            - np.mean(np.abs(reference_error[selected]))
        )
    return {
        "draws": draws,
        "seed": seed,
        "difference_definition": "retrained minus matched original; negative is better",
        "rmse_difference_rm": {
            "observed": float(np.sqrt(np.mean(np.square(candidate_error))) - np.sqrt(np.mean(np.square(reference_error)))),
            "ci95_lower": float(np.quantile(rmse_difference, 0.025)),
            "ci95_upper": float(np.quantile(rmse_difference, 0.975)),
        },
        "mae_difference_rm": {
            "observed": float(np.mean(np.abs(candidate_error)) - np.mean(np.abs(reference_error))),
            "ci95_lower": float(np.quantile(mae_difference, 0.025)),
            "ci95_upper": float(np.quantile(mae_difference, 0.975)),
        },
    }


def main() -> None:
    started = time.perf_counter()
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    before = _protected_snapshot()
    frame = pd.read_csv(DATA_PATH)
    if len(frame) != EXPECTED_ROWS or frame["listing_id"].nunique() != EXPECTED_ROWS:
        raise AssertionError("Canonical row grain changed.")
    descriptions, linkage = link_descriptions(RAW_PATH, frame["listing_id"])
    regex = extract_regex_features(descriptions)
    original_prediction, original_fold = load_reference(frame)

    audit, audit_diagnostics = build_audit(frame, descriptions, regex)
    clear_mask = audit["validity_status"].eq("CLEARLY_INVALID").to_numpy()
    retained_mask = ~clear_mask
    retained_indices = np.flatnonzero(retained_mask)
    cleaned = frame.iloc[retained_indices].reset_index(drop=True)
    cleaned_regex = regex.iloc[retained_indices].reset_index(drop=True)

    top5_threshold = audit_diagnostics["top5_threshold_rm"]
    top1_threshold = audit_diagnostics["top1_threshold_rm"]
    actual = frame["price"].to_numpy(float)
    top5 = actual >= top5_threshold
    top1 = actual >= top1_threshold
    p95_p99 = top5 & ~top1
    canonical_metrics = metric_bundle(actual, original_prediction, top5, p95_p99, top1)

    cleaned_prediction, cleaned_fold, cleaned_fold_rows = fit_cleaned(
        cleaned, cleaned_regex, top5_threshold, top1_threshold
    )
    cleaned_actual = cleaned["price"].to_numpy(float)
    cleaned_top5 = cleaned_actual >= top5_threshold
    cleaned_top1 = cleaned_actual >= top1_threshold
    cleaned_p95_p99 = cleaned_top5 & ~cleaned_top1
    cleaned_metrics = metric_bundle(
        cleaned_actual, cleaned_prediction, cleaned_top5, cleaned_p95_p99, cleaned_top1
    )
    matched_original = original_prediction[retained_indices]
    matched_metrics = metric_bundle(
        cleaned_actual, matched_original, cleaned_top5, cleaned_p95_p99, cleaned_top1
    )
    rmse_gain = matched_metrics["rmse_rm"] - cleaned_metrics["rmse_rm"]
    mae_gain = matched_metrics["mae_rm"] - cleaned_metrics["mae_rm"]

    canonical_fold_rows = []
    for fold in range(1, 6):
        mask = original_fold == fold
        fold_actual = actual[mask]
        metrics = metric_bundle(
            fold_actual,
            original_prediction[mask],
            fold_actual >= top5_threshold,
            (fold_actual >= top5_threshold) & (fold_actual < top1_threshold),
            fold_actual >= top1_threshold,
        )
        canonical_fold_rows.append(
            {
                "variant": "A_current_canonical",
                "fold": fold,
                "train_rows": int(len(frame) - mask.sum()),
                "validation_rows": int(mask.sum()),
                **_flatten_metrics(metrics),
            }
        )
    fold_frame = pd.DataFrame([*canonical_fold_rows, *cleaned_fold_rows])

    comparison = pd.DataFrame(
        [
            {"variant": "A_current_canonical", "comparison_role": "headline", "rows": len(frame), **_flatten_metrics(canonical_metrics)},
            {"variant": "B_clearly_invalid_removed", "comparison_role": "headline", "rows": len(cleaned), **_flatten_metrics(cleaned_metrics)},
            {"variant": "A_original_on_B_retained", "comparison_role": "mandatory_matched_baseline", "rows": len(cleaned), **_flatten_metrics(matched_metrics)},
        ]
    )

    audit["original_oof_prediction_rm"] = original_prediction
    audit["original_oof_absolute_error_rm"] = np.abs(original_prediction - actual)
    audit["original_oof_fold"] = original_fold
    audit.to_csv(EXPERIMENT / "validity_audit.csv", index=False)
    invalid = audit.loc[clear_mask].copy()
    suspicious_rows = audit.loc[audit["validity_status"].eq("SUSPICIOUS")].copy()
    invalid.to_csv(EXPERIMENT / "clearly_invalid_rows.csv", index=False)
    suspicious_rows.to_csv(EXPERIMENT / "suspicious_rows.csv", index=False)
    comparison.to_csv(EXPERIMENT / "model_comparison.csv", index=False)
    fold_frame.to_csv(EXPERIMENT / "fold_metrics.csv", index=False)

    canonical_oof = pd.DataFrame(
        {
            "variant": "A_current_canonical",
            "canonical_row_index": np.arange(len(frame)),
            "row_id": frame["listing_id"].astype(int),
            "fold": original_fold,
            "actual_price_rm": actual,
            "predicted_price_rm": original_prediction,
            "residual_rm": original_prediction - actual,
            "absolute_error_rm": np.abs(original_prediction - actual),
            "retained_after_audit": retained_mask,
        }
    )
    cleaned_oof = pd.DataFrame(
        {
            "variant": "B_clearly_invalid_removed",
            "canonical_row_index": retained_indices,
            "row_id": cleaned["listing_id"].astype(int),
            "fold": cleaned_fold,
            "actual_price_rm": cleaned_actual,
            "predicted_price_rm": cleaned_prediction,
            "residual_rm": cleaned_prediction - cleaned_actual,
            "absolute_error_rm": np.abs(cleaned_prediction - cleaned_actual),
            "retained_after_audit": True,
        }
    )
    oof_frame = pd.concat([canonical_oof, cleaned_oof], ignore_index=True)
    oof_frame.to_csv(EXPERIMENT / "oof_predictions.csv", index=False)

    if len(invalid):
        removed_actual = invalid["price"].to_numpy(float)
        removed_prediction = invalid["original_oof_prediction_rm"].to_numpy(float)
        removed_diagnostics = {
            "count": len(invalid),
            "rmse_rm": float(mean_squared_error(removed_actual, removed_prediction) ** 0.5),
            "mae_rm": float(mean_absolute_error(removed_actual, removed_prediction)),
            "rows": invalid[
                [
                    "row_id", "price", "property_size_sqft", "ppsf",
                    "original_oof_prediction_rm", "original_oof_absolute_error_rm",
                    "flag_reason", "audit_evidence",
                ]
            ].to_dict("records"),
        }
    else:
        removed_diagnostics = {"count": 0, "rmse_rm": None, "mae_rm": None, "rows": []}

    bootstrap = None
    if rmse_gain > 0 and mae_gain > 0:
        bootstrap = paired_bootstrap(
            cleaned_actual, cleaned_prediction, matched_original, draws=5_000, seed=42
        )

    flattened_canonical = _flatten_metrics(canonical_metrics)
    expected_checks = {
        key: bool(math.isclose(flattened_canonical[key], value, abs_tol=1e-6, rel_tol=0.0))
        for key, value in EXPECTED_REFERENCE.items()
    }
    premium_removed = int((invalid["price"] > 850_000).sum())
    top5_removed = int(invalid["canonical_top5_flag"].sum())
    top1_removed = int(invalid["canonical_top1_flag"].sum())

    after = _protected_snapshot()
    changed = sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
    removal_reasons = sorted(
        {
            reason
            for value in invalid["flag_reason"]
            for reason in str(value).split(";")
            if reason
        }
    )
    results = {
        "question": "Can defensibly invalid/corrupted rows be removed without deleting legitimate difficult properties, and does retraining improve matched-row performance?",
        "audit_scope": {
            "dataset": DATA_PATH.relative_to(ROOT).as_posix(),
            "canonical_rows": len(frame),
            "grain": "one unique canonical listing_id",
            "dataset_sha256": _sha256(DATA_PATH),
            "raw_description_sha256": _sha256(RAW_PATH),
            "reference_oof_sha256": _sha256(REFERENCE_OOF_PATH),
            "description_linkage": linkage,
            "rules_optimized_for_model_metrics": False,
            "price_percentile_used_as_deletion_rule": False,
            "ppsf_used_as_deletion_rule": False,
            "optional_missingness_used_as_deletion_rule": False,
        },
        "validity_rules": RULES,
        "audit_results": audit_diagnostics,
        "classification_counts": audit["validity_status"].value_counts().to_dict(),
        "exact_duplicates": {
            "rows_found": audit_diagnostics["exact_duplicate_rows"],
            "rows_removed_after_first": audit_diagnostics["exact_duplicate_rows_removed"],
        },
        "possible_repeats": {
            "rows": audit_diagnostics["possible_repeat_rows"],
            "groups": audit_diagnostics["possible_repeat_groups"],
            "identical_description_rows": audit_diagnostics["possible_repeat_identical_description_rows"],
            "automatic_deletion": False,
            "group_members": (
                audit.loc[audit["possible_repeat_flag"]]
                .groupby("possible_repeat_group")["row_id"]
                .apply(list)
                .to_dict()
            ),
        },
        "cleaned_experimental_variant": {
            "original_rows": len(frame),
            "clearly_invalid_rows": int(clear_mask.sum()),
            "rows_retained": int(retained_mask.sum()),
            "retention_pct": float(retained_mask.mean() * 100.0),
            "removal_reasons": removal_reasons,
            "premium_over_850k_removed": premium_removed,
            "top5_removed": top5_removed,
            "top1_removed": top1_removed,
            "replacement_dataset_created": False,
        },
        "model": {
            "configuration": "exact existing Position-regex LightGBM PPSF model",
            "position_regex_features": POSITION_FEATURES,
            "predictor_count": PREDICTOR_COUNT,
            "cv": {"class": "KFold", "n_splits": 5, "shuffle": True, "random_state": 42},
            "hyperparameters_retuned": False,
            "canonical": canonical_metrics,
            "cleaned_retrained": cleaned_metrics,
            "original_on_retained_rows": matched_metrics,
            "retraining_gain": {
                "definition": "matched original minus retrained; positive is better",
                "rmse_gain_rm": rmse_gain,
                "mae_gain_rm": mae_gain,
                "improved_both": bool(rmse_gain > 0 and mae_gain > 0),
            },
            "reference_metrics_reproduced": all(expected_checks.values()),
            "reference_checks": expected_checks,
        },
        "removed_row_diagnostics": removed_diagnostics,
        "bootstrap": {
            "eligibility_rule": "run only if retraining improves both matched-row RMSE and MAE",
            "eligible": bool(rmse_gain > 0 and mae_gain > 0),
            "result": bootstrap,
        },
        "premium_safety": {
            "over_850k_rows_reviewed": audit_diagnostics["premium_over_850k_rows"],
            "top5_rows_reviewed": audit_diagnostics["top5_rows"],
            "top1_rows_reviewed": audit_diagnostics["top1_rows"],
            "over_850k_removed": premium_removed,
            "top5_removed": top5_removed,
            "top1_removed": top1_removed,
            "removal_never_based_only_on_price_ppsf_or_premium_membership": True,
        },
        "production_safety": {
            "protected_file_count": len(before),
            "all_protected_files_unchanged": before == after,
            "changed_protected_files": changed,
            "before_manifest_sha256": _manifest_digest(before),
            "after_manifest_sha256": _manifest_digest(after),
        },
        "internal_tests": {
            "canonical_grain": len(frame) == EXPECTED_ROWS and frame["listing_id"].nunique() == EXPECTED_ROWS,
            "required_audit_columns_present": set(AUDIT_COLUMNS).issubset(audit.columns),
            "allowed_statuses_only": set(audit["validity_status"]).issubset({"VALID", "SUSPICIOUS", "CLEARLY_INVALID"}),
            "no_price_only_removal": all("PRICE" not in reasons or reasons == "CRITICAL_INVALID_PRICE" for reasons in removal_reasons),
            "no_ppsf_removal_rule": all("PPSF" not in reason for reason in removal_reasons),
            "no_optional_missing_removal_rule": all("OPTIONAL" not in reason and "MISSING" not in reason for reason in removal_reasons),
            "every_removed_row_has_reason": bool(invalid["flag_reason"].ne("").all()),
            "oof_complete_unique": bool(oof_frame.groupby("variant")["row_id"].nunique().eq(oof_frame.groupby("variant").size()).all()),
            "reference_metrics_reproduced": all(expected_checks.values()),
            "protected_files_unchanged": before == after,
            "all_passed": False,
        },
        "decision": {
            "genuinely_invalid_rows_found": int(clear_mask.sum()),
            "removal_genuinely_improved_both_metrics": bool(rmse_gain > 0 and mae_gain > 0),
            "recommendation": (
                "Remove only the documented corrupt observations from future canonical rebuilds after source-owner review; retain suspicious, premium, PPSF-extreme, repeated-looking, and incomplete listings."
                if clear_mask.any()
                else "Retain the canonical dataset unchanged because no row met a defensible clearly-invalid rule."
            ),
        },
        "runtime_seconds": time.perf_counter() - started,
        "artifacts": [
            "experiments/data_validity_audit/results.json",
            "experiments/data_validity_audit/validity_audit.csv",
            "experiments/data_validity_audit/clearly_invalid_rows.csv",
            "experiments/data_validity_audit/suspicious_rows.csv",
            "experiments/data_validity_audit/model_comparison.csv",
            "experiments/data_validity_audit/fold_metrics.csv",
            "experiments/data_validity_audit/oof_predictions.csv",
            "experiments/data_validity_audit/run_experiment.py",
            "experiments/data_validity_audit/test_invariants.py",
        ],
    }
    results["internal_tests"]["all_passed"] = all(
        value for key, value in results["internal_tests"].items() if key != "all_passed"
    )
    with (EXPERIMENT / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_clean(results), handle, indent=2, allow_nan=False)
    print(
        f"Completed validity audit: clear={clear_mask.sum()}, suspicious={len(suspicious_rows)}, "
        f"retained={retained_mask.sum()}, RMSE gain={rmse_gain:,.2f}, MAE gain={mae_gain:,.2f}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
