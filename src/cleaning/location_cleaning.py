"""Extract standardized Malaysian state and city labels from addresses."""

import re

import pandas as pd


# Order matters because longer multi-word state names must be recognized reliably.
STATES = [
    "Kuala Lumpur",
    "Negeri Sembilan",
    "Pulau Pinang",
    "Penang",
    "Malacca",
    "Melaka",
    "Selangor",
    "Johor",
    "Sabah",
    "Sarawak",
    "Putrajaya",
    "Perak",
    "Pahang",
    "Kedah",
    "Terengganu",
    "Kelantan",
    "Labuan",
]

# Preserve the established spelling aliases used by the strongest city experiment.
CITY_ALIASES = {
    "air itam": "Ayer Itam",
    "ayer itam": "Ayer Itam",
    "georgetown": "George Town",
    "george town": "George Town",
}


def extract_state(address: object) -> str:
    """Return a normalized Malaysian state found in an address, or ``Unknown``."""
    text = str(address) if pd.notna(address) else ""
    for state in STATES:
        if re.search(rf"\b{re.escape(state)}\b", text, flags=re.IGNORECASE):
            return {"Pulau Pinang": "Penang", "Malacca": "Melaka"}.get(state, state)
    return "Unknown"


def extract_city(address: object) -> str:
    """Abstract the established reusable city/locality label from a full address."""
    if pd.isna(address) or not str(address).strip():
        return "Unknown"
    text = " ".join(str(address).split())
    state = extract_state(text)
    if state in {"Kuala Lumpur", "Putrajaya", "Labuan"}:
        return state
    parts = [part.strip() for part in text.split(",") if part.strip()]
    state_names = {state_name.lower() for state_name in STATES}
    candidates = [
        part
        for part in parts
        if part.lower() not in state_names and not part.isdigit()
    ]
    if not candidates:
        return state
    city = candidates[-1]
    if len(parts) <= 2 and city.lower().startswith(
        ("jalan ", "jln ", "lorong ", "persiaran ", "off ")
    ):
        return state
    return CITY_ALIASES.get(city.lower(), city)

