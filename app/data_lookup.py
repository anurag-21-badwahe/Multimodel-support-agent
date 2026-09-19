"""Strict Pandas-based synthetic knowledge-base lookup.

There is deliberately no vector database, embedding model, or semantic search
layer here. The graph only receives evidence from exact CSV matches.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CSV_SCHEMAS: dict[str, list[str]] = {
    "error_codes.csv": [
        "error_code",
        "meaning",
        "possible_cause",
        "resolution",
        "related_component",
    ],
    "components.csv": [
        "component_id",
        "component_name",
        "purpose",
        "location",
        "related_error",
    ],
    "troubleshooting.csv": [
        "error_code",
        "step_number",
        "instruction",
        "safety_note",
    ],
}


def _load_and_validate_csv(filename: str) -> pd.DataFrame:
    """Load one CSV and validate its required schema and basic integrity."""

    expected_columns = CSV_SCHEMAS[filename]
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Required synthetic dataset is missing: {path}")

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [column for column in expected_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{filename} is missing required columns: {missing}")

    # Keep the public shape predictable even if extra columns are added later.
    frame = frame[expected_columns].copy()

    for column in expected_columns:
        frame[column] = frame[column].astype(str).str.strip()

    if frame.empty:
        raise ValueError(f"{filename} must contain at least one record")

    if filename == "error_codes.csv":
        if frame["error_code"].duplicated().any():
            duplicated = sorted(frame.loc[frame["error_code"].duplicated(), "error_code"].unique())
            raise ValueError(f"error_codes.csv contains duplicate error codes: {duplicated}")
    elif filename == "troubleshooting.csv":
        if frame[["error_code", "step_number"]].duplicated().any():
            raise ValueError("troubleshooting.csv contains duplicate error_code + step_number rows")

    return frame


@lru_cache(maxsize=1)
def load_datasets() -> dict[str, pd.DataFrame]:
    """Load and validate all three datasets once per process.

    ``lru_cache`` keeps repeated chat requests cheap while retaining deterministic
    behavior. Restarting the service reloads the files, which is appropriate for
    this small synthetic knowledge base.
    """

    return {filename: _load_and_validate_csv(filename) for filename in CSV_SCHEMAS}


def get_error_catalog() -> list[dict[str, str]]:
    """Return the small code/meaning catalog used by the extraction prompt.

    This is still a Pandas table lookup; no semantic index is built.
    """

    df = load_datasets()["error_codes.csv"]
    columns = ["error_code", "meaning", "related_component"]
    return df[columns].to_dict(orient="records")


def lookup_by_error_code(error_code: str) -> dict[str, Any]:
    """Perform exact-match lookup across the three synthetic CSVs.

    ``error_code`` is not lowercased or otherwise transformed. Whitespace around
    the value is removed, but the stored code itself is preserved exactly.
    """

    code = (error_code or "").strip()
    if not code:
        return {
            "error_code": None,
            "found": False,
            "error_code_record": None,
            "component_records": [],
            "troubleshooting_records": [],
            "sources": [],
        }

    datasets = load_datasets()

    error_df = datasets["error_codes.csv"]
    component_df = datasets["components.csv"]
    troubleshooting_df = datasets["troubleshooting.csv"]

    error_match = error_df[error_df["error_code"] == code]
    component_match = component_df[component_df["related_error"] == code]
    troubleshooting_match = troubleshooting_df[troubleshooting_df["error_code"] == code].copy()

    if not troubleshooting_match.empty:
        # Sort numerically where possible so "10" comes after "2".
        troubleshooting_match["_step_num"] = pd.to_numeric(
            troubleshooting_match["step_number"], errors="coerce"
        )
        troubleshooting_match = (
            troubleshooting_match.sort_values(["_step_num", "step_number"])
            .drop(columns=["_step_num"])
        )

    error_record = error_match.iloc[0].to_dict() if not error_match.empty else None

    # Evidence is considered present only when the primary error-code record exists.
    # Related tables enrich the answer but can never manufacture a supported error.
    found = error_record is not None

    sources: list[dict[str, Any]] = []
    if error_record is not None:
        sources.append(
            {
                "file": "error_codes.csv",
                "record": code,
                "match_type": "exact error_code match",
            }
        )
    if not component_match.empty:
        sources.append(
            {
                "file": "components.csv",
                "record": code,
                "match_type": "exact related_error match",
            }
        )
    if not troubleshooting_match.empty:
        sources.append(
            {
                "file": "troubleshooting.csv",
                "record": code,
                "match_type": "exact error_code match",
            }
        )

    return {
        "error_code": code,
        "found": found,
        "error_code_record": error_record,
        "component_records": component_match.to_dict(orient="records"),
        "troubleshooting_records": troubleshooting_match.to_dict(orient="records"),
        "sources": sources,
    }
