from __future__ import annotations

from typing import Any

import pandas as pd


# ============================================================
# Common missing-value placeholders
# ============================================================

MISSING_PLACEHOLDERS = {
    "",
    "na",
    "n/a",
    "nan",
    "null",
    "none",
    "missing",
    "unknown",
    "?",
    "-",
    "--",
}


# ============================================================
# Helper
# ============================================================

def _is_missing_placeholder(value: Any) -> bool:
    """
    Check whether a value represents a missing value.

    This handles both actual NaN/None values and common
    textual representations such as 'N/A', 'null', '?' etc.
    """

    if pd.isna(value):
        return True

    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in MISSING_PLACEHOLDERS

    return False


# ============================================================
# Numeric consistency
# ============================================================

def analyze_numeric_consistency(
    df: pd.DataFrame,
    column: str,
) -> dict[str, Any]:
    """
    Analyze whether values in a column can be interpreted
    as numeric.

    Detects cases such as:

        21
        22
        25
        abc
        27

    where 'abc' is a non-numeric anomaly.
    """

    series = df[column]

    # Remove missing values and common missing placeholders
    non_missing = series[
        ~series.map(_is_missing_placeholder)
    ]

    if len(non_missing) == 0:
        return {
            "column": column,
            "status": "no_data",
            "numeric_percentage": 0.0,
            "non_numeric_count": 0,
            "non_numeric_percentage": 0.0,
            "examples": [],
        }

    # Try converting every value to numeric
    converted = pd.to_numeric(
        non_missing,
        errors="coerce",
    )

    numeric_mask = converted.notna()

    numeric_count = int(
        numeric_mask.sum()
    )

    non_numeric_count = int(
        (~numeric_mask).sum()
    )

    total = len(non_missing)

    numeric_percentage = (
        numeric_count / total * 100
    )

    non_numeric_percentage = (
        non_numeric_count / total * 100
    )

    # Examples of values that could not be converted
    non_numeric_values = (
        non_missing[~numeric_mask]
        .astype(str)
        .drop_duplicates()
        .head(10)
        .tolist()
    )

    # Classify the column
    if numeric_percentage == 100:
        status = "numeric"

    elif numeric_percentage >= 95:
        status = "mostly_numeric_with_anomalies"

    elif numeric_percentage >= 80:
        status = "mixed_type"

    else:
        status = "not_numeric"

    return {
        "column": column,
        "status": status,
        "total_non_missing": total,
        "numeric_count": numeric_count,
        "numeric_percentage": round(
            numeric_percentage,
            2,
        ),
        "non_numeric_count": non_numeric_count,
        "non_numeric_percentage": round(
            non_numeric_percentage,
            2,
        ),
        "examples": non_numeric_values,
    }


# ============================================================
# String pattern analysis
# ============================================================

def analyze_string_patterns(
    df: pd.DataFrame,
    column: str,
) -> dict[str, Any]:
    """
    Analyze patterns inside text/string columns.

    This function is intended for text-like columns.

    Examples:

        Aakash
        Rahul
        Priya
        123Amit

    The last value can be detected as a potential
    pattern anomaly.
    """

    series = df[column]

    values = series[
        ~series.map(_is_missing_placeholder)
    ].astype(str)

    if len(values) == 0:
        return {
            "column": column,
            "status": "no_data",
            "total_values": 0,
            "starts_with_digit": 0,
            "starts_with_digit_percentage": 0.0,
            "contains_digit": 0,
            "contains_digit_percentage": 0.0,
            "contains_special_character": 0,
            "contains_special_character_percentage": 0.0,
            "leading_whitespace": 0,
            "trailing_whitespace": 0,
        }

    # Values beginning with a number
    starts_with_digit = int(
        values.str.match(r"^\d").sum()
    )

    # Values containing at least one number
    contains_digit = int(
        values.str.contains(
            r"\d",
            regex=True,
        ).sum()
    )

    # Values containing symbols other than
    # letters, numbers, or whitespace
    contains_special_character = int(
        values.str.contains(
            r"[^A-Za-z0-9\s]",
            regex=True,
        ).sum()
    )

    # Values containing leading whitespace
    leading_whitespace = int(
        values.str.match(r"^\s").sum()
    )

    # Values containing trailing whitespace
    trailing_whitespace = int(
        values.str.match(r".*\s$").sum()
    )

    total = len(values)

    return {
        "column": column,
        "status": "analyzed",
        "total_values": total,

        "starts_with_digit": starts_with_digit,
        "starts_with_digit_percentage": round(
            starts_with_digit / total * 100,
            2,
        ),

        "contains_digit": contains_digit,
        "contains_digit_percentage": round(
            contains_digit / total * 100,
            2,
        ),

        "contains_special_character": contains_special_character,
        "contains_special_character_percentage": round(
            contains_special_character / total * 100,
            2,
        ),

        "leading_whitespace": leading_whitespace,
        "trailing_whitespace": trailing_whitespace,
    }


# ============================================================
# Missing-value placeholder analysis
# ============================================================

def analyze_missing_placeholders(
    df: pd.DataFrame,
    column: str,
) -> dict[str, Any]:
    """
    Detect textual representations of missing values.

    Examples:

        N/A
        NA
        null
        None
        ?
        -
        unknown
    """

    series = df[column]

    placeholder_counts: dict[str, int] = {}

    for value in MISSING_PLACEHOLDERS:

        if value == "":
            mask = (
                series.astype(str)
                .str.strip()
                == ""
            )

        else:
            mask = (
                series.astype(str)
                .str.strip()
                .str.lower()
                == value
            )

        count = int(mask.sum())

        if count > 0:
            placeholder_counts[value] = count

    total = sum(
        placeholder_counts.values()
    )

    return {
        "column": column,
        "placeholder_count": total,
        "placeholders": placeholder_counts,
    }


# ============================================================
# Complete data consistency analysis
# ============================================================

def run_data_consistency_analysis(
    df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Run all data consistency checks.

    Checks:

    1. Numeric consistency
    2. String patterns
    3. Missing-value placeholders
    """

    results = {
        "summary": {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
        },
        "numeric_consistency": {},
        "string_patterns": {},
        "missing_placeholders": {},
    }

    for column in df.columns:

        # ----------------------------------------------------
        # Numeric consistency
        # ----------------------------------------------------

        numeric_check = analyze_numeric_consistency(
            df,
            column,
        )

        results["numeric_consistency"][column] = (
            numeric_check
        )

        # ----------------------------------------------------
        # String pattern analysis
        # ----------------------------------------------------
        #
        # Only analyze string patterns when the column
        # is not predominantly numeric.
        #
        # This prevents false positives such as:
        #
        # Age = 21
        #
        # being treated as a string-pattern anomaly.
        # ----------------------------------------------------

        is_text_column = (
            pd.api.types.is_object_dtype(df[column])
            or pd.api.types.is_string_dtype(df[column])
        )

        if is_text_column:

            if numeric_check["numeric_percentage"] < 80:

                results["string_patterns"][column] = (
                    analyze_string_patterns(
                        df,
                        column,
                    )
                )

        # ----------------------------------------------------
        # Missing placeholders
        # ----------------------------------------------------

        results["missing_placeholders"][column] = (
            analyze_missing_placeholders(
                df,
                column,
            )
        )

    return results