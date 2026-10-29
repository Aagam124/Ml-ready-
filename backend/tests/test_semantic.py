import numpy as np
import pandas as pd
import pytest

from app.profiling.semantic import infer_semantic_type


def kind(values, name="col"):
    return infer_semantic_type(pd.Series(values), name)


def test_tenure_years_is_a_number_not_a_date():
    # Regression: this was flagged as "temporal" leakage because the NAME contains "year".
    s = pd.Series(np.random.default_rng(0).integers(1, 121, 500) / 12)
    assert infer_semantic_type(s, "tenure_years").type == "continuous"


@pytest.mark.parametrize("name", ["tenure_months", "start_amount", "end_balance", "day_count", "time_spent"])
def test_date_like_names_with_numeric_values_are_never_datetime(name):
    s = pd.Series(np.random.default_rng(1).integers(0, 500, 300))
    assert infer_semantic_type(s, name).type != "datetime"


def test_year_named_integer_column_is_not_datetime():
    assert kind(np.random.default_rng(2).integers(2000, 2024, 300), "year").type != "datetime"


def test_real_date_strings_are_datetime():
    r = kind(pd.date_range("2024-01-01", periods=100).strftime("%Y-%m-%d"))
    assert r.type == "datetime" and "stored_as_text" in r.flags


def test_native_datetime_dtype():
    assert kind(pd.date_range("2024-01-01", periods=50)).type == "datetime"


def test_identifier_by_name_and_uniqueness():
    assert kind(range(1000, 1500), "customer_id").type == "identifier"


def test_sequential_integers_are_identifiers_without_name_hint():
    assert kind(range(1, 301), "row").type == "identifier"


def test_unique_integers_without_hint_or_sequence_are_not_identifiers():
    vals = np.random.default_rng(3).choice(10**6, 300, replace=False)
    assert kind(vals, "population").type != "identifier"


def test_string_identifiers():
    assert kind([f"CUST{i}" for i in range(200)], "customer_id").type == "identifier"


def test_binary_variants():
    assert kind([0, 1] * 50).type == "binary"
    assert kind(["yes", "no"] * 50).type == "binary"
    assert kind([True, False] * 50).type == "binary"


def test_categorical_strings():
    assert kind(np.random.default_rng(4).choice(list("ABCDE"), 300)).type == "categorical"


def test_low_cardinality_integers_are_categorical_codes():
    assert kind(np.random.default_rng(5).integers(1, 6, 300), "rating").type == "categorical"


def test_count_and_continuous():
    assert kind(np.random.default_rng(6).integers(0, 120, 400), "visits").type == "count"
    assert kind(np.random.default_rng(7).normal(50, 10, 400), "score").type == "continuous"


def test_email_url_phone():
    assert kind([f"u{i}@example.com" for i in range(50)]).type == "email"
    assert kind([f"https://site{i}.com/p" for i in range(50)]).type == "url"
    assert kind([f"+91 98765 {10000 + i}" for i in range(50)]).type == "phone"


def test_currency_and_percentage_strings():
    assert kind([f"${i * 10}.50" for i in range(1, 60)]).type == "currency"
    assert kind([f"{i}%" for i in range(1, 60)]).type == "percentage"


def test_numbers_stored_as_text_are_flagged():
    r = kind([str(x) for x in np.random.default_rng(8).normal(10, 2, 200).round(2)])
    assert r.type == "continuous" and "stored_as_text" in r.flags


def test_leading_zero_codes_not_treated_as_numbers():
    r = kind(["00123", "00456", "00789", "01234"] * 20)
    assert r.type != "continuous" and "leading_zeros" in r.flags


def test_free_text():
    texts = [f"this is a fairly long free text sentence number {i} about something" for i in range(100)]
    assert kind(texts).type == "text"


def test_constant_and_empty():
    assert kind(["IN"] * 50, "country").type == "constant"
    assert kind([np.nan] * 10).type == "empty"


def test_mixed_types_flagged():
    r = kind(["10", "20", "abc", "30", "def", "40", "xyz"] * 10)
    assert "mixed_types" in r.flags
