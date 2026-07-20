"""Tests for local tokenizer counting. No API calls."""

import pytest

from planner.tokenizer import (
    TOKENS_PER_MESSAGE,
    TOKENS_PER_REPLY_PRIMING,
    compare,
    count_messages,
    count_text,
    encoding_for,
)


# --- Encoding selection ------------------------------------------------------


def test_known_model_resolves_an_encoding():
    assert encoding_for("gpt-4o-mini") is not None


def test_unknown_model_falls_back_instead_of_raising():
    """A future model name must not crash cost estimation."""
    assert encoding_for("some-model-that-does-not-exist-yet") is not None


# --- Counting ----------------------------------------------------------------


def test_counting_is_deterministic():
    assert count_text("hello world") == count_text("hello world")


def test_empty_text_is_zero_tokens():
    assert count_text("") == 0


def test_longer_text_costs_more_tokens():
    assert count_text("hello world " * 50) > count_text("hello world")


def test_a_token_is_not_a_character():
    """Guards the common misconception that tokens and characters are the same."""
    text = "internationalisation"
    assert 0 < count_text(text) < len(text)


def test_messages_include_per_message_overhead():
    """The role string is encoded too, not just the content."""
    text = "hello"
    expected = (
        count_text(text)
        + count_text("user")
        + TOKENS_PER_MESSAGE
        + TOKENS_PER_REPLY_PRIMING
    )
    assert count_messages([{"role": "user", "content": text}]) == expected


def test_more_messages_cost_more():
    one = count_messages([{"role": "user", "content": "hi"}])
    two = count_messages(
        [{"role": "system", "content": "hi"}, {"role": "user", "content": "hi"}]
    )
    assert two > one


def test_non_string_values_are_skipped_not_crashed():
    assert count_messages([{"role": "user", "content": "hi", "extra": 42}]) > 0


# --- Comparison --------------------------------------------------------------


def test_api_billing_more_than_local_is_reported_as_positive_delta():
    c = compare(local_estimate=1000, api_reported=1338)
    assert c.delta == 338
    assert c.delta_pct == pytest.approx(33.8)
    assert "schema" in c.explanation


def test_local_higher_than_billed_is_explained_too():
    c = compare(local_estimate=1500, api_reported=1338)
    assert c.delta == -162
    assert "caching" in c.explanation


def test_exact_agreement_is_stated_plainly():
    assert "agree exactly" in compare(1338, 1338).explanation


def test_zero_local_estimate_has_no_percentage():
    assert compare(0, 100).delta_pct is None
