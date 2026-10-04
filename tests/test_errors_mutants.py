"""Mutation-testing follow-up for `glide/providers/errors.py` (see docs/MUTATION_REPORT.md).

The error vocabulary is what the chain decides from, and what a user reads when a provider fails. These tests pin
what the first mutation run left unchecked: a cancel passes through unchanged, a spent account is told in fixed
words that never repeat the server's reply, every message names the provider (or says "provider"), a hostile
`Retry-After` never escapes as a bare `ValueError`, and the snippet of a reply is bounded.
"""

from __future__ import annotations

import httpx
import pytest

from glide.providers.errors import (
    CANCELLED,
    SNIPPET,
    AllProvidersFailed,
    ProviderError,
    cancelled,
    from_exception,
    from_status,
    snippet,
    spent,
)


def test_no_provider_tried_is_said_in_the_message():
    assert str(AllProvidersFailed("llm", [])) == "every llm provider failed (none was available)"
    tried = [("a", ProviderError("x", kind="timeout")), ("b", ProviderError("y", kind="auth"))]
    assert str(AllProvidersFailed("llm", tried)) == "every llm provider failed (a: timeout; b: auth)"


def test_a_snippet_is_one_line_and_bounded():
    assert len(snippet("x" * 1000)) == SNIPPET == 300
    assert snippet("a\n  b\t c") == "a b c"
    assert snippet("abc") == "abc"  # a str is shown as it is, not as its repr
    assert snippet({"a": 1}) == "{'a': 1}"  # anything else by its repr


@pytest.mark.parametrize("status", [400, 402, 403, 429])
def test_a_spent_account_is_recognised_by_status_and_by_its_codes(status):
    assert spent(status, "billing_hard_limit_reached" if status != 402 else "")
    assert spent(status, {"error": {"code": "insufficient_quota"}}) is True  # a parsed body is read as well


@pytest.mark.parametrize("status", [200, 401, 404, 500, 503])
def test_other_statuses_are_not_a_spent_account_even_when_the_body_names_the_code(status):
    assert spent(status, "insufficient_quota") is False


@pytest.mark.parametrize("status", [400, 402, 403, 429])
def test_a_spent_account_is_told_in_fixed_words_that_never_repeat_the_reply(status):
    error = from_status(status, "insufficient_quota sk-should-never-be-shown", provider="openai")
    assert error.kind == "auth" and error.status == status and error.provider == "openai"
    assert str(error).startswith("openai has no credit or quota left")
    assert "sk-should-never-be-shown" not in str(error) and "insufficient_quota" not in str(error)
    assert from_status(status, "insufficient_quota").args[0].startswith("provider has no credit")


def test_a_refusal_that_is_not_about_credit_keeps_a_short_snippet_and_names_its_provider():
    error = from_status(500, "boom", provider="p")
    assert str(error) == "p answered 500: boom"
    assert str(from_status(500, "boom")) == "provider answered 500: boom"
    assert str(from_status(502, "")) == "provider answered 502"  # no dangling colon when there is no reply


def test_a_provider_error_passes_through_from_exception_unchanged():
    """A cancel the adapter already made must never be re-labelled as a transport fault."""
    own = cancelled("p", "stopped by the user")
    assert from_exception(own, provider="other") is own
    assert from_exception(own).kind == CANCELLED


def test_each_kind_of_exception_has_its_own_fixed_message_and_never_the_exceptions_text():
    timeout = from_exception(httpx.ReadTimeout("secret-text"), provider="p")
    assert (timeout.kind, str(timeout)) == ("timeout", "p timed out")
    assert str(from_exception(httpx.ReadTimeout("x"))) == "provider timed out"

    unreachable = from_exception(httpx.ConnectError("secret-text"), provider="p")
    assert (unreachable.kind, str(unreachable)) == ("transport", "p could not be reached (ConnectError)")
    assert str(from_exception(httpx.ConnectError("x"))) == "provider could not be reached (ConnectError)"

    other = from_exception(RuntimeError("secret-text sk-123"), provider="p")
    assert (other.kind, str(other)) == ("transport", "p failed (RuntimeError)")
    assert str(from_exception(RuntimeError("x"))) == "provider failed (RuntimeError)"
    assert all("secret-text" not in str(e) for e in (timeout, unreachable, other))


def _status_error(code: int, headers: dict[str, str]) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://provider.invalid/v1/chat")
    response = httpx.Response(code, headers=headers, text="body", request=request)
    return httpx.HTTPStatusError("failed", request=request, response=response)


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("7", 7.0),
        ("2.5", 2.5),
        ("1.2.3", None),
        ("not a date at all", None),
        ("inf", None),
        ("nan", None),
        ("", None),
        ("-3", 0.0),  # a negative wait is no wait, never a negative sleep
        ("Wed, 21 Oct 2015 07:28:00 GMT", 0.0),  # a date already past is no wait
    ],
)
def test_retry_after_is_read_when_it_is_a_number_and_never_raises_when_it_is_not(header, expected):
    error = from_exception(_status_error(429, {"retry-after": header}), provider="p")
    assert isinstance(error, ProviderError) and error.kind == "rate_limit"
    assert error.retry_after == expected


def test_retry_after_given_as_an_http_date_is_the_seconds_until_that_date_and_never_negative():
    from datetime import UTC, datetime, timedelta
    from email.utils import format_datetime

    soon = format_datetime(datetime.now(UTC) + timedelta(seconds=120), usegmt=True)
    error = from_exception(_status_error(429, {"retry-after": soon}), provider="p")
    assert error.retry_after is not None and 100 <= error.retry_after <= 120


def test_an_http_status_error_without_retry_after_has_none():
    assert from_exception(_status_error(503, {}), provider="p").retry_after is None
