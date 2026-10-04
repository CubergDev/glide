"""R2 audit: an error reply that quotes part of the request, or the key in another spelling, is cut."""

from __future__ import annotations

import json
from urllib.parse import quote, quote_plus

from glide.providers.errors import redact, status_error

REQUEST = (
    "Please transfer 500 dollars from my savings account ending 4421 to Dmitri Volkov and do not tell my wife about it. " * 12
)


def test_a_reply_that_quotes_an_excerpt_of_the_request_has_the_excerpt_cut():
    body = json.dumps({"error": {"message": f"Invalid prompt: '{REQUEST[:150]}...' contains disallowed content"}})
    error = status_error(400, body, provider="p", request_texts=[REQUEST])
    text = str(error)
    assert all(REQUEST[i : i + 20] not in text for i in range(0, 150 - 20))
    assert "contains disallowed content" in text  # what the server meant stays


def test_an_excerpt_quoted_with_other_case_and_spacing_is_cut_too():
    quoted = " ".join(REQUEST[40:140].upper().split())
    error = status_error(400, f"bad input near {quoted} end", provider="p", request_texts=[REQUEST])
    assert "DMITRI VOLKOV" not in str(error)


def test_a_short_phrase_shared_with_the_request_is_not_cut():
    error = status_error(400, json.dumps({"error": "invalid request"}), provider="p", request_texts=["an invalid request here"])
    assert "invalid request" in str(error)


def test_a_key_echoed_url_encoded_or_in_other_case_is_cut():
    key = "ab/cd+ef=ghijk"
    body = f"bad ?key={quote(key, safe='')} and again {quote_plus(key)}"
    assert "ab%2Fcd" not in str(status_error(401, body, provider="p", secrets=[key]))
    assert "%2Bef" not in str(status_error(401, body, provider="p", secrets=[key]))
    upper = "sk-ABCDEFGHIJKLMNOP1234"
    error = status_error(401, "rejected SK-abcdefghijklmnop1234 <html>", provider="p", secrets=[upper])
    assert "abcdefghijklmnop1234" not in str(error).lower()


def test_redact_leaves_ordinary_words_alone_for_a_short_secret():
    assert redact("a Cat sat", ["cat"], min_len=4) == "a Cat sat"
