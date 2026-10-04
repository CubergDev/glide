"""R2 audit: addresses in any spelling and prefixed or unprefixed credential assignments are refused."""

import pytest

from glide.memory.store import validate_summary, validate_text


@pytest.mark.parametrize(
    "text",
    [
        "opened bank.example.com/account/123?id=9",
        "see mailto:bob@x.com",
        "file:/Users/me/secret.txt",
        "/Users/cuberg/Documents/tax.pdf",
        "localhost:8080/admin",
        "went to 192.168.0.1 admin",
        "saved to ~/notes/plan.txt",
    ],
)
def test_a_summary_with_an_address_in_any_spelling_is_refused(text):
    with pytest.raises(ValueError):
        validate_summary(text)


@pytest.mark.parametrize(
    "text",
    ["used the search box then pressed Enter", "clicked the first result, e.g. the blue button", "opened main.py and/or run.sh"],
)
def test_a_plain_tactic_summary_is_accepted(text):
    assert validate_summary(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "GROQ_API_KEY=gsk_abcdef1234567890",
        "my_client_secret=abcd1234",
        "token=abc123xyz",
        "Authorization: Basic dXNlcjpwYXNz",
        "DEEPGRAM_API_KEY: abc123",
        "AIzaSyA-abcdefghijklmnopqrstuvwxyz0123456",
        "gsk_abcdef1234567890ab",
    ],
)
def test_a_named_or_vendor_credential_is_refused(text):
    with pytest.raises(ValueError):
        validate_text(text)


def test_a_placeholder_assignment_is_still_accepted():
    assert validate_text("api_key=your_api_key") == "api_key=your_api_key"
