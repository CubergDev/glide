"""The credential guard: a field that asks for a secret is never filled from model text. Offline, pure functions."""

import pytest

from glide.computer.writer import CREDENTIAL_HINTS, looks_credential


@pytest.mark.parametrize(
    "label",
    ["Password", "Enter your one-time code", "CVV", "Card number", "API key", "PIN", "seed phrase"],
)
def test_credential_guard_catches_dangerous_fields(label):
    assert looks_credential(label)


@pytest.mark.parametrize("label", ["Search invoices", "Customer name", "PO number", "Email address"])
def test_credential_guard_allows_ordinary_fields(label):
    assert not looks_credential(label)


def test_credential_hints_cover_the_obvious_cases():
    for must in ("password", "otp", "cvv", "token"):
        assert any(must in hint for hint in CREDENTIAL_HINTS)
