"""Mutation-testing follow-up for `glide/webhooks/auth.py` (see docs/MUTATION_REPORT.md).

The standard-library half: the existing suite proves the attacks it names (algorithm confusion, key discovery, rotation);
a mutation run showed that the limits behind them were not pinned. Here are the exact edges of the header, key, timestamp,
tolerance and signature-count rules, the shape of a GitHub signature, and the message for a missing extra. The JWT half
(agent tokens and the Google push verifier) is in tests/test_webhooks_auth_crypto_mutants.py and needs the extra. Each test
fails on a mutant of one of those lines.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import sys

import pytest
from guards_webhooks import no_webhook_machine_reach  # noqa: F401  (autouse: see tests/guards_webhooks.py)
from webhooks.helpers import Hdrs

from glide.webhooks import auth
from glide.webhooks.auth import header, verify_github, verify_standard, webhook_keys
from glide.webhooks.contracts import AuthError

# -- the standard library parts -------------------------------------------------------------------------------------


def test_the_header_limit_is_8192_characters_inclusive():
    assert header(Hdrs({"x": "a" * 8192}), "x") == "a" * 8192
    with pytest.raises(AuthError):
        header(Hdrs({"x": "a" * 8193}), "x")


def test_a_github_signature_that_is_not_plain_hex_is_an_authentication_failure_not_a_crash():
    """Without the shape check a non-ASCII signature would reach `hmac.compare_digest` and raise TypeError."""
    secrets = ("fixture-github-secret-" + "b" * 32,)
    for signature in ("sha256=" + "\u00e9" * 64, "sha256=" + "\u0661" * 64, "SHA256=" + "a" * 64):
        with pytest.raises(AuthError):
            verify_github(b"x", signature, secrets)


def secret(raw: bytes) -> str:
    return "whsec_" + base64.b64encode(raw).decode()


@pytest.mark.parametrize("length", [24, 64])
def test_a_symmetric_key_of_24_to_64_bytes_is_accepted(length):
    assert webhook_keys((secret(b"k" * length),)) == [("v1", b"k" * length)]


@pytest.mark.parametrize("length", [23, 65])
def test_a_symmetric_key_outside_24_to_64_bytes_is_refused(length):
    with pytest.raises(ValueError, match="Invalid Standard Webhooks key"):
        webhook_keys((secret(b"k" * length),))


def test_a_key_with_characters_outside_the_base64_alphabet_is_refused_not_cleaned():
    with pytest.raises(ValueError, match="Invalid Standard Webhooks key"):
        webhook_keys((secret(b"k" * 32) + "!!!!",))


BODY = b'{"goal": "x"}'
KEY = b"k" * 32


def v1(body: bytes = BODY, identity: str = "evt_1", timestamp: str = "1000") -> str:
    signed = identity.encode() + b"." + timestamp.encode() + b"." + body
    return "v1," + base64.b64encode(hmac.digest(KEY, signed, hashlib.sha256)).decode()


def standard(signature: str, *, timestamp: str = "1000", now: float = 1000.0) -> str:
    headers = Hdrs({"webhook-id": "evt_1", "webhook-timestamp": timestamp, "webhook-signature": signature})
    return verify_standard(BODY, headers, webhook_keys((secret(KEY),)), now=now)


@pytest.mark.parametrize("timestamp", ["abc", "1_000", "+1000", " 1000", "0x3e8", "", "1" * 13, "\u0661\u0660\u0660\u0660"])
def test_a_timestamp_that_is_not_plain_digits_is_an_authentication_failure(timestamp):
    with pytest.raises(AuthError):
        standard(v1(timestamp=timestamp), timestamp=timestamp)


def test_the_tolerance_is_inclusive_in_both_directions():
    assert standard(v1(), now=1300.0) == "evt_1"  # exactly 300 s after the timestamp
    assert standard(v1(), now=700.0) == "evt_1"  # exactly 300 s before it
    for now in (1300.5, 699.5):
        with pytest.raises(AuthError):
            standard(v1(), now=now)


def test_up_to_eight_signatures_are_considered_and_a_ninth_is_never_looked_at():
    junk = ["v1,AAAA"] * 7
    assert standard(" ".join([*junk, v1()])) == "evt_1"  # the valid one is the eighth
    with pytest.raises(AuthError):
        standard(" ".join([*junk, "v1,AAAA", v1()]))  # the valid one is the ninth


def test_a_signature_with_stray_characters_in_its_encoding_is_not_accepted_as_the_same_signature():
    with pytest.raises(AuthError):
        standard(v1() + "!!")
    with pytest.raises(AuthError):
        standard(v1().replace("v1,", "v1,!"))


@pytest.mark.parametrize("label", ["v1a", "v2", "V1", ""])
def test_a_correct_signature_under_the_wrong_version_label_is_not_accepted(label):
    """A symmetric signature must not pass because it was relabelled: the label picks the scheme."""
    with pytest.raises(AuthError):
        standard(v1().replace("v1,", label + ",", 1))


def test_the_signature_header_may_not_be_empty():
    with pytest.raises(AuthError):
        standard("")


# -- the optional extra ----------------------------------------------------------------------------------------------


def test_a_missing_webhooks_extra_is_reported_in_words(monkeypatch):
    monkeypatch.setitem(sys.modules, "jwt", None)  # `import jwt` now raises ImportError
    with pytest.raises(ImportError, match="Install the webhooks extra"):
        auth._crypto()
