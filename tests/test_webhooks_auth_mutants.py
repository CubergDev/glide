"""Mutation-testing follow-up for `glide/webhooks/auth.py` (see docs/MUTATION_REPORT.md).

The existing suite proves the attacks it names (algorithm confusion, key discovery, rotation). A mutation run showed
that the limits behind them were not pinned: the exact edges of every length and lifetime, the shape of the
timestamp, the strength of a Google signing key, what the Google verifier demands of a token's account, the token
type, and the extra that must be installed. Each test here fails on a mutant of one of those lines. Keys are
generated in the test and nothing is fetched.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import hmac
import json
import sys
import time
from types import SimpleNamespace

import pytest
from guards_webhooks import no_webhook_machine_reach  # noqa: F401  (autouse: see tests/guards_webhooks.py)
from webhooks.helpers import Hdrs

from glide.webhooks.auth import header, verify_standard, webhook_keys
from glide.webhooks.contracts import AuthError

# -- the standard library parts -------------------------------------------------------------------------------------


def test_the_header_limit_is_8192_characters_inclusive():
    assert header(Hdrs({"x": "a" * 8192}), "x") == "a" * 8192
    with pytest.raises(AuthError):
        header(Hdrs({"x": "a" * 8193}), "x")


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


# -- agent bearer tokens ---------------------------------------------------------------------------------------------

jwt = pytest.importorskip("jwt")
pytest.importorskip("cryptography")
from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ed25519, rsa  # noqa: E402

from glide.webhooks import auth  # noqa: E402
from glide.webhooks.auth import AgentVerifier, GoogleVerifier  # noqa: E402
from glide.webhooks.settings import AgentAuth, VerificationKey  # noqa: E402


def pem(private) -> str:
    return private.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()


AGENT_KEY = ed25519.Ed25519PrivateKey.generate()


def verifier(*, kid: str = "key", max_token_seconds: int = 3600) -> AgentVerifier:
    return AgentVerifier(
        AgentAuth(
            issuer="urn:fixture:issuer",
            audience="glide-workers",
            keys=(VerificationKey(kid=kid, algorithm="EdDSA", public_key=pem(AGENT_KEY)),),
            max_token_seconds=max_token_seconds,
        )
    )


def claims(**changes) -> dict:
    now = int(time.time())
    return {
        "iss": "urn:fixture:issuer",
        "aud": "glide-workers",
        "sub": "worker-1",
        "agent_id": "team-agent",
        "scope": "agent:claim",
        "iat": now,
        "exp": now + 120,
    } | changes


def token(*, kid: str = "key", typ: str | None = None, **changes) -> str:
    headers = {"kid": kid} | ({"typ": typ} if typ is not None else {})
    return jwt.encode(claims(**changes), AGENT_KEY, algorithm="EdDSA", headers=headers)


def accepted(**kwargs) -> bool:
    try:
        verifier().verify(token(**kwargs), "team-agent", "agent:claim")
    except (AuthError, PermissionError):
        return False
    return True


def test_a_lifetime_must_be_a_whole_number_of_seconds():
    now = int(time.time())
    assert accepted(iat=now, exp=now + 120)
    assert not accepted(iat=now, exp=now + 120.5)
    assert not accepted(iat=now + 0.5, exp=now + 120)


def test_a_lifetime_must_be_positive_and_at_most_the_configured_maximum_inclusive():
    now = int(time.time())
    assert not accepted(iat=now, exp=now)  # zero seconds
    assert accepted(iat=now, exp=now + 1)
    assert accepted(iat=now, exp=now + 3600)
    assert not accepted(iat=now, exp=now + 3601)


def test_a_configured_maximum_lifetime_is_the_one_that_applies():
    now = int(time.time())
    short = verifier(max_token_seconds=60)
    assert short.verify(token(iat=now, exp=now + 60), "team-agent", "agent:claim").subject == "worker-1"
    with pytest.raises(AuthError):
        short.verify(token(iat=now, exp=now + 61), "team-agent", "agent:claim")


def test_a_token_that_expired_more_than_the_five_second_leeway_ago_is_refused():
    now = int(time.time())
    assert not accepted(iat=now - 125, exp=now - 5)


def test_a_subject_is_one_to_200_characters():
    assert accepted(sub="a")
    assert accepted(sub="a" * 200)
    assert not accepted(sub="a" * 201)
    assert not accepted(sub="")


def test_the_key_id_may_be_100_characters_and_the_key_must_be_known():
    long_kid = "k" * 100
    assert verifier(kid=long_kid).verify(token(kid=long_kid), "team-agent", "agent:claim").subject == "worker-1"
    with pytest.raises(AuthError):
        verifier().verify(token(kid="k" * 101), "team-agent", "agent:claim")
    with pytest.raises(AuthError):
        verifier().verify(token(kid="unknown"), "team-agent", "agent:claim")


@pytest.mark.parametrize("typ", ["JWT", "at+jwt"])
def test_the_two_accepted_token_types(typ):
    assert accepted(typ=typ)


@pytest.mark.parametrize("typ", ["JOSE", "jwt", "secevent+jwt"])
def test_any_other_token_type_is_refused(typ):
    assert not accepted(typ=typ)


def test_the_scope_claim_is_at_most_500_characters():
    exact = "agent:claim " + "x" * (500 - len("agent:claim "))
    assert len(exact) == 500
    assert accepted(scope=exact)
    assert not accepted(scope=exact + "x")


def test_the_algorithm_pin_does_not_rely_on_the_jwt_library_alone(monkeypatch):
    """Even if `jwt.decode` accepted anything, a token whose header names another algorithm is refused first."""
    forged = ".".join(
        jwt.utils.base64url_encode(json.dumps(part).encode()).decode()
        for part in ({"kid": "key", "alg": "HS256", "typ": "JWT"}, claims(), "signature")
    )
    monkeypatch.setattr(jwt, "decode", lambda *args, **kwargs: claims())
    with pytest.raises(AuthError):
        verifier().verify(forged, "team-agent", "agent:claim")


def test_a_verified_principal_cannot_be_changed_afterwards():
    principal = verifier().verify(token(), "team-agent", "agent:claim")
    with pytest.raises(dataclasses.FrozenInstanceError):
        principal.agent_id = "another-agent"


def test_a_key_that_is_not_a_public_key_is_refused_with_the_configured_message():
    bad = VerificationKey(
        kid="key", algorithm="EdDSA", public_key="-----BEGIN PUBLIC KEY-----\nnot a key at all\n-----END-----\n"
    )
    with pytest.raises(ValueError, match="Invalid agent verification key"):
        AgentVerifier(AgentAuth(issuer="i", audience="a", keys=(bad,)))


# -- the Google push verifier ----------------------------------------------------------------------------------------

ISSUER = auth.GOOGLE_PUSH_ISSUERS[0]
SERVICE_ACCOUNT = "push@example.invalid"
RSA_2048 = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def google_claims(**changes) -> dict:
    now = int(time.time())
    return {
        "iss": ISSUER,
        "aud": "glide-push",
        "sub": "google-1",
        "email": SERVICE_ACCOUNT,
        "email_verified": True,
        "iat": now,
        "exp": now + 600,
    } | changes


def google(public=None) -> GoogleVerifier:
    verifier_ = GoogleVerifier()
    verifier_.keys = SimpleNamespace(get_signing_key_from_jwt=lambda _token: SimpleNamespace(key=public or RSA_2048.public_key()))
    return verifier_


def push(signing=None, **changes) -> str:
    return jwt.encode(google_claims(**changes), signing or RSA_2048, algorithm="RS256", headers={"kid": "google"})


def push_accepted(public=None, signing=None, **changes) -> bool:
    try:
        google(public).verify(push(signing, **changes), audience="glide-push", service_account=SERVICE_ACCOUNT)
    except AuthError:
        return False
    return True


def test_a_push_token_for_the_expected_account_is_accepted_and_its_claims_returned():
    returned = google().verify(push(), audience="glide-push", service_account=SERVICE_ACCOUNT)
    assert returned["email"] == SERVICE_ACCOUNT and returned["sub"] == "google-1"


def test_a_push_token_from_another_account_is_refused():
    assert not push_accepted(email="someone-else@example.invalid")
    assert not push_accepted(email=None)


@pytest.mark.parametrize("verified", [False, "true", 1, None])
def test_a_push_token_whose_account_is_not_verified_with_a_real_true_is_refused(verified):
    assert not push_accepted(email_verified=verified)


def test_a_push_token_lives_at_most_an_hour_inclusive():
    now = int(time.time())
    assert push_accepted(iat=now, exp=now + 3600)
    assert not push_accepted(iat=now, exp=now + 3601)


def test_a_push_signing_key_must_be_an_rsa_key_of_at_least_2048_bits():
    weak = rsa.generate_private_key(public_exponent=65537, key_size=1024)
    assert not push_accepted(public=weak.public_key(), signing=weak)
    assert push_accepted(public=RSA_2048.public_key(), signing=RSA_2048)


def test_the_audience_and_issuer_of_a_push_token_are_checked():
    assert not push_accepted(aud="another-audience")
    assert not push_accepted(iss="https://issuer.invalid")


# -- the optional extra ----------------------------------------------------------------------------------------------


def test_a_missing_webhooks_extra_is_reported_in_words(monkeypatch):
    monkeypatch.setitem(sys.modules, "jwt", None)  # `import jwt` now raises ImportError
    with pytest.raises(ImportError, match="Install the webhooks extra"):
        auth._crypto()
