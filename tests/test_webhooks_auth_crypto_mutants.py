"""Mutation-testing follow-up for the JWT parts of `glide/webhooks/auth.py` (see docs/MUTATION_REPORT.md).

The agent bearer token profile and the Google push verifier, over real keys generated in the test and a stubbed key
lookup (nothing is fetched). Split from tests/test_webhooks_auth_mutants.py the way tests/webhooks/test_webhooks_auth_hmac.py
is split from its crypto twin: this file needs the `webhooks` extra and is skipped without it.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import time
from types import SimpleNamespace

import pytest

jwt = pytest.importorskip("jwt")
pytest.importorskip("cryptography")
from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ed25519, rsa  # noqa: E402
from guards_webhooks import no_webhook_machine_reach  # noqa: E402, F401  (autouse: see tests/guards_webhooks.py)

from glide.webhooks import auth  # noqa: E402
from glide.webhooks.auth import AgentVerifier, GoogleVerifier  # noqa: E402
from glide.webhooks.contracts import AuthError  # noqa: E402
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


def unsigned_header_token(header: dict) -> str:
    def segment(value):
        return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")

    return segment(header) + "." + segment({}) + ".c2ln"


def test_a_key_id_of_100_characters_reaches_key_discovery_and_one_of_101_never_does():
    lookups = []

    def lookup(token):
        lookups.append(token)
        raise AuthError("stop here")

    google = GoogleVerifier()
    google.keys = SimpleNamespace(get_signing_key_from_jwt=lookup)
    for kid, reaches in (("k" * 100, True), ("k" * 101, False)):
        before = len(lookups)
        with pytest.raises(AuthError):
            google.verify(
                unsigned_header_token({"kid": kid, "alg": "RS256"}), audience="glide-push", service_account=SERVICE_ACCOUNT
            )
        assert (len(lookups) - before == 1) is reaches


def test_the_audience_and_issuer_of_a_push_token_are_checked():
    assert not push_accepted(aud="another-audience")
    assert not push_accepted(iss="https://issuer.invalid")
