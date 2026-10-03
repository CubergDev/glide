"""Real cryptographic fixtures; no key discovery or provider requests."""

import base64
import hmac
import json
import time
from types import SimpleNamespace

import pytest

jwt = pytest.importorskip("jwt")
pytest.importorskip("cryptography")
from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ed25519, rsa  # noqa: E402

from glide.webhooks.auth import (  # noqa: E402
    AgentVerifier,
    GoogleVerifier,
    bearer,
    verify_standard,
    webhook_keys,
)
from glide.webhooks.contracts import AuthError  # noqa: E402
from glide.webhooks.settings import AgentAuth, VerificationKey  # noqa: E402
from glide.webhooks.trust_anchors import GOOGLE_PUSH_JWKS_URL  # noqa: E402
from webhooks.helpers import Hdrs as Headers  # noqa: E402


def public_pem(private):
    return private.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()


def agent_verifier(*pairs):
    return AgentVerifier(
        AgentAuth(
            issuer="urn:fixture:issuer",
            audience="glide-workers",
            keys=tuple(
                VerificationKey(kid=kid, algorithm=algorithm, public_key=public_pem(private)) for kid, algorithm, private in pairs
            ),
        )
    )


def worker_claims(**changes):
    now = int(time.time())
    return {
        "iss": "urn:fixture:issuer",
        "aud": "glide-workers",
        "sub": "worker-1",
        "agent_id": "team-agent",
        "scope": "agent:claim agent:report",
        "iat": now,
        "exp": now + 120,
    } | changes


@pytest.mark.parametrize("algorithm", ["RS256", "EdDSA"])
def test_symmetric_signature_cannot_reuse_a_pinned_asymmetric_public_key(algorithm):
    private = (
        rsa.generate_private_key(public_exponent=65537, key_size=2048)
        if algorithm == "RS256"
        else ed25519.Ed25519PrivateKey.generate()
    )
    verifier = agent_verifier(("key", algorithm, private))
    # Build the malicious wire token directly: current JWT encoders also reject
    # asymmetric DER keys as HMAC secrets. The server must still pin algorithms.
    public = private.public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    signed = b".".join(
        jwt.utils.base64url_encode(json.dumps(value).encode())
        for value in ({"kid": "key", "alg": "HS256", "typ": "JWT"}, worker_claims())
    )
    forged = (signed + b"." + jwt.utils.base64url_encode(hmac.digest(public, signed, "sha256"))).decode()
    with pytest.raises(AuthError):
        verifier.verify(forged, "team-agent", "agent:claim")
    proper = jwt.encode(worker_claims(), private, algorithm=algorithm, headers={"kid": "key"})
    assert verifier.verify(proper, "team-agent", "agent:claim").subject == "worker-1"


def test_unsigned_jwt_is_rejected_even_with_a_known_key_id():
    key = ed25519.Ed25519PrivateKey.generate()
    token = jwt.encode(worker_claims(), None, algorithm="none", headers={"kid": "key"})
    with pytest.raises(AuthError):
        agent_verifier(("key", "EdDSA", key)).verify(token, "team-agent", "agent:claim")


def test_agent_key_rotation_accepts_both_trusted_keys_and_then_retires_the_old_key():
    old, new = ed25519.Ed25519PrivateKey.generate(), ed25519.Ed25519PrivateKey.generate()
    both = agent_verifier(("old", "EdDSA", old), ("new", "EdDSA", new))
    old_token = jwt.encode(worker_claims(), old, algorithm="EdDSA", headers={"kid": "old"})
    new_token = jwt.encode(worker_claims(), new, algorithm="EdDSA", headers={"kid": "new"})
    assert both.verify(old_token, "team-agent", "agent:claim").agent_id == "team-agent"
    assert both.verify(new_token, "team-agent", "agent:claim").agent_id == "team-agent"
    new_only = agent_verifier(("new", "EdDSA", new))
    with pytest.raises(AuthError):
        new_only.verify(old_token, "team-agent", "agent:claim")
    assert new_only.verify(new_token, "team-agent", "agent:claim").subject == "worker-1"


@pytest.mark.parametrize(
    "header", [{"jku": "urn:fixture:untrusted-keys"}, {"x5u": "urn:fixture:untrusted-cert"}, {"jwk": {}}, {"crit": []}]
)
def test_request_supplied_key_discovery_is_never_allowed(header):
    key = ed25519.Ed25519PrivateKey.generate()
    signed = jwt.encode(worker_claims(), key, algorithm="EdDSA", headers={"kid": "key"} | header)
    with pytest.raises(AuthError):
        agent_verifier(("key", "EdDSA", key)).verify(signed, "team-agent", "agent:claim")


@pytest.mark.parametrize(
    "algorithm,private",
    [("RS256", ed25519.Ed25519PrivateKey.generate()), ("EdDSA", rsa.generate_private_key(public_exponent=65537, key_size=2048))],
)
def test_configured_key_type_must_match_the_algorithm(algorithm, private):
    with pytest.raises(ValueError, match="configured algorithm"):
        agent_verifier(("key", algorithm, private))


@pytest.mark.parametrize(
    "change",
    [{"aud": ["glide-workers"]}, {"iat": True}, {"exp": True}, {"exp": int(time.time()) + 7200}, {"agent_id": "another-agent"}],
)
def test_worker_token_profile_rejects_ambiguous_audience_lifetime_and_agent_identity(change):
    key = ed25519.Ed25519PrivateKey.generate()
    signed = jwt.encode(worker_claims(**change), key, algorithm="EdDSA", headers={"kid": "key"})
    with pytest.raises(AuthError):
        agent_verifier(("key", "EdDSA", key)).verify(signed, "team-agent", "agent:claim")


def test_authorized_worker_without_required_scope_is_forbidden():
    key = ed25519.Ed25519PrivateKey.generate()
    signed = jwt.encode(worker_claims(scope="agent:read"), key, algorithm="EdDSA", headers={"kid": "key"})
    with pytest.raises(PermissionError):
        agent_verifier(("key", "EdDSA", key)).verify(signed, "team-agent", "agent:claim")


def standard_public_key(private):
    raw = private.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return "whpk_" + base64.b64encode(raw).decode()


def standard_headers(private, body, *, identity="event_1", timestamp="1000", signatures=None):
    content = identity.encode() + b"." + timestamp.encode() + b"." + body
    signature = "v1a," + base64.b64encode(private.sign(content)).decode()
    return Headers({"webhook-id": identity, "webhook-timestamp": timestamp, "webhook-signature": signatures or signature})


def test_ed25519_standard_webhooks_verify_exact_original_unicode_bytes():
    key = ed25519.Ed25519PrivateKey.generate()
    raw = '{"goal": "查看通知", "number": 7}\n'.encode()
    headers = standard_headers(key, raw)
    keys = webhook_keys((standard_public_key(key),))
    assert verify_standard(raw, headers, keys, now=1000) == "event_1"
    reserialized = json.dumps(json.loads(raw), ensure_ascii=False, separators=(",", ":")).encode()
    assert reserialized != raw
    with pytest.raises(AuthError):
        verify_standard(reserialized, headers, keys, now=1000)


def test_standard_webhooks_sign_both_identity_and_attempt_timestamp():
    key = ed25519.Ed25519PrivateKey.generate()
    raw = b"original"
    headers = standard_headers(key, raw)
    keys = webhook_keys((standard_public_key(key),))
    for change in ({"webhook-id": "event_2"}, {"webhook-timestamp": "1001"}):
        changed = Headers(dict(headers) | change)
        with pytest.raises(AuthError):
            verify_standard(raw, changed, keys, now=1000)


@pytest.mark.parametrize("now", [699, 1301])
def test_standard_webhooks_reject_expired_and_future_attempts(now):
    key = ed25519.Ed25519PrivateKey.generate()
    raw = b"original"
    with pytest.raises(AuthError):
        verify_standard(raw, standard_headers(key, raw), webhook_keys((standard_public_key(key),)), now=now)


def test_ed25519_rotation_accepts_a_valid_trusted_signature_among_multiple_signatures():
    old, new, outsider = [ed25519.Ed25519PrivateKey.generate() for _ in range(3)]
    raw = b"original"
    old_headers = standard_headers(old, raw)
    new_signature = standard_headers(new, raw)["webhook-signature"]
    outsider_signature = standard_headers(outsider, raw)["webhook-signature"]
    mixed = Headers(dict(old_headers) | {"webhook-signature": outsider_signature + " " + new_signature})
    both = webhook_keys((standard_public_key(old), standard_public_key(new)))
    assert verify_standard(raw, old_headers, both, now=1000) == "event_1"
    assert verify_standard(raw, mixed, both, now=1000) == "event_1"
    new_only = webhook_keys((standard_public_key(new),))
    with pytest.raises(AuthError):
        verify_standard(raw, old_headers, new_only, now=1000)
    assert verify_standard(raw, mixed, new_only, now=1000) == "event_1"


def test_public_key_supplied_in_webhook_headers_cannot_expand_the_trust_list():
    trusted, outsider = ed25519.Ed25519PrivateKey.generate(), ed25519.Ed25519PrivateKey.generate()
    raw = b"original"
    headers = Headers(dict(standard_headers(outsider, raw)) | {"webhook-public-key": standard_public_key(outsider)})
    with pytest.raises(AuthError):
        verify_standard(raw, headers, webhook_keys((standard_public_key(trusted),)), now=1000)


@pytest.mark.parametrize(
    "encoded",
    [
        "whpk_" + base64.b64encode(b"a" * 31).decode(),
        "whsec_" + base64.b64encode(b"a" * 23).decode(),
        "whsk_" + base64.b64encode(b"a" * 32).decode(),
        "whpk_invalid!",
    ],
)
def test_standard_webhooks_configuration_rejects_invalid_key_material(encoded):
    with pytest.raises(ValueError, match="Invalid Standard Webhooks key"):
        webhook_keys((encoded,))


def test_duplicate_authentication_headers_are_rejected():
    with pytest.raises(AuthError):
        bearer(Headers(raw=[(b"authorization", b"Bearer first"), (b"authorization", b"Bearer second")]))
    key = ed25519.Ed25519PrivateKey.generate()
    raw = b"original"
    headers = standard_headers(key, raw)
    duplicated = Headers(raw=[*headers.raw, (b"webhook-id", b"event_1")])
    with pytest.raises(AuthError):
        verify_standard(raw, duplicated, webhook_keys((standard_public_key(key),)), now=1000)


def test_google_key_discovery_uses_only_the_fixed_trusted_url_and_bounded_cache(monkeypatch):
    captured = []

    def client(url, **options):
        captured.append((url, options))
        return SimpleNamespace()

    monkeypatch.setattr(jwt, "PyJWKClient", client)
    GoogleVerifier()
    assert captured == [(GOOGLE_PUSH_JWKS_URL, {"cache_jwk_set": True, "lifespan": 300, "timeout": 3, "cooldown_duration": 30})]


def unsigned_header_token(header):
    def segment(value):
        return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")

    return segment(header) + "." + segment({}) + ".c2ln"


@pytest.mark.parametrize(
    "header",
    [
        {"kid": "google"},
        {"kid": "google", "alg": "HS256"},
        {"kid": True, "alg": "RS256"},
        {"kid": "google", "alg": "RS256", "typ": {}},
        {"kid": "google", "alg": "RS256", "jku": "urn:fixture:untrusted-keys"},
    ],
)
def test_malformed_google_headers_fail_uniformly_before_key_discovery(header):
    google = GoogleVerifier()

    def refuse_lookup(token):
        raise AssertionError("An invalid header reached key discovery")

    google.keys = SimpleNamespace(get_signing_key_from_jwt=refuse_lookup)
    with pytest.raises(AuthError, match="Invalid push authentication"):
        google.verify(unsigned_header_token(header), audience="glide-push", service_account="push@example.com")
