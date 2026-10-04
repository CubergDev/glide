"""Provider signatures and scoped, issuer-bound bearer authentication.

The HMAC checks (GitHub, Standard Webhooks `v1`) use only the standard library. PyJWT and `cryptography`
belong to the optional `webhooks` extra and are imported where they are needed, so importing this module
never requires them.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import re
import time
from dataclasses import dataclass
from types import ModuleType
from typing import NamedTuple

from .contracts import AuthError
from .trust_anchors import GOOGLE_PUSH_ISSUERS, GOOGLE_PUSH_JWKS_URL


class _Crypto(NamedTuple):
    jwt: ModuleType
    serialization: ModuleType
    ed25519: ModuleType
    rsa: ModuleType
    InvalidSignature: type[Exception]


def _crypto() -> _Crypto:
    """PyJWT and `cryptography`, or a clear error if the webhooks extra is missing."""
    try:
        import jwt
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ed25519, rsa
    except ImportError:
        raise ImportError("Install the webhooks extra (PyJWT with cryptography) to verify these signatures.") from None
    return _Crypto(jwt, serialization, ed25519, rsa, InvalidSignature)


def header(headers, name: str) -> str:
    values = headers.getlist(name)
    if len(values) != 1 or len(values[0]) > 8192:
        raise AuthError("Invalid authentication headers.")
    return values[0]


def bearer(headers) -> str:
    value = header(headers, "authorization")
    parts = value.split(" ")
    if len(parts) != 2 or parts[0].casefold() != "bearer" or not parts[1] or len(parts[1]) > 8192:
        raise AuthError("Invalid bearer authentication.")
    return parts[1]


def delivery_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", value):
        raise AuthError("Invalid delivery identity.")
    return value


def verify_github(body: bytes, signature: str, secrets: tuple[str, ...]) -> None:
    if not re.fullmatch(r"sha256=[a-f0-9]{64}", signature):
        raise AuthError("Invalid webhook authentication.")
    matched = False
    for secret in secrets:
        expected = "sha256=" + hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
        matched |= hmac.compare_digest(expected, signature)
    if not matched:
        raise AuthError("Invalid webhook authentication.")


def webhook_keys(secrets: tuple[str, ...]):
    keys = []
    for secret in secrets:
        try:
            prefix, encoded = secret.split("_", 1)
            decoded = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            raise ValueError("Invalid Standard Webhooks key.") from None
        if prefix == "whsec" and 24 <= len(decoded) <= 64:
            keys.append(("v1", decoded))
        elif prefix == "whpk" and len(decoded) == 32:
            keys.append(("v1a", _crypto().ed25519.Ed25519PublicKey.from_public_bytes(decoded)))
        else:
            raise ValueError("Invalid Standard Webhooks key.")
    return keys


def _signature_matches(scheme: str, key, signed: bytes, candidate: bytes) -> bool:
    if scheme == "v1":
        return hmac.compare_digest(hmac.digest(key, signed, "sha256"), candidate)
    try:
        key.verify(candidate, signed)
    except _crypto().InvalidSignature:
        return False
    return True


def verify_standard(body: bytes, headers, keys, tolerance=300, *, now=None) -> str:
    """Check a Standard Webhooks delivery against every configured key; returns its (signed) identity.

    The identity and attempt time are signed along with the exact body. Entries that are malformed or use another
    scheme are skipped, never trusted; every key and entry is tried so that timing does not say which one matched.
    """
    identity = delivery_id(header(headers, "webhook-id"))
    timestamp = header(headers, "webhook-timestamp")
    if not re.fullmatch(r"[0-9]{1,12}", timestamp):
        raise AuthError("Invalid webhook authentication.")
    if abs((time.time() if now is None else now) - int(timestamp)) > tolerance:
        raise AuthError("Expired webhook authentication.")
    signatures = header(headers, "webhook-signature").split(" ")
    if not 1 <= len(signatures) <= 8:
        raise AuthError("Invalid webhook authentication.")
    signed = identity.encode() + b"." + timestamp.encode() + b"." + body
    matched = False
    for signature in signatures:
        try:
            version, encoded = signature.split(",", 1)
            candidate = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            continue
        for scheme, key in keys:
            if version == scheme:
                matched |= _signature_matches(scheme, key, signed, candidate)
    if not matched:
        raise AuthError("Invalid webhook authentication.")
    return identity


def _claims(token, key, algorithm, issuer, audience, max_seconds):
    claims = _crypto().jwt.decode(
        token,
        key,
        algorithms=[algorithm],
        issuer=issuer,
        audience=audience,
        leeway=5,
        options={"require": ["exp", "iat", "iss", "aud", "sub"], "strict_aud": True},
    )
    if any(type(claims[name]) is not int for name in ("iat", "exp")):
        raise AuthError("Invalid token lifetime.")
    if not 0 < claims["exp"] - claims["iat"] <= max_seconds:
        raise AuthError("Invalid token lifetime.")
    if not isinstance(claims["sub"], str) or not 1 <= len(claims["sub"]) <= 200:
        raise AuthError("Invalid token subject.")
    return claims


def _token_header(token):
    parsed = _crypto().jwt.get_unverified_header(token)
    if not isinstance(parsed.get("kid"), str) or len(parsed["kid"]) > 100:
        raise AuthError("Invalid signing key.")
    if any(name in parsed for name in ("jku", "x5u", "jwk", "crit")):
        raise AuthError("Unsupported token header.")
    if parsed.get("typ", "JWT") not in {"JWT", "at+jwt"}:
        raise AuthError("Invalid token type.")
    return parsed


@dataclass(frozen=True)
class Principal:
    subject: str
    agent_id: str
    scopes: frozenset[str]


class AgentVerifier:
    """Verifies worker bearer tokens against the pinned public keys of an `AgentAuth` (never keys a token names)."""

    def __init__(self, settings):
        crypto = _crypto()
        self.settings, self.keys = settings, {}
        for configured in settings.keys:
            try:
                key = crypto.serialization.load_pem_public_key(configured.public_key.encode())
            except ValueError:
                raise ValueError("Invalid agent verification key.") from None
            valid = (configured.algorithm == "RS256" and isinstance(key, crypto.rsa.RSAPublicKey) and key.key_size >= 2048) or (
                configured.algorithm == "EdDSA" and isinstance(key, crypto.ed25519.Ed25519PublicKey)
            )
            if not valid:
                raise ValueError("Agent key does not match its configured algorithm.")
            self.keys[configured.kid] = (configured.algorithm, key)

    def verify(self, token, agent_id, scope) -> Principal:
        jwt = _crypto().jwt
        try:
            parsed = _token_header(token)
            algorithm, key = self.keys[parsed["kid"]]
            if parsed["alg"] != algorithm:
                raise AuthError("Invalid token algorithm.")
            claims = _claims(token, key, algorithm, self.settings.issuer, self.settings.audience, self.settings.max_token_seconds)
            scopes = claims.get("scope", claims.get("scp", ""))
            if not isinstance(scopes, str) or len(scopes) > 500 or claims.get("agent_id") != agent_id:
                raise AuthError("Invalid agent identity.")
            principal = Principal(claims["sub"], agent_id, frozenset(scopes.split()))
        except (AuthError, jwt.PyJWTError, KeyError, ValueError, TypeError):
            raise AuthError("Invalid bearer authentication.") from None
        if scope not in principal.scopes:
            raise PermissionError("Insufficient agent scope.")
        return principal


class GoogleVerifier:
    def __init__(self):
        # The URL is a fixed trust anchor (trust_anchors.py), never supplied by a callback, a JWT header or the
        # configuration. PyJWT caches the key set, refreshes on a new kid and bounds unknown-key refresh.
        self.keys = _crypto().jwt.PyJWKClient(
            GOOGLE_PUSH_JWKS_URL, cache_jwk_set=True, lifespan=300, timeout=3, cooldown_duration=30
        )

    def verify(self, token, *, audience, service_account):
        crypto = _crypto()
        try:
            parsed = _token_header(token)
            if parsed.get("alg") != "RS256":
                raise AuthError("Invalid push token algorithm.")
            key = self.keys.get_signing_key_from_jwt(token).key
            if not isinstance(key, crypto.rsa.RSAPublicKey) or key.key_size < 2048:
                raise AuthError("Invalid push signing key.")
            claims = _claims(token, key, "RS256", GOOGLE_PUSH_ISSUERS, audience, 3600)
            if claims.get("email") != service_account or claims.get("email_verified") is not True:
                raise AuthError("Invalid push service account.")
            return claims
        except (AuthError, crypto.jwt.PyJWTError, ValueError, TypeError, KeyError, OSError):
            raise AuthError("Invalid push authentication.") from None
