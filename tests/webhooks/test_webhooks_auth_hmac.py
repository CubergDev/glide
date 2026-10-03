"""Signature checks that need only the standard library, so they run in the plain suite."""

import base64
import hashlib
import hmac

import pytest

from glide.webhooks.auth import bearer, delivery_id, header, verify_github, verify_standard, webhook_keys
from glide.webhooks.contracts import AuthError
from webhooks.helpers import Hdrs

SECRETS = ("fixture-github-secret-" + "b" * 32,)


def sign(body, secret=SECRETS[0]):
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def test_github_signature_is_checked_over_the_original_bytes():
    body = '{"a": "查看"}'.encode()
    verify_github(body, sign(body), SECRETS)
    with pytest.raises(AuthError):
        verify_github(body + b" ", sign(body), SECRETS)


@pytest.mark.parametrize("signature", ["", "sha256=", "sha256=" + "A" * 64, "sha1=" + "a" * 40, "sha256=" + "a" * 63])
def test_github_signature_must_have_the_exact_shape(signature):
    with pytest.raises(AuthError):
        verify_github(b"x", signature, SECRETS)


def test_github_rotation_accepts_either_configured_secret_and_not_a_third():
    old, new = "old-secret-" + "c" * 32, "new-secret-" + "d" * 32
    body = b"payload"
    verify_github(body, sign(body, old), (old, new))
    verify_github(body, sign(body, new), (old, new))
    with pytest.raises(AuthError):
        verify_github(body, sign(body, "outsider-" + "e" * 32), (old, new))


def test_authentication_failures_name_no_secret_or_signature():
    with pytest.raises(AuthError) as error:
        verify_github(b"x", "sha256=" + "0" * 64, SECRETS)
    assert SECRETS[0] not in str(error.value) and "0" * 64 not in str(error.value)


def test_symmetric_standard_webhooks_verify_and_expire():
    secret = "whsec_" + base64.b64encode(b"k" * 32).decode()
    keys = webhook_keys((secret,))
    body = b'{"goal": "x"}'
    content = b"evt_1.1000." + body
    signature = "v1," + base64.b64encode(hmac.digest(b"k" * 32, content, "sha256")).decode()
    headers = Hdrs({"webhook-id": "evt_1", "webhook-timestamp": "1000", "webhook-signature": signature})
    assert verify_standard(body, headers, keys, now=1000) == "evt_1"
    with pytest.raises(AuthError):
        verify_standard(body, headers, keys, now=1301)
    with pytest.raises(AuthError):
        verify_standard(body + b" ", headers, keys, now=1000)


def test_duplicate_or_oversized_authentication_headers_are_rejected():
    with pytest.raises(AuthError):
        bearer(Hdrs(raw=[(b"authorization", b"Bearer first"), (b"authorization", b"Bearer second")]))
    with pytest.raises(AuthError):
        header(Hdrs({"x": "a" * 8193}), "x")
    with pytest.raises(AuthError):
        bearer(Hdrs({"authorization": "Basic abc"}))
    assert bearer(Hdrs({"authorization": "bearer tok"})) == "tok"


@pytest.mark.parametrize("value", ["", "a b", "a" * 201, "../x"])
def test_delivery_ids_are_bounded_tokens(value):
    with pytest.raises(AuthError):
        delivery_id(value)
