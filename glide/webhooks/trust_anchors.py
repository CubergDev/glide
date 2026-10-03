"""Fixed trust anchors: the only network locations and issuer names that may be literals in this package.

D6 says endpoints are configuration, never code. Verifying a Gmail Pub/Sub push token is the one place where the
verifier must know a third party's identity without being told it by the request: the key set that Google signs
those tokens with, and the issuer name inside them. Both are fixed public facts of that protocol. They are not
supplied by a callback, a JWT header or the configuration file, so nothing an attacker sends can redirect where
keys are fetched from. They live here, in one named block, so that `tests/webhooks/test_webhooks_literals.py` can
allow URL literals in this file and in no other file of the package.

If Google changes either value, edit this block; do not copy a URL anywhere else.
"""

# --- trust-anchor allowlist (the D6 literal scan reads exactly this block) ---------------------------------------
GOOGLE_PUSH_JWKS_URL = "https://www.googleapis.com/oauth2/v3/certs"
GOOGLE_PUSH_ISSUERS = ("https://accounts.google.com", "accounts.google.com")
# --- end allowlist ----------------------------------------------------------------------------------------------
