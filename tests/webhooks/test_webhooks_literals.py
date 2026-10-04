"""D6: no endpoint, OAuth URL, model id or voice id is a literal in the webhook package.

The one exception is `trust_anchors.py` (a fixed key-set location and issuer names), documented there. A new
URL anywhere else fails this test: move it to configuration, or add it to that block with a reason.
"""

import re
from pathlib import Path

import glide.webhooks as package
from glide.webhooks import trust_anchors

ROOT = Path(package.__file__).parent
URL = re.compile(r"https?://[^\s\"')]+")
# Words that name a vendor or model family; configuration may mention them, code must not.
MODEL_WORDS = re.compile(r"\b(gpt-[\w.-]+|claude-[\w.-]+|gemini-[\w.-]+|o[134]-[\w.-]+|eleven_[\w]+)", re.I)


def sources():
    return sorted(path for path in ROOT.glob("*.py"))


def test_url_literals_appear_only_in_the_trust_anchor_block():
    offenders = {}
    for path in sources():
        urls = URL.findall(path.read_text(encoding="utf-8"))
        if path.name == "trust_anchors.py":
            assert set(urls) <= {trust_anchors.GOOGLE_PUSH_JWKS_URL, *trust_anchors.GOOGLE_PUSH_ISSUERS}, urls
            continue
        if urls:
            offenders[path.name] = urls
    assert not offenders, f"URL literals outside trust_anchors.py: {offenders}"


def test_no_model_or_voice_identifiers_in_the_package():
    found = {path.name: MODEL_WORDS.findall(path.read_text(encoding="utf-8")) for path in sources()}
    assert not {name: words for name, words in found.items() if words}


def test_trust_anchor_block_is_fixed_and_https():
    assert trust_anchors.GOOGLE_PUSH_JWKS_URL.startswith("https://")
    assert all(not issuer.startswith("http://") for issuer in trust_anchors.GOOGLE_PUSH_ISSUERS)


def test_old_product_names_are_gone():
    for path in sources():
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"PERMIT_|CLICKER_|clicker-|\.permit-", text), path.name
