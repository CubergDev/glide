"""D6: nothing vendor-specific is hard-coded in the package.

Model ids, vendor endpoints, prices, voice ids and OAuth URLs are configuration (`glide.toml`), never code, and
they go stale. This test parses every `.py` file under `glide/` with `ast` and checks the string constants (and a few
named assignments) against the rules below. Comments never reach the AST, and docstrings are skipped on purpose:
documentation may name a vendor as an example. Any other string is code and is checked.

Rules (the `rule` of a finding):

- `model-id`: claude-*, gpt-*, gemini-*, deepseek-*, llama, mistral, qwen, grok, o1/o3/o4 style, vendor/model slugs,
  ElevenLabs `scribe_*` and `eleven_*` model ids, `jev-*`, and any non-empty string assigned to a model-named field
  (`model`, `DEFAULT_MODEL`, `model_id`), because a new vendor's id matches none of the patterns.
- `endpoint`: a URL or bare hostname that names a vendor API (a known vendor domain, or an `api.`, `auth.`, `login.`
  ... host), or any URL with a path, or any URL whose host is an IP address. A plain site address such as
  `https://github.com/` is a destination, not an endpoint, and passes. Loopback, documentation-range addresses and the
  RFC 2606 / 6761 test names (`.test`, `.example`, `.invalid`, `.localhost`) pass; `.local` and `.internal` do not, since
  those resolve to real services. Wire paths an adapter appends to a configured base
  (`/chat/completions`, `/v1/messages`) are the adapter's protocol, not a hard-coded endpoint, and are not checked.
- `oauth`: issuer, authorize, token and JWKS URLs, `.well-known` paths, scopes, and OAuth client ids and secrets.
- `price`: a price per token or per million tokens, in text or as a number (or constant arithmetic such as
  `3 / 1_000_000`) assigned to a price-like name.
- `voice-id`: a literal voice id, or a non-empty string assigned to a voice-like name or parameter default.
- `secret`: something shaped like an API key, or a non-empty string assigned to a credential-named field (`api_key`,
  `token`, `password`, `secret`, `access_key`; not `api_key_env`, which holds a variable's name). Such a finding never
  prints the value.

Exceptions are explicit and listed below, each with a reason, and nothing is allowlisted silently:

- `ALLOWLIST` is for literals that are legitimate and stay (for example a fixed trust anchor).
- `D6_DEBT` is for real violations that already exist in the tree. They are recorded, not excused: each is something
  D6 says must move into `glide.toml` or documented examples, and the entry is deleted in the change that moves it.

Both lists are checked in both directions. A literal that is not listed fails the test, and an entry that no longer
matches anything fails it too, so a fixed violation cannot leave a stale excuse behind. An entry matches by file, rule
and the matched text (not by line), so it survives edits around it. It also admits any further occurrence of the same
text in the same file, so keep the matched text specific.

To add an exception, add an `Exception_` with a reason a reviewer can judge. Do not widen a rule to make a test pass.
"""

from __future__ import annotations

import ast
import ipaddress
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "glide"

RULES = ("model-id", "endpoint", "oauth", "price", "voice-id", "secret")


@dataclass(frozen=True)
class Hit:
    path: str  # posix, relative to the repository root
    rule: str
    token: str  # the matched text (never the value, for a secret)
    line: int

    def key(self) -> tuple[str, str, str]:
        return (self.path, self.rule, self.token)

    def __str__(self) -> str:
        return f"{self.path}:{self.line} [{self.rule}] {self.token!r}"


@dataclass(frozen=True)
class Exception_:
    path: str
    rule: str
    token: str
    reason: str

    def key(self) -> tuple[str, str, str]:
        return (self.path, self.rule, self.token)


# --- the rules -------------------------------------------------------------------------------------------------

# Model ids. Each pattern needs the vendor's own separator or digit, so ordinary prose ("gemini" as a provider name,
# "echo", "o3" inside a word) is not a model id.
_MODEL_PATTERNS = (
    r"\b(?:chat)?gpt-[a-z0-9]",
    r"\bclaude-[a-z0-9]",
    r"\bgemini-[a-z0-9]",
    r"\bgemma-?\d",
    r"\bdeepseek[-/][a-z0-9]",
    r"\bllama(?:-?\d|-[a-z])",
    r"\b(?:mistral|mixtral|codestral|ministral|pixtral)-[a-z0-9]",
    r"\bqwen(?:\d|[-/:])",
    r"\bgrok-?\d",
    r"\bcommand-r",
    r"\bo[1-9]-(?:mini|pro|preview|high|medium|low|\d{4})\b",
    r"\bwhisper-(?:\d|large|tiny|base|small|medium)",
    r"\bscribe_v\d",
    r"\beleven_(?:v\d|multilingual|turbo|flash|monolingual)",
    r"\btext-embedding-",
    r"\bjev-(?:latest|\d)",
    r"\b(?:openai|anthropic|google|meta-llama|mistralai|x-ai|qwen|deepseek|cohere|nvidia|microsoft)/[a-z0-9][a-z0-9._:-]*",
)
# The match is stretched over the rest of the id ("claude-h" becomes "claude-haiku-4-5") so a finding names the whole thing.
_MODEL_RE = re.compile("(?:" + "|".join(f"(?:{p})" for p in _MODEL_PATTERNS) + r")[a-z0-9._:/@-]*", re.IGNORECASE)
_O_SERIES_WHOLE = re.compile(r"o[1-9]", re.IGNORECASE)  # the whole string is "o1", "o3", ...

_HOST = r"(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}"
_IP_HOST = r"\d{1,3}(?:\.\d{1,3}){3}|\[[0-9a-f:.]+\]"
_URL_RE = re.compile(
    rf"(?<![\w.@/-])(?P<scheme>(?:https?|wss?)://)?(?P<host>{_HOST}|{_IP_HOST})(?::\d+)?(?P<path>/[^\s\"'<>)\]}}]*)?(?![\w-])",
    re.IGNORECASE,
)
# Without a scheme only these endings count as a hostname, so "self.io" and "os.path" are not mistaken for one.
_BARE_TLDS = frozenset({"com", "ai", "io", "net", "org", "dev", "app", "cloud", "xyz"})
_LOCAL_HOSTS = frozenset({"localhost", "localhost.localdomain"})
# Written out, not `is_private` / `is_reserved`, whose answers have changed between Python patch releases.
_DOCUMENTATION_NETS = tuple(
    ipaddress.ip_network(net) for net in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24", "2001:db8::/32")
)
_RESERVED_SUFFIXES = (".localhost", ".invalid", ".test", ".example")  # not .local or .internal: real LAN names
_RESERVED_DOMAINS = ("example.com", "example.org", "example.net")
VENDOR_DOMAINS = (
    "openai.com",
    "openai.azure.com",
    "chatgpt.com",
    "anthropic.com",
    "claude.ai",
    "openrouter.ai",
    "deepseek.com",
    "googleapis.com",
    "elevenlabs.io",
    "typesafe.ai",
    "mistral.ai",
    "groq.com",
    "together.ai",
    "together.xyz",
    "fireworks.ai",
    "cohere.com",
    "cohere.ai",
    "x.ai",
    "perplexity.ai",
    "huggingface.co",
    "azure.com",
    "deepgram.com",
    "assemblyai.com",
    "cartesia.ai",
    "auth0.com",
    "okta.com",
    "microsoftonline.com",
    "accounts.google.com",
)
_API_LABEL = re.compile(r"^(?:api|oauth2?|auth|login|accounts|sso|identity|gateway)(?:[-\d].*)?$", re.IGNORECASE)

_OAUTH_TEXT = re.compile(
    r"/oauth2?\b|/authori[sz]e\b|/token\b|/\.well-known/|/jwks?\b|jwks\.json|\bopenid\b|\boffline_access\b",
    re.IGNORECASE,
)

_PRICE_TEXT = re.compile(
    r"\$\s?\d[\d,]*(?:\.\d+)?\s*(?:/|per\s)\s*(?:m\b|k\b|1m\b|1k\b|mtok|ktok|million|thousand|token|tok\b)"
    r"|\b\d+(?:\.\d+)?\s*(?:usd|dollars?)\s*(?:/|per\s)\s*(?:m\b|1m\b|mtok|million|token)"
    r"|\bper[_ ](?:million|1m|mtok|token)s?\b.{0,20}\d",
    re.IGNORECASE,
)

_SECRET_TEXT = re.compile(
    r"\bsk-[A-Za-z0-9][A-Za-z0-9_-]{19,}|\bAIza[0-9A-Za-z_-]{30,}|\bgh[pousr]_[A-Za-z0-9]{30,}|\bxox[abprs]-[A-Za-z0-9-]{10,}"
)

_VOICE_ID_WHOLE = re.compile(r"[A-Za-z0-9]{20,22}")  # ElevenLabs voice ids look like this

# Names whose string (or number) value is vendor configuration. Anchored so that `voice_env` or `client_id_env`
# (an environment variable's name) are not caught.
_VOICE_NAME = re.compile(r"(?:^|_)voice(?:_?(?:ids?|names?))?s?$", re.IGNORECASE)
_MODEL_NAME = re.compile(r"(?:^|_)models?(?:_?ids?)?$", re.IGNORECASE)
# A credential's value, not the name of the environment variable that holds it (`api_key_env` does not end this way).
_SECRET_NAME = re.compile(r"(?:^|_)(?:api_?key|secret|password|passwd|token|access_?key|private_?key)$", re.IGNORECASE)
_OAUTH_NAME = re.compile(
    r"(?:^|_)(?:client_?(?:id|secret)|issuer|authori[sz](?:e|ation)_(?:url|uri|endpoint)|token_(?:url|uri|endpoint)"
    r"|auth_url|jwks(?:_url|_uri)?|(?:oauth_)?scopes?)$",
    re.IGNORECASE,
)
_PRICE_NAME = re.compile(
    r"price|pricing|(?:^|_)usd(?:_|$)|per_(?:m|1m|k|1k|million|mtok|mtoks|token|tokens|tok)(?:_|$)|cost_per|rate_card",
    re.IGNORECASE,
)


def _classify_url(match: re.Match[str]) -> str | None:
    """A token for the matched URL or hostname if it names a vendor endpoint, else None."""
    scheme, host, path = match.group("scheme"), match.group("host").lower(), match.group("path") or ""
    if host[0].isdigit() or host[0] == "[":  # an IP literal; without a scheme it is a version number, not an address
        try:
            address = ipaddress.ip_address(host.strip("[]"))
        except ValueError:
            return None
        if not scheme or address.is_loopback or address.is_unspecified or any(address in n for n in _DOCUMENTATION_NETS):
            return None
        return match.group(0).rstrip(".,;:")
    tld = host.rsplit(".", 1)[-1]
    if not scheme and tld not in _BARE_TLDS:
        return None
    if host in _LOCAL_HOSTS or host.endswith(_RESERVED_SUFFIXES):
        return None
    if any(host == d or host.endswith("." + d) for d in _RESERVED_DOMAINS):
        return None
    vendor = any(host == d or host.endswith("." + d) for d in VENDOR_DOMAINS)
    api_host = bool(_API_LABEL.match(host.split(".", 1)[0]))
    has_path = bool(scheme) and path not in ("", "/")
    if vendor or api_host or has_path:
        return match.group(0).rstrip(".,;:")
    return None


def _scan_text(text: str) -> Iterator[tuple[str, str, int]]:
    """(rule, token, offset into the string) for every violation in one string constant."""
    stripped = text.strip()
    if _O_SERIES_WHOLE.fullmatch(stripped):
        yield "model-id", stripped, 0
    for match in _MODEL_RE.finditer(text):
        yield "model-id", match.group(0).rstrip(".:/-"), match.start()
    for match in _URL_RE.finditer(text):
        token = _classify_url(match)
        if token:
            yield "endpoint", token, match.start()
    for match in _OAUTH_TEXT.finditer(text):
        yield "oauth", match.group(0), match.start()
    for match in _PRICE_TEXT.finditer(text):
        yield "price", match.group(0).strip(), match.start()
    if secret := _SECRET_TEXT.search(text):
        yield "secret", "<key-shaped literal, value withheld>", secret.start()
    mixed = any(c.islower() for c in stripped) and any(c.isupper() for c in stripped) and any(c.isdigit() for c in stripped)
    if mixed and _VOICE_ID_WHOLE.fullmatch(stripped):
        yield "voice-id", stripped, 0


# --- finding what to check -------------------------------------------------------------------------------------


def _docstrings(tree: ast.AST) -> set[int]:
    """The ids of the Constant nodes that are docstrings (module, class and function): documentation, not code."""
    found: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                found.add(id(first.value))
    return found


def _leaves(value: ast.AST | None) -> Iterator[ast.Constant]:
    """The constants an assignment really stores: the value itself, container members, `a or "x"` and `d.get(k, "x")`
    defaults. Not constants that are merely arguments (`options.get("voice")` stores no voice)."""
    if value is None:
        return
    if isinstance(value, ast.Constant):
        yield value
    elif isinstance(value, ast.BinOp | ast.UnaryOp):
        if (number := _fold(value)) is not None:
            yield ast.copy_location(ast.Constant(number), value)
    elif isinstance(value, ast.Dict):
        for item in value.values:
            yield from _leaves(item)
    elif isinstance(value, ast.List | ast.Tuple | ast.Set):
        for item in value.elts:
            yield from _leaves(item)
    elif isinstance(value, ast.BoolOp):
        for item in value.values:
            yield from _leaves(item)
    elif isinstance(value, ast.IfExp):
        yield from _leaves(value.body)
        yield from _leaves(value.orelse)
    elif (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr in ("get", "getenv", "setdefault")
        and len(value.args) >= 2
    ):
        yield from _leaves(value.args[1])


def _fold(node: ast.AST) -> int | float | None:
    """The number a constant arithmetic expression (`3 / 1_000_000`, `-15.0`) stands for; None for anything else."""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, int | float) and not isinstance(node.value, bool) else None
    try:
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
            inner = _fold(node.operand)
            return None if inner is None else -inner if isinstance(node.op, ast.USub) else inner
        if isinstance(node, ast.BinOp):
            left, right = _fold(node.left), _fold(node.right)
            if left is None or right is None:
                return None
            match node.op:
                case ast.Add():
                    return left + right
                case ast.Sub():
                    return left - right
                case ast.Mult():
                    return left * right
                case ast.Div():
                    return left / right
                case ast.FloorDiv():
                    return left // right
                case ast.Pow() if abs(right) <= 64:
                    return left**right
    except ArithmeticError:  # division by zero, overflow
        pass
    return None


def _target_name(target: ast.AST) -> str | None:
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
        return target.slice.value
    return None


def _named_values(tree: ast.AST) -> Iterator[tuple[str, ast.AST | None]]:
    """(name, value) pairs: assignments, annotated assignments, keyword arguments, dict entries, parameter defaults."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if name := _target_name(target):
                    yield name, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            if name := _target_name(node.target):
                yield name, node.value
        elif isinstance(node, ast.keyword) and node.arg:
            yield node.arg, node.value
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    yield key.value, value
        elif isinstance(node, ast.arguments):
            positional = [*node.posonlyargs, *node.args]
            for arg, default in zip(positional[len(positional) - len(node.defaults) :], node.defaults, strict=True):
                yield arg.arg, default
            for arg, default in zip(node.kwonlyargs, node.kw_defaults, strict=True):
                if default is not None:
                    yield arg.arg, default


def scan_source(source: str, path: str) -> list[Hit]:
    """Every violation in one module's source. Raises SyntaxError for a file that does not parse."""
    tree = ast.parse(source, filename=path)
    skip = _docstrings(tree)
    hits: dict[tuple[str, str, str], Hit] = {}

    def add(rule: str, token: str, line: int) -> None:
        hits.setdefault((path, rule, token), Hit(path, rule, token, line))

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            for rule, token, offset in _scan_text(node.value):
                add(rule, token, node.lineno + node.value.count("\n", 0, offset))

    for name, value in _named_values(tree):
        for leaf in _leaves(value):
            if id(leaf) in skip:
                continue
            if isinstance(leaf.value, str) and leaf.value.strip():
                value = leaf.value.strip()
                withheld = f"{name} = <value withheld>"
                if _VOICE_NAME.search(name):
                    add("voice-id", f"{name} = {value!r}", leaf.lineno)
                if _OAUTH_NAME.search(name):  # a client id is public; a client secret is never printed
                    add("oauth", withheld if _SECRET_NAME.search(name) else f"{name} = {value!r}", leaf.lineno)
                elif _SECRET_NAME.search(name):
                    add("secret", withheld, leaf.lineno)
                if _MODEL_NAME.search(name):
                    add("model-id", value, leaf.lineno)
            elif isinstance(leaf.value, int | float) and not isinstance(leaf.value, bool) and leaf.value:
                if _PRICE_NAME.search(name):
                    add("price", f"{name} = {leaf.value!r}", leaf.lineno)
    return sorted(hits.values(), key=lambda h: (h.path, h.line, h.rule, h.token))


def scan_tree(package: Path = PACKAGE, root: Path = ROOT) -> list[Hit]:
    """Every violation in every `.py` file under `package` (a file that does not parse is an error, not a pass)."""
    found: list[Hit] = []
    for file in sorted(package.rglob("*.py")):
        if "__pycache__" in file.parts:
            continue
        found.extend(scan_source(file.read_text(encoding="utf-8"), file.relative_to(root).as_posix()))
    return found


def reconcile(hits: Iterable[Hit], exceptions: Iterable[Exception_]) -> list[str]:
    """What is wrong between the findings and the listed exceptions, as readable lines (empty when they agree)."""
    allowed = {e.key(): e for e in exceptions}
    seen = {h.key() for h in hits}
    problems = [f"not allowed: {h}" for h in hits if h.key() not in allowed]
    problems += [
        f"stale exception (no longer matches anything): {e.path} [{e.rule}] {e.token!r}"
        for e in allowed.values()
        if e.key() not in seen
    ]
    return problems


# --- exceptions ------------------------------------------------------------------------------------------------

# Legitimate literals that stay. Empty today. A fixed trust anchor (for example the Google JWKS URL a webhook
# verifier pins) goes here with the reason it must be a constant and why configuration would be unsafe.
ALLOWLIST: tuple[Exception_, ...] = (
    Exception_(
        "glide/computer/execution/dom.py",
        "secret",
        "SECRET = <value withheld>",
        "A regular expression, injected into the page script, that recognises password-like field names so their values are "
        "never read. It holds no credential.",
    ),
    Exception_(
        "glide/routing/decision.py",
        "model-id",
        "stop_model",
        "WHY_STOP_MODEL is a routing reason code that says the stop came from the model, not the id of any model.",
    ),
    Exception_(
        "glide/computer/config.py",
        "endpoint",
        "https://console.typesafe.ai/",
        "A web page the SITES catalog lets the classifier open by name, a destination the person sees, not an API endpoint.",
    ),
    Exception_(
        "glide/webhooks/trust_anchors.py",
        "endpoint",
        "https://www.googleapis.com/oauth2/v3/certs",
        "Fixed trust anchor: the key set that verifies Google push-delivery JWTs. It must be a constant, not configuration.",
    ),
    Exception_(
        "glide/webhooks/trust_anchors.py",
        "oauth",
        "/oauth2",
        "The path of the fixed Google key-set trust anchor above, matched separately by the oauth rule.",
    ),
    Exception_(
        "glide/webhooks/trust_anchors.py",
        "oauth",
        "GOOGLE_PUSH_JWKS_URL = 'https://www.googleapis.com/oauth2/v3/certs'",
        "The named constant that holds the fixed Google key-set trust anchor; the one place the URL may live.",
    ),
    Exception_(
        "glide/webhooks/trust_anchors.py",
        "endpoint",
        "accounts.google.com",
        "Fixed trust anchor: the issuer name Google puts in push-delivery JWTs, compared against the token claim.",
    ),
    Exception_(
        "glide/webhooks/trust_anchors.py",
        "endpoint",
        "https://accounts.google.com",
        "Fixed trust anchor: the other spelling of Google's issuer claim, compared against the token claim.",
    ),
    Exception_(
        "glide/webhooks/app.py",
        "oauth",
        "WORKER_SCOPES = 'agent:claim'",
        "The webhook service's own worker permission names (claim), not an OAuth provider scope or endpoint.",
    ),
    Exception_(
        "glide/webhooks/app.py",
        "oauth",
        "WORKER_SCOPES = 'agent:report'",
        "The webhook service's own worker permission names (report), not an OAuth provider scope or endpoint.",
    ),
)

# Real violations found in the tree when this test was written (4 Oct 2026). Each is something D6 says to move into
# glide.toml or a documented example. Delete the entry in the change that moves it; the test fails if it lingers.
D6_DEBT: tuple[Exception_, ...] = (
    Exception_(
        "glide/providers/config.py",
        "endpoint",
        "https://api.openai.com/v1",
        "PRESETS holds a vendor base URL; D6 wants endpoints in glide.toml or documented examples only.",
    ),
    Exception_(
        "glide/providers/config.py",
        "endpoint",
        "https://openrouter.ai/api/v1",
        "PRESETS holds a vendor base URL; D6 wants endpoints in glide.toml or documented examples only.",
    ),
    Exception_(
        "glide/providers/config.py",
        "endpoint",
        "https://api.deepseek.com",
        "PRESETS holds a vendor base URL; D6 wants endpoints in glide.toml or documented examples only.",
    ),
    Exception_(
        "glide/providers/config.py",
        "endpoint",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "PRESETS holds a vendor base URL; D6 wants endpoints in glide.toml or documented examples only.",
    ),
    Exception_(
        "glide/providers/config.py",
        "endpoint",
        "https://api.typesafe.ai",
        "PRESETS holds a vendor base URL; D6 wants endpoints in glide.toml or documented examples only.",
    ),
    Exception_(
        "glide/providers/config.py",
        "model-id",
        "deepseek/deepseek-v4.1-flash",
        "DEFAULT_TOML fallback chain hard-codes model ids; D6 says configuration, never code.",
    ),
    Exception_(
        "glide/providers/config.py",
        "model-id",
        "gpt-6-luna",
        "DEFAULT_TOML fallback chain hard-codes model ids; D6 says configuration, never code.",
    ),
    Exception_(
        "glide/providers/config.py",
        "model-id",
        "gemini-3.5-flash-lite",
        "DEFAULT_TOML fallback chain hard-codes model ids; D6 says configuration, never code.",
    ),
    Exception_(
        "glide/providers/config.py",
        "model-id",
        "gpt-6.1-sol",
        "DEFAULT_TOML fallback chain hard-codes model ids; D6 says configuration, never code.",
    ),
    Exception_(
        "glide/providers/config.py",
        "model-id",
        "scribe_v2_realtime",
        "DEFAULT_TOML fallback chain hard-codes model ids; D6 says configuration, never code.",
    ),
    Exception_(
        "glide/providers/config.py",
        "model-id",
        "gpt-transcribe",
        "DEFAULT_TOML fallback chain hard-codes model ids; D6 says configuration, never code.",
    ),
    Exception_(
        "glide/providers/config.py",
        "model-id",
        "eleven_v4_turbo",
        "DEFAULT_TOML fallback chain hard-codes model ids; D6 says configuration, never code.",
    ),
    Exception_(
        "glide/providers/config.py",
        "model-id",
        "jev-latest",
        "DEFAULT_TOML classifier chain hard-codes a classifier model id; D6 says configuration, never code.",
    ),
    Exception_(
        "glide/providers/stt.py",
        "endpoint",
        "https://api.elevenlabs.io",
        "ELEVENLABS_BASE_URL is a default vendor host in the adapter; D6 wants it in configuration.",
    ),
    Exception_(
        "glide/providers/stt.py",
        "model-id",
        "scribe_v2_realtime",
        "DEFAULT_REALTIME_MODEL is a default model id in the adapter; D6 wants it in configuration.",
    ),
    Exception_(
        "glide/providers/tts.py",
        "endpoint",
        "https://api.elevenlabs.io",
        "ELEVENLABS_BASE_URL is a default vendor host in the adapter; D6 wants it in configuration.",
    ),
    Exception_(
        "glide/providers/tts.py",
        "endpoint",
        "https://api.openai.com/v1",
        "OPENAI_BASE_URL is a default vendor host in the adapter; D6 wants it in configuration.",
    ),
    Exception_(
        "glide/providers/tts.py",
        "voice-id",
        "OPENAI_DEFAULT_VOICE = 'alloy'",
        "A default voice id in code; D6 says voice ids are configuration.",
    ),
    Exception_(
        "glide/providers/tts.py",
        "voice-id",
        "SAY_DEFAULT_VOICES = 'Samantha'",
        "Default macOS voice names in code (SAY_DEFAULT_VOICES); D6 says voices are configuration.",
    ),
    Exception_(
        "glide/providers/tts.py",
        "voice-id",
        "SAY_DEFAULT_VOICES = 'Sinji'",
        "Default macOS voice names in code (SAY_DEFAULT_VOICES); D6 says voices are configuration.",
    ),
    Exception_(
        "glide/providers/tts.py",
        "voice-id",
        "SAY_DEFAULT_VOICES = 'Tingting'",
        "Default macOS voice names in code (SAY_DEFAULT_VOICES); D6 says voices are configuration.",
    ),
)


# --- the tests on the real package ----------------------------------------------------------------------------


def test_package_files_are_found():
    files = [f for f in PACKAGE.rglob("*.py") if "__pycache__" not in f.parts]
    assert len(files) > 10, "the scan would pass vacuously: the package was not found"


def test_package_has_no_hardcoded_vendor_literals():
    problems = reconcile(scan_tree(), (*ALLOWLIST, *D6_DEBT))
    assert not problems, "hard-coded vendor literals (D6):\n  " + "\n  ".join(problems)


def test_every_exception_is_justified_and_unambiguous():
    entries = (*ALLOWLIST, *D6_DEBT)
    for entry in entries:
        assert entry.rule in RULES, entry
        assert len(entry.reason.split()) >= 4, f"give a real reason: {entry}"
        assert (ROOT / entry.path).is_file(), f"no such file: {entry.path}"
    keys = [e.key() for e in entries]
    assert len(keys) == len(set(keys)), "an exception is listed twice (or is both allowed and debt)"


# --- negative tests: the scanner catches what it claims to --------------------------------------------------------


def _rules(source: str) -> set[str]:
    return {h.rule for h in scan_source(source, "glide/x.py")}


@pytest.mark.parametrize(
    "model",
    [
        "claude-haiku-4-5",
        "claude-sonnet-5",
        "gpt-4o-mini",
        "gpt-6.1-sol",
        "chatgpt-writer",
        "gemini-3.5-flash-lite",
        "deepseek-chat",
        "deepseek/deepseek-v4.1-flash",
        "o3-mini",
        "o4-mini",
        "llama-3.3-70b",
        "llama3",
        "mistral-large-latest",
        "mixtral-8x7b",
        "qwen2.5-coder",
        "qwen/qwen3-32b",
        "grok-4",
        "meta-llama/llama-3.1-8b-instruct",
        "openai/gpt-oss-120b",
        "whisper-1",
        "scribe_v2_realtime",
        "eleven_v4_turbo",
        "eleven_multilingual_v2",
        "typesafe:jev-latest",
    ],
)
def test_model_ids_are_caught(model):
    assert "model-id" in _rules(f"MODEL = {model!r}\n")
    assert "model-id" in _rules(f"call(model={model!r})\n")
    assert "model-id" in _rules(f'TEXT = "use {model} here"\n')
    assert "model-id" in _rules(f'TEXT = f"{{x}} {model}"\n')


@pytest.mark.parametrize("model", ["o1", "o3", "o4", "O3"])
def test_a_bare_o_series_id_is_caught_only_as_a_whole_string(model):
    assert "model-id" in _rules(f"MODEL = {model!r}\n")
    assert _rules(f'TEXT = "use {model} here"\n') == set()


@pytest.mark.parametrize(
    "text",
    [
        "https://api.openai.com/v1",
        "https://openrouter.ai/api/v1",
        "https://api.deepseek.com",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "https://api.elevenlabs.io",
        "wss://api.elevenlabs.io/v1/speech-to-text/realtime",
        "api.anthropic.com",
        "http://models.internal.corp.com/v1/chat/completions",
        "https://models.acme-host.com/some/path",
        "https://console.typesafe.ai/",
        "https://model-host.local/v1",  # mDNS names resolve to real LAN services
        "https://models.internal/v1",
        "https://34.1.2.3/v1",
        "http://10.0.0.5:8080/v1/chat",
        "http://169.254.169.254/latest/meta-data",
        "https://[2001:4860:4860::8888]/v1",
        "wss://34.1.2.3:443/stream",
    ],
)
def test_endpoints_are_caught(text):
    assert "endpoint" in _rules(f"URL = {text!r}\n")


@pytest.mark.parametrize(
    "text",
    [
        "https://auth.example-idp.io/oauth2/authorize",
        "https://login.example-idp.com/oauth/token",
        "https://accounts.example-idp.com/.well-known/openid-configuration",
        "https://www.example-idp.com/.well-known/jwks.json",
        "openid profile email",
        "offline_access",
    ],
)
def test_oauth_urls_and_scopes_are_caught(text):
    assert "oauth" in _rules(f"X = {text!r}\n")


def test_oauth_named_assignments_are_caught():
    assert "oauth" in _rules('CLIENT_ID = "app-123"\n')
    assert "oauth" in _rules('ISSUER = "idp"\n')
    assert "oauth" in _rules('conf = dict(client_secret="shh-not-a-real-one")\n')
    assert "oauth" in _rules('def f(token_url="x"): ...\n')
    assert "oauth" in _rules('SCOPES = ("read", "write")\n')


def test_an_oauth_secret_is_caught_and_never_printed():
    value = "opaque-Zx9-secret-value"
    hits = scan_source(f"client_secret = {value!r}\n", "glide/x.py")
    assert {h.rule for h in hits} == {"oauth"}
    assert value not in "".join(str(h) for h in hits)


@pytest.mark.parametrize(
    "source",
    [
        'API_KEY = "opaque-value-123"\n',
        'password = "hunter2hunter2"\n',
        'call(api_key="opaque-value-123")\n',
        'def connect(token="opaque-value-123"): ...\n',
        'conf = {"access_key": "opaque-value-123"}\n',
        'self.auth_token = "opaque-value-123"\n',
        'SECRET = "opaque-value-123"\n',
        'api_key = options.get("api_key", "opaque-value-123")\n',
    ],
)
def test_credentials_assigned_to_key_named_fields_are_caught_and_never_printed(source):
    hits = scan_source(source, "glide/x.py")
    assert {h.rule for h in hits} == {"secret"}
    assert "opaque-value-123" not in str(hits) and "hunter2hunter2" not in str(hits)


@pytest.mark.parametrize(
    "source",
    [
        'API_KEY_ENV = "OPENAI_API_KEY"\n',
        'token_env = "GLIDE_TOKEN"\n',
        'api_key = ""\n',
        "api_key = None\n",
        "token = next(tokens)\n",
        'def f(api_key: str | None = None, password=""): ...\n',
        'x = {"token_count": "12"}\n',
    ],
)
def test_key_named_fields_that_hold_no_credential_are_not_flagged(source):
    assert scan_source(source, "glide/x.py") == []


@pytest.mark.parametrize(
    "source",
    [
        'DEFAULT_MODEL = "vendor-new"\n',
        'call(model="custom-v2")\n',
        'def f(model_id="custom-v2"): ...\n',
        'cfg = {"model": "custom-v2"}\n',
        'model = options.get("model", "custom-v2")\n',
    ],
)
def test_model_named_fields_are_caught_whatever_the_id_looks_like(source):
    assert "model-id" in _rules(source)


@pytest.mark.parametrize(
    "source",
    ['MODEL_ENV = "GLIDE_MODEL"\n', 'model = ""\n', "model = None\n", "def f(model: str | None = None): ...\n"],
)
def test_model_named_fields_that_name_no_model_are_not_flagged(source):
    assert scan_source(source, "glide/x.py") == []


@pytest.mark.parametrize(
    "source",
    [
        "PRICE_PER_MTOK = 3.0\n",
        "price_in = 0.25\n",
        "cost = dict(usd_per_million_input=5)\n",
        'RATES = {"price_out": 15.0}\n',
        'HELP = "costs $3 per million tokens"\n',
        'HELP = "$0.25/1M tokens"\n',
        "def f(price_per_token=0.000003): ...\n",
        "PRICE_PER_TOKEN = 3 / 1_000_000\n",
        "price_in = 1.5 * 2\n",
        "PRICE_OUT = -(15.0)\n",
        "price_in = 3 * 10**-6\n",
        "RATES = {'price_in': 3 / 1e6}\n",
    ],
)
def test_prices_are_caught(source):
    assert "price" in _rules(source)


@pytest.mark.parametrize(
    "source",
    [
        "price_cents = price * 100\n",
        "price = 0 / 5\n",
        "price = 1 / 0\n",
        "price = count + 1\n",
        "price = 2 ** 100\n",
    ],  # a huge power is not a price
)
def test_arithmetic_that_is_not_a_constant_price_is_not_flagged(source):
    assert scan_source(source, "glide/x.py") == []


@pytest.mark.parametrize(
    "source",
    [
        'VOICE_ID = "21m00Tcm4TlvDq8ikWAM"\n',
        'x = "21m00Tcm4TlvDq8ikWAM"\n',
        'OPENAI_DEFAULT_VOICE = "alloy"\n',
        'tts(voice="nova")\n',
        'def speak(text, voice="alloy"): ...\n',
        'voices = {"en": "abc", "fr": "def"}\n',
        'voice = options.get("voice", "alloy")\n',
        'voice = chosen or "alloy"\n',
        'DEFAULT_VOICE_NAME = "Samantha"\n',
        'voice_names = ["Samantha"]\n',
        'def speak(voice_name="Samantha"): ...\n',
    ],
)
def test_voice_ids_are_caught(source):
    assert "voice-id" in _rules(source)


def test_key_shaped_literals_are_caught_and_never_printed():
    key = "sk-" + "a1B2c3D4" * 4
    hits = scan_source(f"KEY = {key!r}\n", "glide/x.py")
    assert [h.rule for h in hits] == ["secret"]
    assert key not in str(hits[0])


def test_findings_carry_the_file_the_line_and_the_text():
    (hit,) = scan_source('x = 1\n\nMODEL = "claude-haiku-4-5"\n', "glide/where.py")
    assert (hit.path, hit.rule, hit.token, hit.line) == ("glide/where.py", "model-id", "claude-haiku-4-5", 3)


def test_a_nested_offender_is_found():
    source = "class A:\n    def f(self):\n        return {'k': [('x', 'gpt-6-luna')]}\n"
    assert "model-id" in _rules(source)


# --- and it does not cry wolf ---------------------------------------------------------------------------------


def test_docstrings_and_comments_are_skipped():
    source = '''"""Module docs: try claude-haiku-4-5 or https://api.openai.com/v1."""

# a comment naming gpt-4o and https://api.deepseek.com/chat
class A:
    """Docs may say deepseek-chat."""

    def f(self):
        """See https://openrouter.ai/api/v1 and eleven_v4_turbo."""
        return 1
'''
    assert scan_source(source, "glide/x.py") == []


def test_only_the_docstring_position_is_skipped():
    source = 'def f():\n    """ok"""\n    x = "claude-haiku-4-5"\n    "gpt-4o"\n    return x\n'
    assert {h.token for h in scan_source(source, "glide/x.py")} == {"claude-haiku-4-5", "gpt-4o"}


@pytest.mark.parametrize(
    "source",
    [
        'URL = "http://localhost:8080/v1/chat"\n',
        'URL = "http://127.0.0.1:11434"\n',
        'URL = "https://example.com/some/path"\n',
        'SITE = "https://github.com/"\n',
        'SITE = "https://www.notion.so/"\n',
        '_URL = re.compile(r"https?://\\S+")\n',
        'NAME = "os.path.join"\n',
        'NAME = "self.io.warn"\n',
        'NAME = "glide.computer.config"\n',
        'MAIL = "someone@example.org"\n',
        'PROVIDER = "gemini"\n',
        'PROVIDER = "deepseek"\n',
        'KEY_ENV = "OPENAI_API_KEY"\n',
        'VOICE_ENV = "GLIDE_VOICE"\n',
        'CLIENT_ID_ENV = "GLIDE_CLIENT_ID"\n',
        'voice = options.get("voice")\n',
        'def f(voice: str | None = None, language=""): ...\n',
        'VOICE = ""\n',
        'PATH = "/v1/text-to-speech/{voice_id}/stream"\n',
        'WORD = "echo"\n',
        'WORD = "o3 is a word here"\n',
        'WORD = "a long mixed identifier without digits ABCDEFGHIJKLMNOPQRSTUV"\n',
        'WORD = "ABCDEFGHIJKLMNOPQRSTUV"\n',
        "COUNT = 3\n",
        "cost = 2\n",
        'price_text = "free"\n',
        "tokens = 1000\n",
        'URL = "http://[::1]:8080/v1"\n',
        'URL = "http://0.0.0.0:8000"\n',
        'URL = "https://192.0.2.1/v1"\n',  # documentation ranges (RFC 5737, RFC 3849)
        'URL = "http://198.51.100.7:9000/v1"\n',
        'URL = "http://203.0.113.9/v1"\n',
        'URL = "http://[2001:db8::1]/v1"\n',
        'URL = "https://model-host.test/v1"\n',
        'URL = "http://svc.localhost:8000/v1"\n',
        'VERSION = "34.1.2.3"\n',  # no scheme: a version number, not an address
        'VERSION = "1.2.3.4"\n',
        'VOICE_NAME_ENV = "GLIDE_VOICE_NAME"\n',
    ],
)
def test_ordinary_code_is_not_flagged(source):
    assert scan_source(source, "glide/x.py") == []


# --- the exception mechanism ------------------------------------------------------------------------------------


def test_reconcile_flags_an_unlisted_literal():
    hits = scan_source('M = "claude-haiku-4-5"\n', "glide/x.py")
    assert reconcile(hits, []) == [f"not allowed: {hits[0]}"]


def test_reconcile_accepts_a_listed_literal_by_file_rule_and_text():
    hits = scan_source('\n\nM = "claude-haiku-4-5"\n', "glide/x.py")
    entry = Exception_("glide/x.py", "model-id", "claude-haiku-4-5", "a documented example kept on purpose")
    assert reconcile(hits, [entry]) == []


def test_reconcile_does_not_accept_the_same_text_in_another_file():
    hits = scan_source('M = "claude-haiku-4-5"\n', "glide/y.py")
    entry = Exception_("glide/x.py", "model-id", "claude-haiku-4-5", "a documented example kept on purpose")
    assert len(reconcile(hits, [entry])) == 2  # y.py is not allowed, and the x.py entry is stale


def test_reconcile_flags_a_stale_exception():
    entry = Exception_("glide/x.py", "model-id", "claude-haiku-4-5", "a documented example kept on purpose")
    (problem,) = reconcile([], [entry])
    assert problem.startswith("stale exception")


def test_a_file_that_does_not_parse_is_an_error_not_a_pass(tmp_path):
    package = tmp_path / "glide"
    package.mkdir()
    (package / "broken.py").write_text("def (:\n", encoding="utf-8")
    with pytest.raises(SyntaxError):
        scan_tree(package, tmp_path)


def test_scan_tree_reports_paths_relative_to_the_root(tmp_path):
    package = tmp_path / "glide" / "sub"
    package.mkdir(parents=True)
    (package / "m.py").write_text('M = "gpt-4o"\n', encoding="utf-8")
    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "skipped.py").write_text('M = "gpt-4o"\n', encoding="utf-8")
    assert [h.path for h in scan_tree(tmp_path / "glide", tmp_path)] == ["glide/sub/m.py"]
