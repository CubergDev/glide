"""glide.toml: from a small TOML file to working provider chains.

The file names providers (`[providers.<name>]`) and, for each job, an ordered chain of them
(`[llm.fast]`, `[llm.smart]`, `[stt]`, `[tts]`, `[classifier]`). `GlideConfig` turns that into the
facades the rest of Glide uses (`LLM`, `STT`, `TTS`, `ChainedClassifier`, `ChainWriter`), each over a
`Chain` (chain.py), so the user changes vendors by editing a file or an environment variable and no
code. `load_config` finds the file; `glide.toml.example` is the team's starting point.

What is structural and what is environmental, because the difference decides when an error is raised:

- A mistake in the file (an unknown provider, a kind the role cannot use, a misspelled key, a bad policy
  value) is a `ConfigError` at load, so it cannot hide until the one role that is affected is used.
- A slot whose key variable is unset is not a mistake. It is skipped with a warning, so a partial setup
  still runs, and it stays on record (`GlideConfig.skipped`, `slots()`) so `glide doctor` can say so.
  Only a role with no usable slot left is an error, and it is raised when that role is asked for
  (`llm()`, `stt()`, ...), naming the variables to set. Raising it at load would make `glide doctor`
  unable to print its table on exactly the machine that needs it.

Keys are only ever read from the environment variable a provider names, when a slot is built. They are never
written in the file (an `api_key` entry is refused without echoing it), never copied onto an attribute (only the
environment mapping is held), and `scrub()` removes them from any text that is about to be shown.

Built-in presets cover the common vendors, so a file only has to name the key variable it differs on. A
chain entry is "provider:model" (split at the FIRST colon, since OpenRouter ids may contain more) or an
inline table `{provider=, model=, options={...}, name=}` for options on one slot, such as `reasoning_effort`.
In the classifier chain the entry "llm.fast" (or "llm.smart") means the classifier prompt run over that LLM chain.

A glide.toml found in the current directory (not `--config`, not $GLIDE_CONFIG, not the user's own
~/.config/glide/glide.toml) is somebody else's file until the user says otherwise: a repository can ship one. It is
used, and said to be (a notice in `GlideConfig.warnings` names the file), but a key is never sent from it to a host
the user did not name: the vendors Glide knows by name, the hosts in the user's own config and this machine are
trusted, any other host makes that slot "skipped" with the reason. See `load_config`.

Pinning, with no code change: GLIDE_PIN_LLM_FAST, GLIDE_PIN_LLM_SMART, GLIDE_PIN_STT, GLIDE_PIN_TTS and
GLIDE_PIN_CLASSIFIER name a slot (its full name or a prefix naming one); a trailing "!" makes it strict, so
nothing else is ever tried. `GlideConfig.pin` and `unpin` do the same at runtime. A pin that cannot be honoured
is an error that names the slots there are, never a quiet fallback.
"""

from __future__ import annotations

import contextlib
import ipaddress
import logging
import os
import re
import threading
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from . import classifier as classifier_mod
from . import llm as llm_mod
from . import stt as stt_mod
from . import tts as tts_mod
from .base import ProviderSpec
from .chain import ORDERS, Chain, ChainPolicy, Slot, SwitchEvent
from .classifier import ChainedClassifier, LLMClassifier
from .errors import ProviderError, redact
from .llm import LLM
from .stt import STT
from .tts import TTS
from .writer_client import ChainWriter, UnavailableFacade

log = logging.getLogger("glide.config")

ROLES = ("llm.fast", "llm.smart", "stt", "tts", "classifier")
# Two more LLM jobs a file may give a chain of their own. A file that does not stands them on llm.smart (the same
# chain, the same health and pin, nothing built twice), so no default chain, and so no default model, is added here.
EXTRA_LLM_ROLES = ("llm.planner", "llm.research")
ALL_ROLES = (*ROLES, *EXTRA_LLM_ROLES)
LLM_SHORT = ("fast", "smart", "planner", "research")
# What each job can be served by. Validated at load, whatever builders are installed.
ROLE_KINDS = {
    "llm": ("openai_compat",),
    "stt": ("elevenlabs", "openai_compat"),
    "tts": ("elevenlabs", "openai_compat", "macos_say"),
    "classifier": ("typesafe", "openai_compat"),
}
KINDS = ("openai_compat", "elevenlabs", "typesafe", "macos_say")
NO_MODEL_NEEDED = ("macos_say", "typesafe")  # the adapter has a default
CONFIG_NAME = "glide.toml"
PIN_PREFIX = "GLIDE_PIN_"
MIN_SECRET = 4  # shorter than this and scrubbing a "key" would mangle ordinary words
_ENV_NAME = re.compile(r"[A-Z_][A-Z0-9_]*")
_CREDENTIAL_HEADER = re.compile(r"authorization|key|token|secret|password|cookie", re.IGNORECASE)
_HEADER_OPTIONS = ("extra_headers", "headers")  # an llm slot's and a typesafe slot's
_CLASSIFIER_OPTIONS = ("max_tokens", "temperature", "timeout")  # LLMClassifier's own, from a slot's options

# The vendors Glide knows by name. The base URLs were read from each vendor's documentation, and the key variables
# are the vendors' own conventions. DeepSeek's documented base URL has no /v1 (its request path is /chat/completions),
# so none is added. "typesafe" repeats the installed SDK's own constants, and a test pins them to it.
PRESETS: dict[str, ProviderSpec] = {
    "openai": ProviderSpec("openai", "openai_compat", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "openrouter": ProviderSpec("openrouter", "openai_compat", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "deepseek": ProviderSpec("deepseek", "openai_compat", "https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "gemini": ProviderSpec(
        "gemini", "openai_compat", "https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"
    ),
    # No base_url: each ElevenLabs adapter has its own default host (a regional one may be set in a file).
    "elevenlabs": ProviderSpec("elevenlabs", "elevenlabs", "", "ELEVENLABS_API_KEY"),
    "typesafe": ProviderSpec("typesafe", "typesafe", "https://api.typesafe.ai", "TYPESAFE_API_KEY"),
    "macos_say": ProviderSpec("macos_say", "macos_say"),  # no key: a slot with no key variable is never skipped
}


def _host(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def _loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


# The hosts a project-local glide.toml may send a key to without the user having named them: the vendors above.
PRESET_HOSTS = frozenset(_host(spec.base_url) for spec in PRESETS.values() if spec.base_url)

POLICY_KEYS = ("order", "fail_threshold", "cooldown_s", "auth_cooldown_s", "hedge_after_s", "latency_alpha")
SPEECH_KEYS = ("language", "silence_ms", "headset", "vad_model_path", "vad_model_url", "vad_model_sha256")
# Tables another module reads for itself (`glide.memory.settings`, `glide.mcp.config`, `glide.webhooks.cli`, `glide.computer.browser_settings`,
# `glide.computer.config`, `glide.routing.settings`): known
# here so the file does not warn about them, and not parsed here, so the one reader of each stays the only one.
OWN_TABLES = ("memory", "mcp", "webhooks", "browser", "research", "routing")
KNOWN_TABLES = ("providers", "llm", "stt", "tts", "classifier", "speech", *OWN_TABLES)
_SHA256 = re.compile(r"[0-9a-f]{64}")

# Used when no glide.toml is found, and for any role a file leaves out. They are data (default.toml, shipped inside
# the package because glide.toml.example sits outside it), never code: the model ids in it are the team's starting
# point and are unverified until `glide doctor --live` has run with real keys. A test keeps these chains identical
# to the example's.
DEFAULT_TOML = resources.files(__package__).joinpath("default.toml").read_text(encoding="utf-8")


class ConfigError(ValueError):
    """The configuration cannot be used. `missing` names the environment variables whose absence is the reason.

    Messages name files, tables, slots and variables, never a key's value.
    """

    def __init__(self, message: str, *, missing: tuple[str, ...] = ()):
        super().__init__(message)
        self.missing = tuple(missing)


class NoUsableProvider(ConfigError):
    """A role has no slot left to use: every one was skipped (no key, no voice). Not a mistake in the file."""


# ---------------------------------------------------------------------------------------------
# The file, parsed and checked
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SlotSpec:
    """One entry of a chain, as written. `uses` is a role the slot reuses (the classifier over an LLM chain)."""

    name: str
    provider: str = ""
    model: str = ""
    options: dict = field(default_factory=dict)
    uses: str | None = None


@dataclass(frozen=True)
class RoleSpec:
    role: str
    slots: tuple[SlotSpec, ...]
    policy: ChainPolicy = field(default_factory=ChainPolicy)
    deadline_s: float | None = None  # the longest one request of this role may take; the request's own limit otherwise


@dataclass(frozen=True)
class SpeechSettings:
    """The `[speech]` table: what a voice session needs that is not one provider slot's own.

    Voice ids and model ids belong to the tts and stt slots (their `options`). What is here is the rest, and each
    field is None until the file says so: nothing has a built-in value, and `vad_model()` says what is missing.
    """

    language: str | None = None  # the spoken language, as the stt and tts slots take it
    silence_ms: int | None = None  # how long a pause ends an utterance
    headset: bool | None = None  # whether the speaker may be interrupted while it plays
    vad_model_path: str | None = None  # a voice-activity model on disk
    vad_model_url: str | None = None  # where to fetch it once, when it is not on disk
    vad_model_sha256: str | None = None  # the digest the file must have, whether fetched or already there

    def vad_model(self) -> tuple[str, str | None, str]:
        """(path, url, sha256) of the voice-activity model, or a ConfigError naming what `[speech]` lacks."""
        missing = [k for k, v in (("vad_model_path", self.vad_model_path), ("vad_model_sha256", self.vad_model_sha256)) if not v]
        if missing:
            raise ConfigError(f"[speech] needs {' and '.join(missing)}: the voice-activity model is not configured")
        return self.vad_model_path or "", self.vad_model_url, self.vad_model_sha256 or ""


@dataclass(frozen=True)
class SlotInfo:
    """What became of one slot: built and usable ("ready") or left out ("skipped", with the reason).

    `env_var` is the variable that holds the slot's key, whether or not it is set. `missing` lists the variables
    to set for a skipped slot. `short` is the reason in a few words, for a table. `client` is the adapter, for
    `glide doctor`; it holds the key, so it stays out of the repr.
    """

    role: str
    name: str
    provider: str
    model: str
    state: str
    reason: str = ""
    short: str = ""
    env_var: str = ""
    missing: tuple[str, ...] = ()
    options: dict = field(default_factory=dict, repr=False)
    client: Any = field(default=None, repr=False, compare=False)


def _where(role: str) -> str:
    return f"[{role}]"


def _table(value: object, where: str) -> dict:
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be a table")
    return value


def _text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where} must be a non-empty string")
    return value.strip()


def _only(table: Mapping, allowed: tuple[str, ...], where: str) -> None:
    """Refuse a key that is not known: a misspelt `hedge_after` would otherwise be ignored without a word."""
    for key in table:
        if key not in allowed:
            hint = ""
            if key in ("api_key", "key", "token", "secret"):
                hint = ": keys are never written in this file; name the environment variable with api_key_env"
            raise ConfigError(f"{where} has an unknown key {key!r}{hint} (known: {', '.join(allowed)})")


def _options(value: object, where: str) -> dict:
    """An `options` table, refused when a header in it would carry a credential: keys come from the environment only."""
    options = _table(value, where)
    for option in _HEADER_OPTIONS:
        headers = options.get(option)
        for name in headers if isinstance(headers, dict) else ():
            if _CREDENTIAL_HEADER.search(str(name)):
                raise ConfigError(
                    f"{where} {option} sets the header {str(name)!r}, which looks like a credential: keys are never written "
                    "in this file; name the environment variable with api_key_env"
                )
    return options


def _providers(table: Mapping) -> dict[str, ProviderSpec]:
    """The presets, overlaid by the file's `[providers.<name>]` tables field by field.

    Writing only `api_key_env` under a preset's name keeps its kind and base_url. Changing the kind starts
    from nothing, since a preset's URL means nothing to another kind of server.
    """
    specs = dict(PRESETS)
    for name, raw in _table(table, "[providers]").items():
        where = f"[providers.{name}]"
        entry = _table(raw, where)
        _only(entry, ("kind", "base_url", "api_key_env", "options"), where)
        preset = PRESETS.get(name)
        kind = entry.get("kind") or (preset.kind if preset else None)
        if kind is None:
            raise ConfigError(f"{where} needs a kind: one of {', '.join(KINDS)}")
        if kind not in KINDS:
            raise ConfigError(f"{where} kind must be one of {', '.join(KINDS)}, not {kind!r}")
        if preset is not None and preset.kind != kind:
            preset = None
        base_url = entry.get("base_url", preset.base_url if preset else "")
        env_var = entry.get("api_key_env", preset.api_key_env if preset else "")
        if not isinstance(base_url, str) or not isinstance(env_var, str):
            raise ConfigError(f"{where} base_url and api_key_env must be strings")
        if env_var and not _ENV_NAME.fullmatch(env_var):
            # Said without the value: someone who pasted a key here must not see it again in a log.
            raise ConfigError(f"{where} api_key_env must be the NAME of an environment variable (capitals, digits, _), not a key")
        if kind in ("elevenlabs", "typesafe") and not env_var:
            raise ConfigError(f"{where} needs api_key_env: a {kind} provider cannot be used without a key")
        options = _options(entry.get("options", {}), f"{where} options")
        specs[name] = ProviderSpec(name, kind, base_url.strip(), env_var, {**(preset.options if preset else {}), **options})
    return specs


def _number(value: object, where: str, *, low: float, high: float | None = None, integer: bool = False) -> float:
    kind = int if integer else int | float
    if isinstance(value, bool) or not isinstance(value, kind):
        raise ConfigError(f"{where} must be {'a whole number' if integer else 'a number'}")
    if value < low or (high is not None and value > high):
        raise ConfigError(f"{where} must be {'at least ' + str(low) if high is None else f'from {low} to {high}'}")
    return value


def _policy(table: Mapping, where: str) -> ChainPolicy:
    out: dict[str, Any] = {}
    if "order" in table:
        order = table["order"]
        if order not in ORDERS:
            raise ConfigError(f"{where} order must be one of {', '.join(ORDERS)}, not {order!r}")
        out["order"] = order
    if "fail_threshold" in table:
        out["fail_threshold"] = _number(table["fail_threshold"], f"{where} fail_threshold", low=1, integer=True)
    for key in ("cooldown_s", "auth_cooldown_s"):
        if key in table:
            out[key] = float(_number(table[key], f"{where} {key}", low=0))
    if "hedge_after_s" in table:
        value = float(_number(table["hedge_after_s"], f"{where} hedge_after_s", low=0))
        if value == 0:
            raise ConfigError(f"{where} hedge_after_s must be more than 0 (leave it out to switch hedging off)")
        out["hedge_after_s"] = value
    if "latency_alpha" in table:
        value = float(_number(table["latency_alpha"], f"{where} latency_alpha", low=0, high=1))
        if value == 0:
            raise ConfigError(f"{where} latency_alpha must be more than 0")
        out["latency_alpha"] = value
    return ChainPolicy(**out)


def _slot(entry: object, role: str, index: int, providers: Mapping[str, ProviderSpec]) -> SlotSpec:
    where = f"{_where(role)} chain entry {index + 1}"
    family = role.split(".")[0]
    if isinstance(entry, str):
        text = _text(entry, where)
        if text in ROLES:
            if role != "classifier" or not text.startswith("llm."):
                raise ConfigError(
                    f"{where}: {text!r} can only appear in the classifier chain, as the classifier over an LLM chain"
                )
            return SlotSpec(name=text, uses=text)
        provider, _, model = text.partition(":")  # the first colon only: OpenRouter ids may contain others
        entry_options: dict = {}
        label = None
    else:
        table = _table(entry, where)
        _only(table, ("provider", "model", "options", "name"), where)
        provider = _text(table.get("provider"), f"{where} provider")
        model = table.get("model", "")
        if not isinstance(model, str):
            raise ConfigError(f"{where} model must be a string")
        entry_options = _options(table.get("options", {}), f"{where} options")
        label = _text(table["name"], f"{where} name") if "name" in table else None
    provider, model = provider.strip(), model.strip()
    spec = providers.get(provider)
    if spec is None:
        raise ConfigError(f"{where}: no provider named {provider!r} (known: {', '.join(sorted(providers))})")
    if spec.kind not in ROLE_KINDS[family]:
        raise ConfigError(
            f"{where}: {provider!r} is a {spec.kind} provider, which cannot serve {role} (it takes: {', '.join(ROLE_KINDS[family])})"
        )
    if not model and spec.kind not in NO_MODEL_NEEDED:
        raise ConfigError(f"{where}: {provider!r} needs a model, written {provider}:<model>")
    return SlotSpec(
        name=label or (f"{provider}:{model}" if model else provider), provider=provider, model=model, options=entry_options
    )


def _role(role: str, table: object, providers: Mapping[str, ProviderSpec]) -> RoleSpec:
    where = _where(role)
    entry = _table(table, where)
    _only(entry, ("chain", *POLICY_KEYS, *(("deadline_s",) if role.startswith("llm.") else ())), where)
    chain = entry.get("chain")
    if not isinstance(chain, list) or not chain:
        raise ConfigError(f'{where} needs chain = [...], a list of "provider:model" entries or inline tables')
    slots = tuple(_slot(item, role, i, providers) for i, item in enumerate(chain))
    names = [s.name for s in slots]
    for name in names:
        if names.count(name) > 1:
            raise ConfigError(f'{where} lists {name!r} twice; give one entry a distinct name = "..."')
    deadline = float(_number(entry["deadline_s"], f"{where} deadline_s", low=0)) if "deadline_s" in entry else None
    if deadline == 0:
        raise ConfigError(f"{where} deadline_s must be more than 0 (leave it out to use each request's own limit)")
    return RoleSpec(role, slots, _policy(entry, where), deadline)


def _roles(data: Mapping, providers: Mapping[str, ProviderSpec]) -> dict[str, RoleSpec]:
    found: dict[str, object] = {}
    llm = data.get("llm")
    if llm is not None:
        for key, table in _table(llm, "[llm]").items():
            if f"llm.{key}" not in ALL_ROLES:
                raise ConfigError(
                    f"[llm.{key}] is not a role (the LLM roles are {', '.join(r for r in ALL_ROLES if r.startswith('llm.'))})"
                )
            found[f"llm.{key}"] = table
    for role in ("stt", "tts", "classifier"):
        if role in data:
            found[role] = data[role]
    return {role: _role(role, table, providers) for role, table in found.items()}


def _voice(entry: Mapping):
    """The `[speech]` table as the voice stack reads it (`glide.speech.settings`), validated now rather than at `glide voice`."""
    from ..speech.settings import SpeechSettings as VoiceSettings

    return VoiceSettings.from_mapping(entry)


def _speech(table: object) -> SpeechSettings:
    """The part of `[speech]` that is about providers, checked first so its messages stay what they were.

    The voice-only keys (merge_window_s, idle_s, vad, ...) are known here too and checked by `_voice`.
    """
    from ..speech.settings import SpeechSettings as VoiceSettings

    where = "[speech]"
    entry = _table(table, where)
    _only(entry, tuple(dict.fromkeys((*SPEECH_KEYS, *(f.name for f in fields(VoiceSettings))))), where)
    out: dict[str, Any] = {}
    if "language" in entry:
        out["language"] = _text(entry["language"], f"{where} language")
    if "silence_ms" in entry:
        out["silence_ms"] = int(_number(entry["silence_ms"], f"{where} silence_ms", low=1, integer=True))
    if "headset" in entry:
        if not isinstance(entry["headset"], bool):
            raise ConfigError(f"{where} headset must be true or false")
        out["headset"] = entry["headset"]
    for key in ("vad_model_path", "vad_model_url"):
        if key in entry:
            out[key] = _text(entry[key], f"{where} {key}")
    if "vad_model_url" in out and not out["vad_model_url"].startswith("https://"):
        raise ConfigError(f"{where} vad_model_url must be an https URL")
    if "vad_model_sha256" in entry:
        digest = _text(entry["vad_model_sha256"], f"{where} vad_model_sha256").lower()
        if not _SHA256.fullmatch(digest):
            raise ConfigError(f"{where} vad_model_sha256 must be 64 hexadecimal digits")
        out["vad_model_sha256"] = digest
    return SpeechSettings(**out)


def _default_roles() -> dict[str, RoleSpec]:
    data = tomllib.loads(DEFAULT_TOML)
    return _roles(data, PRESETS)


# ---------------------------------------------------------------------------------------------
# The configuration object
# ---------------------------------------------------------------------------------------------


class _Lent:
    """A classifier slot lent to a `ChainedClassifier`, with no `close` and no `__exit__`.

    `runner.run()` closes the classifier it is given, and `ChainedClassifier.close` closes every slot. If the
    chain held the real clients, the first run would leave the second with closed connections. So the chain
    holds these, a run's close reaches nothing, and `GlideConfig.close()` closes the real clients.
    """

    def __init__(self, client: Any, name: str):
        self._client = client
        self.name = name
        self.model = str(getattr(client, "model", "") or "")

    def __repr__(self) -> str:
        return f"<classifier slot {self.name}>"

    def system_one(self, **request: Any) -> Any:
        return self._client.system_one(**request)


def _llm_role(role: str) -> str:
    name = role if role.startswith("llm.") else f"llm.{role}"
    if name not in ALL_ROLES:
        raise ConfigError(f"the LLM role must be one of {', '.join(LLM_SHORT)}, not {role!r}")
    return name


def _canon(role: str) -> str:
    """`fast`, `smart`, `planner` and `research` are short for llm.fast and so on."""
    return f"llm.{role}" if role in LLM_SHORT else role


def pin_variable(role: str) -> str:
    """The environment variable that pins a role: llm.fast is GLIDE_PIN_LLM_FAST."""
    return PIN_PREFIX + role.upper().replace(".", "_")


class GlideConfig:
    """The providers and chains of one glide.toml, built on demand and kept.

    A facade is built the first time its role is asked for and then reused, because a chain carries state
    that has to live as long as the process: which slots are resting, the measured latencies, the pin.
    `chains` holds the roles that could be built; `chain(role)` says why one cannot.

    `builders` maps (job, provider kind) to the function that builds an adapter, `build_client(spec, model,
    api_key, options)`. They default to the real ones and exist so a test can put fakes in their place.
    """

    def __init__(
        self,
        providers: Mapping[str, ProviderSpec],
        roles: Mapping[str, RoleSpec],
        *,
        env: Mapping[str, str],
        source: str,
        defaulted: tuple[str, ...] = (),
        warnings: tuple[str, ...] = (),
        builders: Mapping[tuple[str, str], Callable[..., Any]] | None = None,
        speech: SpeechSettings | None = None,
        voice: Any = None,
        trusted_hosts: frozenset[str] | None = None,
    ):
        self.providers = dict(providers)
        self.speech = speech or SpeechSettings()
        self.voice = voice if voice is not None else _voice({})  # what `glide.speech.session.build_voice` takes
        self.roles = dict(roles)
        self.source = source
        self.defaulted = tuple(defaulted)  # roles the file left out, served by the built-in chains
        self.warnings = list(warnings)
        self._trusted_hosts = trusted_hosts  # None: the file is the user's own; else the hosts a key may be sent to
        self._env = env
        self._builders: dict[tuple[str, str], Callable[..., Any]] = {
            ("llm", "openai_compat"): llm_mod.build_client,
            ("stt", "elevenlabs"): stt_mod.build_client,
            ("stt", "openai_compat"): stt_mod.build_client,
            ("tts", "elevenlabs"): tts_mod.build_client,
            ("tts", "openai_compat"): tts_mod.build_client,
            ("tts", "macos_say"): tts_mod.build_client,
            ("classifier", "typesafe"): classifier_mod.build_client,
            **(builders or {}),
        }
        self._lock = threading.RLock()
        self._listener_lock = threading.Lock()
        self._listeners: list[Callable[[SwitchEvent], None]] = []
        self._infos: dict[str, list[SlotInfo]] = {}
        self._facades: dict[str, Any] = {}
        self._pins: dict[str, tuple[str, bool]] = {}

    @classmethod
    def from_dict(
        cls,
        data: Mapping,
        *,
        env: Mapping[str, str] | None = None,
        source: str = "<dict>",
        builders: Mapping[tuple[str, str], Callable[..., Any]] | None = None,
        trusted_hosts: frozenset[str] | None = None,
    ) -> GlideConfig:
        """A configuration from an already parsed glide.toml. `env` defaults to the process environment.

        `trusted_hosts` is for a file that is not the user's own (see the module docstring): the hosts, besides the
        vendors Glide knows by name and this machine, that a key may be sent to. Left out, the file is trusted.
        """
        providers = _providers(data.get("providers", {}))
        roles = _roles(data, providers)
        warnings = tuple(f"{source}: ignoring the unknown table [{key}]" for key in data if key not in KNOWN_TABLES)
        if trusted_hosts is not None:
            warnings += (
                f"{source}: read from the current directory, so a key is sent only to the vendors Glide knows, "
                "this machine, and hosts named in your own configuration (~/.config/glide/glide.toml)",
            )
        for warning in warnings:
            log.warning(warning)
        # Chains the file leaves out come from the built-in defaults. They name presets, and a file that
        # overlays a preset (say, another key variable) changes it for them too.
        defaults = {role: spec for role, spec in _default_roles().items() if role not in roles}
        return cls(
            providers,
            {**defaults, **roles},
            env=os.environ if env is None else env,
            source=source,
            defaulted=tuple(role for role in ROLES if role in defaults),
            warnings=warnings,
            builders=builders,
            speech=_speech(data["speech"]) if "speech" in data else None,
            voice=_voice(data["speech"]) if "speech" in data else None,
            trusted_hosts=None if trusted_hosts is None else frozenset(trusted_hosts | PRESET_HOSTS),
        )

    @classmethod
    def from_toml(
        cls,
        text: str,
        *,
        env: Mapping[str, str] | None = None,
        source: str = "<toml>",
        builders: Mapping[tuple[str, str], Callable[..., Any]] | None = None,
        trusted_hosts: frozenset[str] | None = None,
    ) -> GlideConfig:
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f"{source} is not valid TOML: {e}") from None
        return cls.from_dict(data, env=env, source=source, builders=builders, trusted_hosts=trusted_hosts)

    def __repr__(self) -> str:
        return f"<GlideConfig {self.source}: {', '.join(self.roles)}>"  # nothing from the environment, ever

    def __enter__(self) -> GlideConfig:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- keys ----------------------------------------------------------------------------------

    def _key(self, env_var: str) -> str:
        return (self._env.get(env_var) or "").strip()

    def scrub(self, text: str) -> str:
        """`text` with every key this configuration can read replaced by '***', for anything about to be shown."""
        keys = (self._key(spec.api_key_env) for spec in self.providers.values() if spec.api_key_env)
        return redact(text, keys, "***", min_len=MIN_SECRET)

    # -- the slots -----------------------------------------------------------------------------

    def _resolve(self, role: str) -> str:
        """The role as the file defines it: planner and research stand on smart where the file gives them no chain."""
        role = _canon(role)
        return "llm.smart" if role in EXTRA_LLM_ROLES and role not in self.roles else role

    def _spec(self, role: str) -> RoleSpec:
        role = self._resolve(role)
        if role not in ALL_ROLES:
            raise ConfigError(f"{role!r} is not a role (the roles are {', '.join(ALL_ROLES)})")
        return self.roles[role]

    def slots(self, role: str) -> list[SlotInfo]:
        """Every slot of a role in listed order, the skipped ones too. Builds the usable ones once.

        Raises ConfigError for a slot whose own options the adapter refuses.
        """
        role = self._resolve(role)
        with self._lock:
            if role not in self._infos:
                infos = [self._open(role, slot) for slot in self._spec(role).slots]
                self._infos[role] = infos
                for info in infos:
                    if info.state == "skipped":
                        log.warning("skipping %s for %s: %s", info.name, role, info.reason)
            return list(self._infos[role])

    @property
    def skipped(self) -> list[SlotInfo]:
        """The slots left out of the roles resolved so far, with the variable each one needs."""
        with self._lock:
            return [i for infos in self._infos.values() for i in infos if i.state == "skipped"]

    def _open(self, role: str, slot: SlotSpec) -> SlotInfo:
        family = role.split(".")[0]
        if slot.uses:
            try:
                client = LLMClassifier(self._facade(slot.uses), name=slot.name)
            except NoUsableProvider as e:
                return SlotInfo(
                    role,
                    slot.name,
                    "",
                    "",
                    "skipped",
                    f"{slot.uses} has no usable provider",
                    f"{slot.uses} unusable",
                    missing=e.missing,
                )
            return SlotInfo(role, slot.name, "", "", "ready", client=client)
        spec = self.providers[slot.provider]
        if spec.kind not in ROLE_KINDS[family]:  # a file that turned a preset into another kind, under a built-in chain
            raise ConfigError(
                f"{_where(role)} {slot.name}: {slot.provider!r} is a {spec.kind} provider, which cannot serve {role}"
            )
        options = {**spec.options, **slot.options}
        left_out = {"role": role, "name": slot.name, "provider": slot.provider, "model": slot.model, "state": "skipped"}
        key = ""
        if spec.api_key_env:
            key = self._key(spec.api_key_env)
            if not key:
                return SlotInfo(
                    **left_out,
                    reason=f"{spec.api_key_env} is not set",
                    short="no key",
                    env_var=spec.api_key_env,
                    missing=(spec.api_key_env,),
                    options=options,
                )
        if key and not self._may_send_key_to(spec):
            return SlotInfo(
                **left_out,
                reason=(
                    f"{spec.api_key_env} will not be sent to {_host(spec.base_url)}: this glide.toml is from the current "
                    "directory and that host is not one you named in your own configuration"
                ),
                short="untrusted host",
                env_var=spec.api_key_env,
                options=options,
            )
        if family == "tts" and spec.kind == "elevenlabs" and not (options.get("voice") or options.get("voices")):
            # A voice is a library id, not a name, and ElevenLabs has no default one. Without this the slot would
            # fail on every sentence before the next one spoke.
            return SlotInfo(
                **left_out,
                reason="no voice configured (set options.voice to a voice id, or options.voices)",
                short="no voice",
                env_var=spec.api_key_env,
                options=options,
            )
        try:
            client = self._build(family, spec, slot, key)
        except (ValueError, TypeError, ProviderError) as e:
            detail = self.scrub(str(e)) or type(e).__name__
            raise ConfigError(f"{_where(role)} {slot.name} cannot be set up: {detail}") from None
        return SlotInfo(
            role, slot.name, slot.provider, slot.model, "ready", env_var=spec.api_key_env, options=options, client=client
        )

    def _may_send_key_to(self, spec: ProviderSpec) -> bool:
        host = _host(spec.base_url)
        return self._trusted_hosts is None or not host or _loopback(host) or host in self._trusted_hosts

    def _build(self, family: str, spec: ProviderSpec, slot: SlotSpec, key: str) -> Any:
        options = dict(slot.options)
        if family == "classifier" and spec.kind == "openai_compat":
            # Any chat model can classify: the same endpoint, wrapped in the classifier prompt.
            inner = self._builders[("llm", "openai_compat")](spec, slot.model, key, options)
            extra = {k: options[k] for k in _CLASSIFIER_OPTIONS if k in options}
            return LLMClassifier(inner, name=slot.name, close_llm=True, **extra)
        return self._builders[(family, spec.kind)](spec, slot.model, key, options)

    # -- the facades ---------------------------------------------------------------------------

    def _facade(self, role: str) -> Any:
        role = self._resolve(role)
        self._spec(role)  # an unknown role is refused here, with the list of roles
        with self._lock:
            if role in self._facades:
                return self._facades[role]
            infos = self.slots(role)
            ready = [i for i in infos if i.state == "ready"]
            if not ready:
                raise self._unusable(role, infos)
            family = role.split(".")[0]
            slots = [Slot(i.name, _Lent(i.client, i.name) if family == "classifier" else i.client) for i in ready]
            chain = Chain(role, slots, self.roles[role].policy, on_event=self._dispatch)
            self._pin_from_environment(role, chain)
            extra = {"deadline_s": self.roles[role].deadline_s} if family == "llm" else {}
            facade = {"llm": LLM, "stt": STT, "tts": TTS, "classifier": ChainedClassifier}[family](chain, **extra)
            self._facades[role] = facade
            return facade

    def _unusable(self, role: str, infos: list[SlotInfo]) -> ConfigError:
        missing = tuple(dict.fromkeys(v for i in infos for v in i.missing))
        parts = [f"no usable {role} provider."]
        if missing:
            parts.append(f"Set at least one of: {', '.join(missing)}.")
        parts.append("Skipped: " + "; ".join(f"{i.name} ({i.reason})" for i in infos) + ".")
        if role in self.defaulted:
            parts.append(f"This chain is the built-in default; add [{role}] to {CONFIG_NAME} to change it.")
        return NoUsableProvider(" ".join(parts), missing=missing)

    def llm(self, role: str = "fast") -> LLM:
        """The LLM facade for `fast`, `smart`, `planner` or `research` (the last two are `smart` unless the file says)."""
        return self._facade(_llm_role(role))

    def stt(self) -> STT:
        return self._facade("stt")

    def tts(self) -> TTS:
        return self._facade("tts")

    def classifier(self) -> ChainedClassifier:
        """The classifier, one for the life of the configuration.

        Its slots are lent, so closing it (as `runner.run()` does on exit) leaves the real clients open for the
        next run, and the chain's health and pin carry over. `close()` of the configuration ends them.
        """
        return self._facade("classifier")

    def writer(self, *, timeout: float | None = None) -> ChainWriter:
        """The writer's one `generate` call, answered by the LLM chains: by role, fast, smart, planner and research."""
        deadlines = {
            role.split(".")[1]: spec.deadline_s
            for role in ALL_ROLES
            if role.startswith("llm.") and (spec := self.roles.get(role)) is not None and spec.deadline_s is not None
        }
        return ChainWriter(
            self.llm("fast"),
            self.llm("smart"),
            planner=self._optional_llm("planner"),
            research=self._optional_llm("research"),
            timeout=timeout,
            deadlines=deadlines,
        )

    def _optional_llm(self, role: str) -> Any:
        """The facade of planner or research. A chain the file gave it that has no usable slot (a missing key) fails
        only the requests that need it; fast and smart, which every task needs, are still required by `writer`."""
        try:
            return self.llm(role)
        except NoUsableProvider as error:
            return UnavailableFacade(str(error))

    def chain(self, role: str) -> Chain:
        """The chain of a role, for status, pinning and events. Raises ConfigError when no slot is usable."""
        return self._facade(role).chain

    @property
    def active_roles(self) -> tuple[str, ...]:
        """The roles that have a chain: the five every file has, and planner and research only when the file gives them one."""
        return (*ROLES, *(r for r in EXTRA_LLM_ROLES if r in self.roles))

    @property
    def chains(self) -> dict[str, Chain]:
        """The chains of every role that can be built, keyed llm.fast, llm.smart, stt, tts, classifier, and
        llm.planner and llm.research when the file gives them a chain."""
        out: dict[str, Chain] = {}
        for role in self.active_roles:
            with contextlib.suppress(NoUsableProvider):  # `chain(role)` raises it, with the reason
                out[role] = self.chain(role)
        return out

    def close(self) -> None:
        """Close every adapter that holds a connection. The first failure is raised after all have been closed."""
        failure: Exception | None = None
        with self._lock:
            infos = [i for group in self._infos.values() for i in group]
        for info in infos:
            close = getattr(info.client, "close", None)
            if close is None:
                continue
            try:
                close()
            except Exception as e:
                failure = failure or e
        if failure is not None:
            raise failure

    # -- switches ------------------------------------------------------------------------------

    def on_switch(self, callback: Callable[[SwitchEvent], None]) -> Callable[[], None]:
        """Call `callback(event)` for every switch on every chain, now or built later. Returns a function that removes it.

        Each chain has one `on_event` slot, so every chain is given the same dispatcher, and one listener raising
        does not stop the others from hearing the event.
        """
        with self._listener_lock:
            self._listeners.append(callback)

        def remove() -> None:
            with self._listener_lock, contextlib.suppress(ValueError):
                self._listeners.remove(callback)

        return remove

    def _dispatch(self, event: SwitchEvent) -> None:
        with self._listener_lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(event)
            except Exception:
                log.warning("a switch listener raised on %s -> %s", event.from_slot, event.to_slot, exc_info=True)

    # -- pinning -------------------------------------------------------------------------------

    def pin(self, role: str, name: str, strict: bool = False) -> str:
        """Prefer one slot of a role, by full name or by a prefix that names exactly one; strict means only that one.

        Returns the slot's full name. Raises ConfigError, listing the slots, when `name` picks none or several.
        """
        role = self._resolve(role)
        return self._pin(role, self.chain(role), name, strict, f"cannot pin {name!r}")

    def unpin(self, role: str) -> None:
        role = self._resolve(role)
        with self._lock:
            self._pins.pop(role, None)
            facade = self._facades.get(role)
        if facade is not None:
            facade.chain.unpin()

    def pinned(self, role: str) -> tuple[str, bool] | None:
        """(slot name, strict) while a role is pinned, None otherwise.

        Builds the role's chain if it is not built yet, so that a pin from the environment is already applied; a
        role with no usable slot has no pin. A pin that cannot be honoured raises the ConfigError it always does.
        """
        role = self._resolve(role)
        with contextlib.suppress(NoUsableProvider):
            self._facade(role)
        return self._pins.get(role)

    def _pin(self, role: str, chain: Chain, name: str, strict: bool, what: str) -> str:
        name = (name or "").strip()
        try:
            if not name:
                raise ValueError(name)
            matched = chain.pin(name, strict=strict)
        except ValueError:
            skipped = [f"{i.name} ({i.reason})" for i in self.slots(role) if i.state == "skipped"]
            raise ConfigError(
                f"{what}: it does not pick exactly one usable {role} provider (usable: {', '.join(chain.names)})"
                + (f"; skipped: {'; '.join(skipped)}" if skipped else "")
            ) from None
        with self._lock:
            self._pins[role] = (matched, strict)
        return matched

    def _pin_from_environment(self, role: str, chain: Chain) -> None:
        variable = pin_variable(role)
        raw = (self._env.get(variable) or "").strip()
        if not raw:
            return
        strict = raw.endswith("!")
        name = raw[:-1].strip() if strict else raw
        self._pin(role, chain, name, strict, f"{variable}={raw!r}")


# ---------------------------------------------------------------------------------------------
# Finding the file
# ---------------------------------------------------------------------------------------------


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigError(f"cannot read {path}: {e.strerror or type(e).__name__}") from None
    except UnicodeDecodeError:
        raise ConfigError(f"{path} is not UTF-8 text") from None


def load_config(
    path: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    *,
    cwd: Path | None = None,
    home: Path | None = None,
) -> GlideConfig:
    """Find and load glide.toml: `path`, else $GLIDE_CONFIG, else ./glide.toml, else ~/.config/glide/glide.toml, else the built-in chains.

    A `path` or $GLIDE_CONFIG that names a file that is not there is an error, never a fallback: the user asked
    for that file. `env` is the only source of keys, GLIDE_CONFIG and the GLIDE_PIN_* variables, and is the
    process environment when left out. `home` defaults to env["HOME"], and `cwd` to the working directory;
    both exist so a test can never read the real ones.
    """
    environment = os.environ if env is None else env
    chosen: Path | None = None
    user_file: Path | None = None
    in_cwd = False
    if path is not None:
        chosen = Path(path).expanduser()
        if not chosen.is_file():
            raise ConfigError(f"the config file {chosen} does not exist")
    elif (named := (environment.get("GLIDE_CONFIG") or "").strip()) != "":
        chosen = Path(named).expanduser()
        if not chosen.is_file():
            raise ConfigError(f"GLIDE_CONFIG names {chosen}, which does not exist")
    else:
        here = (cwd or Path.cwd()) / CONFIG_NAME
        base = home if home is not None else (Path(h) if (h := environment.get("HOME")) else None)
        if base is None and env is None:
            base = Path.home()
        user_file = base / ".config" / "glide" / CONFIG_NAME if base else None
        for candidate in (here, user_file):
            if candidate is not None and candidate.is_file():
                chosen, in_cwd = candidate, candidate is here
                break
    if chosen is None:
        return GlideConfig.from_toml("", env=environment, source="built-in defaults")
    trusted = _own_hosts(user_file) if in_cwd else None
    return GlideConfig.from_toml(_read(chosen), env=environment, source=str(chosen), trusted_hosts=trusted)


def _own_hosts(user_file: Path | None) -> frozenset[str]:
    """The hosts the user's own glide.toml names in its `[providers.*]` tables (none if it is missing or unreadable)."""
    if user_file is None or not user_file.is_file():
        return frozenset()
    try:
        providers = tomllib.loads(user_file.read_text(encoding="utf-8")).get("providers", {})
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return frozenset()
    entries = providers.values() if isinstance(providers, dict) else ()
    return frozenset(_host(e["base_url"]) for e in entries if isinstance(e, dict) and isinstance(e.get("base_url"), str))
