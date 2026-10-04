"""The settings the app shows and changes: what it reads from `glide.toml`, what it may change, and the revision between the two.

What the app sees is built from the loaded configuration on every read: every role, slot, provider and model name is whatever
`glide.toml` says (there is no list here), a slot reports the NAME of its key variable and whether that variable is set, and
nothing else about the key. No setting and no message can carry a key.

What the app may change is a closed set (`voice.hands_free`, `voice.headset`, `voice.silence_ms`, `voice.language`,
`privacy.record_content`, `computer.act_enabled`, `computer.engine`, `roles.pin`). Changes are held in this process only: `glide.toml` is never
written, so a restart returns to the file, and recording content, which is opt-in, always starts off unless the command line
asked for it. Every change is checked before any is applied, and a change that cannot be applied leaves the others unapplied:
`ok: false` means nothing happened.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from .. import features
from ..computer import config as computer_config
from ..providers.config import EXTRA_LLM_ROLES, ROLES, ConfigError
from ..speech.settings import MAX_SILENCE_MS, MIN_SILENCE_MS, SpeechSettings
from . import wire

_LANGUAGE = re.compile(r"[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8}){0,3}")


class VoiceUnavailable(Exception):
    """Hands-free voice could not be started or changed. The message is safe to show."""


@dataclass(frozen=True)
class SettingsState:
    """What the app may change, as it is now."""

    hands_free: bool = False
    headset: bool = False
    silence_ms: int = 600
    language: str | None = None
    record_content: bool = False
    act_enabled: bool = False
    engine: str = computer_config.DEFAULT_ENGINE

    @classmethod
    def from_config(cls, config: object, *, record_content: bool = False) -> SettingsState:
        voice = getattr(config, "voice", None)
        voice = voice if isinstance(voice, SpeechSettings) else SpeechSettings()
        try:
            engine = features.engine_for(config)  # glide.toml and GLIDE_ENGINE: the app shows what a task would use
        except ValueError:  # a bad setting is reported by `glide doctor` and by the task; the app starts on the default
            engine = computer_config.DEFAULT_ENGINE
        return cls(
            headset=voice.headset,
            silence_ms=voice.silence_ms,
            language=voice.language,
            record_content=record_content,
            engine=engine,
        )


class SettingsPanel:
    """The settings payload and `settings_set`, over a configuration and the things a change has to reach.

    `apply_voice(state)` makes the voice loop match `state` (it may raise `VoiceUnavailable`, and then nothing is applied);
    `set_record_content(bool)` tells the rest of the core.
    """

    def __init__(
        self,
        config: Any,
        state: SettingsState,
        *,
        apply_voice: Callable[[SettingsState], None],
        set_record_content: Callable[[bool], None],
    ) -> None:
        self._config = config
        self._state = state
        self._apply_voice = apply_voice
        self._set_record_content = set_record_content
        self._revision = 1
        self._lock = threading.Lock()

    @property
    def state(self) -> SettingsState:
        return self._state

    @property
    def revision(self) -> int:
        return self._revision

    # -- reading ---------------------------------------------------------------------------------

    def snapshot(self) -> tuple[int, dict[str, Any]]:
        with self._lock:
            state, revision = self._state, self._revision
        return revision, {
            "voice": {
                "hands_free": state.hands_free,
                "headset": state.headset,
                "silence_ms": state.silence_ms,
                "language": state.language or "",
                "silence_ms_range": [MIN_SILENCE_MS, MAX_SILENCE_MS],
            },
            "privacy": {"record_content": state.record_content},
            "computer": {"act_enabled": state.act_enabled, "engine": state.engine, "engines": list(computer_config.ENGINES)},
            "roles": [self._role(role) for role in self._roles()],
        }

    def _roles(self) -> list[str]:
        known = getattr(self._config, "roles", {})
        return [*ROLES, *(role for role in EXTRA_LLM_ROLES if role in known)]

    def _role(self, role: str) -> dict[str, Any]:
        config = self._config
        try:
            infos = list(config.slots(role))
        except ConfigError:
            infos = []
        try:
            rows = {row["name"]: row for row in config.chain(role).status()}
        except ConfigError:
            rows = {}
        pinned = None
        try:
            pin = config.pinned(role)
            pinned = pin[0] if pin else None
        except ConfigError:
            pass
        chain = []
        for info in infos:
            resting = (rows.get(info.name) or {}).get("resting_s", 0) or 0
            if info.state == "skipped":
                status = "skipped"
            elif resting > 0:
                status = "resting"
            elif info.state == "ready":
                status = "ready"
            else:
                status = "unknown"
            slot: dict[str, Any] = {
                "name": info.name,
                "provider": info.provider or info.name,
                "key_present": bool(info.env_var) and info.env_var not in info.missing,
                "status": status,
            }
            if info.model:
                slot["model"] = info.model
            if info.env_var:
                slot["key_env"] = info.env_var  # the NAME of the variable, never its value
            chain.append(slot)
        return {"role": role, "pinned": pinned, "chain": chain}

    # -- changing --------------------------------------------------------------------------------

    def apply(self, base_revision: int, changes: list[dict[str, Any]]) -> tuple[bool, int, list[tuple[str, str]]]:
        """Apply `changes` if `base_revision` is current and every one is acceptable. Returns (ok, revision, errors)."""
        with self._lock:
            if base_revision != self._revision:
                return False, self._revision, [("revision", "the settings changed since you read them; read them again")]
            new, pins, errors = self._validate(self._state, changes)
            if errors:
                return False, self._revision, errors
            old = self._state
            voice_changed = (new.hands_free, new.headset, new.silence_ms, new.language, new.act_enabled) != (
                old.hands_free,
                old.headset,
                old.silence_ms,
                old.language,
                old.act_enabled,
            )
            if voice_changed and (new.hands_free or old.hands_free):
                try:
                    self._apply_voice(new)
                except VoiceUnavailable as exc:
                    return False, self._revision, [("voice.hands_free", str(exc))]
            if new.record_content != old.record_content:
                self._set_record_content(new.record_content)
            for role, slot in pins:
                if slot is None:
                    self._config.unpin(role)
                else:
                    self._config.pin(role, slot)
            if new.engine != old.engine:
                self._config.engine_choice = new.engine  # where every task reads it (`features.engine_for`)
            self._state = new
            self._revision += 1
            return True, self._revision, []

    def _validate(
        self, state: SettingsState, changes: list[dict[str, Any]]
    ) -> tuple[SettingsState, list[tuple[str, str | None]], list[tuple[str, str]]]:
        errors: list[tuple[str, str]] = []
        pins: list[tuple[str, str | None]] = []
        values: dict[str, Any] = {}
        for change in changes:
            key, value = change["key"], change["value"]
            if key not in wire.SETTING_KEYS:
                errors.append((key, "this is not a setting"))
            elif key in ("voice.hands_free", "voice.headset", "privacy.record_content", "computer.act_enabled"):
                if isinstance(value, bool):
                    values[key.split(".")[1]] = value
                else:
                    errors.append((key, "must be true or false"))
            elif key == "voice.silence_ms":
                if isinstance(value, int) and not isinstance(value, bool) and MIN_SILENCE_MS <= value <= MAX_SILENCE_MS:
                    values["silence_ms"] = value
                else:
                    errors.append((key, f"must be a whole number from {MIN_SILENCE_MS} to {MAX_SILENCE_MS}"))
            elif key == "computer.engine":
                if isinstance(value, str) and value in computer_config.ENGINES:
                    values["engine"] = value
                else:
                    errors.append((key, f"must be one of: {', '.join(computer_config.ENGINES)}"))
            elif key == "voice.language":
                if isinstance(value, str) and (value == "" or _LANGUAGE.fullmatch(value)):
                    values["language"] = value or None
                else:
                    errors.append((key, "must be a language code such as en, or empty for automatic"))
            else:
                problem = self._check_pin(value)
                if isinstance(problem, str):
                    errors.append((key, problem))
                else:
                    pins.append(problem)
        return replace(state, **values), pins, errors

    def _check_pin(self, value: object) -> tuple[str, str | None] | str:
        if not isinstance(value, Mapping) or not isinstance(value.get("role"), str):
            return "must name a role and a slot"
        role, slot = value["role"], value.get("slot")
        if role not in self._roles():
            return "this is not a role of the configuration"
        if slot is None:
            return role, None
        if not isinstance(slot, str):
            return "the slot must be a name, or null to clear the pin"
        try:
            names = list(self._config.chain(role).names)
        except ConfigError as exc:
            return self._scrub(str(exc))
        if slot not in names:
            return "this is not a usable slot of that role"
        return role, slot

    def _scrub(self, text: str) -> str:
        scrub = getattr(self._config, "scrub", None)
        return scrub(text) if callable(scrub) else text
