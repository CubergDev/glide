"""Word lists and the text folding that makes them hard to dodge.

The lists are data (`default.toml`, one `[lang.<code>]` table per language), never code. Every language applies at
once, because the language of a page is not known. `normalize` is the only place text is folded, for labels, field
hints and task text alike.
"""

# ruff: noqa: RUF001  the look-alike letters are the data of this table

from __future__ import annotations

import re
import tomllib
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import resources

DEFAULT_TOML = resources.files(__package__).joinpath("default.toml").read_text(encoding="utf-8")

# Letters that look like a Latin letter in another script, after case folding. Not a spelling list: a way to read a
# label the way a person does.
_LOOKALIKES = str.maketrans(
    {
        "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
        "і": "i", "ѕ": "s", "ј": "j", "һ": "h", "ԁ": "d", "ԛ": "q", "ѵ": "v",
        "α": "a", "ο": "o", "ρ": "p", "ν": "v", "ι": "i", "κ": "k", "τ": "t",
        "υ": "u", "ε": "e", "ı": "i", "ɡ": "g",
    }
)  # fmt: skip
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])")
_SPREAD = 3  # this many single letters in a row ("p a s s") are read as one word as well


def normalize(text: str) -> tuple[str, ...]:
    """The words of `text`: case, accents, invisible characters, look-alike letters, camelCase and punctuation folded."""
    text = unicodedata.normalize("NFKD", text or "")
    # Format characters (zero-width, soft hyphen) hide inside words; the combining accents of Latin letters are dropped.
    text = "".join(c for c in text if unicodedata.category(c) != "Cf" and not 0x300 <= ord(c) <= 0x36F)
    text = _CAMEL.sub(" ", text).casefold().translate(_LOOKALIKES)
    words: list[str] = []
    current: list[str] = []
    for ch in text:
        if ch.isalnum() or (current and unicodedata.category(ch).startswith("M")):
            current.append(ch)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return tuple(words)


def spread_joined(words: tuple[str, ...]) -> tuple[str, ...]:
    """`words` with every run of single letters joined into one word, or `words` itself when there is none."""
    out: list[str] = []
    run: list[str] = []
    for word in (*words, ""):
        if len(word) == 1 and word.isalpha():
            run.append(word)
            continue
        out.extend(["".join(run)] if len(run) >= _SPREAD else run)
        run = []
        if word:
            out.append(word)
    return tuple(out)


def _phrases(entries: Iterable[str]) -> tuple[tuple[str, ...], ...]:
    return tuple(p for p in (normalize(e) for e in entries) if p)


def _fold(name: str) -> str:
    return "".join(normalize(name))


@dataclass(frozen=True)
class Lexicon:
    classes: Mapping[str, tuple[tuple[str, ...], ...]]
    plural: frozenset[str]
    input_roles: frozenset[str]
    secret_roles: frozenset[str]
    known_roles: frozenset[str]
    secret_types: frozenset[str]
    submit_keys: frozenset[str]
    neutral_keys: frozenset[str]
    kinds: Mapping[str, frozenset[str]]

    def find(self, name: str, texts: Iterable[tuple[str, ...]]) -> str:
        """The first phrase of class `name` that is a run of whole words in any of `texts`, or ''."""
        plural = name in self.plural
        for words in texts:
            for variant in {words, spread_joined(words)}:
                for phrase in self.classes.get(name, ()):
                    if _contains(variant, phrase, plural):
                        return " ".join(phrase)
        return ""

    def role(self, role: str) -> str:
        """A role as one folded word: AXTextField, textfield and text_field are the same."""
        folded = _fold(role)
        bare = folded[2:] if folded.startswith("ax") else folded
        return bare if bare in self.known_roles | self.input_roles else folded

    def kind(self, kind: str) -> str:
        folded = _fold(kind)
        return next((group for group, kinds in self.kinds.items() if folded in kinds), "")


def _contains(words: tuple[str, ...], phrase: tuple[str, ...], plural: bool) -> bool:
    def same(word: str, want: str) -> bool:
        return word == want or (plural and word in (want + "s", want + "es"))

    n = len(phrase)
    return any(all(same(words[i + j], phrase[j]) for j in range(n)) for i in range(len(words) - n + 1))


def load_lexicon(extra: Mapping | None = None) -> Lexicon:
    """The default lists, with `extra` (the same shape as `default.toml`) added to them. Lists extend, never replace."""
    data = tomllib.loads(DEFAULT_TOML)
    classes: dict[str, list[str]] = {}
    for tables in (data.get("lang", {}), (extra or {}).get("lang", {})):
        for table in tables.values():
            for name, words in table.items():
                classes.setdefault(name, []).extend(words)

    def listed(section: str, key: str) -> frozenset[str]:
        words = [*data.get(section, {}).get(key, ()), *((extra or {}).get(section, {}).get(key, ()))]
        return frozenset(_fold(w) for w in words)

    kinds = {k: listed("kinds", k) for k in data["kinds"]}
    return Lexicon(
        classes={name: _phrases(words) for name, words in classes.items()},
        plural=frozenset(data["matching"]["plural"]) | frozenset((extra or {}).get("matching", {}).get("plural", ())),
        input_roles=listed("roles", "input"),
        secret_roles=listed("roles", "secret"),
        known_roles=listed("roles", "known"),
        secret_types=listed("field_types", "secret"),
        submit_keys=listed("keys", "submit"),
        neutral_keys=listed("keys", "neutral"),
        kinds=kinds,
    )
