"""Literal data extraction only; natural-language intent/quantities belong to the model."""

import json
import re
from dataclasses import dataclass, field

from .contracts import MAX_REPETITIONS, safe_url

QUOTES = re.compile(r'"(?:[^"\\\n]|\\.)*"|“[^”\n]*”')
URI = re.compile(r'[a-zA-Z][a-zA-Z0-9+.-]*://[^\s"“”<>]+')
HOST = re.compile(r'(?<![\w/@:])[\w.-]+\.[a-zA-Z]{2,}(?::\d+)?(?:/[^\s"“”<>]*)?')
MODIFIER_NAMES = {
    "cmd": "command",
    "command": "command",
    "ctrl": "control",
    "control": "control",
    "alt": "alt",
    "option": "alt",
    "opt": "alt",
    "shift": "shift",
}
CHORD = re.compile(r"(?<!\w)((?:(?:command|cmd|control|ctrl|option|opt|alt|shift)\s*[+-]\s*)+)([a-z][a-z0-9]*)\b", re.I)


@dataclass
class Literals:
    urls: list[str] = field(default_factory=list)
    text: list[str] = field(default_factory=list)
    counts: list[str] = field(default_factory=list)
    needs_plan: bool = False


class Request(str):
    """The request as a model is shown it, with the questions asked and the replies given. `stated` is what the user
    wrote: the request and the replies. Only that part can name an address, a quoted text or a count."""

    def __new__(cls, shown, *, stated):
        text = super().__new__(cls, shown)
        text.stated = stated
        return text


def stated(goal):
    return getattr(goal, "stated", goal)


def extract(goal):
    goal = stated(goal)
    data, spans = Literals(), []
    for match in QUOTES.finditer(goal):
        spans.append(match.span())
        raw = match.group()
        try:
            value = raw[1:-1] if raw.startswith("“") else json.loads(raw)
        except ValueError:
            data.needs_plan = True
            continue
        if len(value) > 8192:
            data.needs_plan = True
        else:
            data.text.append(value)
    uri_spans = []
    for match in URI.finditer(goal):
        uri_spans.append(match.span())
        spans.append(match.span())
        value = match.group().rstrip(".,!?")
        if safe_url(value):
            if value not in data.urls:
                data.urls.append(value)
        else:
            data.needs_plan = True
    for match in HOST.finditer(goal):
        if any(a <= match.start() < b for a, b in uri_spans):
            continue
        spans.append(match.span())
        value = "https://" + match.group().rstrip(".,!?")
        if safe_url(value) and value not in data.urls:
            data.urls.append(value)
    remaining = list(goal)
    for start, end in spans:
        remaining[start:end] = " " * (end - start)
    data.counts = re.findall(r"(?<![\w.])[+-]?\d+(?:\.\d+)?(?![\w.])", "".join(remaining))
    if re.search(r"(?<!\w)\d+(?:\.\d+)?\s*(?:px|pixels?|pt|points?|percent|%)(?!\w)", "".join(remaining), re.I):
        data.needs_plan = True  # A distance is not a repeated-operation quantity.
    if (
        len(data.urls) > 1
        or len(data.text) > 1
        or len(data.counts) > 1
        or any(not n.isdecimal() or not 1 <= int(n) <= MAX_REPETITIONS for n in data.counts)
    ):
        data.needs_plan = True
    return data


def chords(text):
    """Normalize literal keyboard notation; aliases describe input symbols, not app commands."""
    result = set()
    for prefix, key in CHORD.findall(text):
        names = re.findall(r"[a-z]+", prefix.casefold())
        result.add((key.casefold(), tuple(sorted({MODIFIER_NAMES[n] for n in names}))))
    for symbolic in re.findall(r"([⌘⌃⌥⇧]+)([a-zA-Z][a-zA-Z0-9]*)", text):
        result.add(
            (
                symbolic[1].casefold(),
                tuple(sorted({{"⌘": "command", "⌃": "control", "⌥": "alt", "⇧": "shift"}[s] for s in symbolic[0]})),
            )
        )
    return result
