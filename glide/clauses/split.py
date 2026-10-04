"""Split one utterance into ordered steps, each saying which earlier steps it waits for. Rule-based and deterministic.

The rules, in one list:
- Quoted text (an unclosed double quote runs to the end) is masked first and put back at the end, so nothing inside quotes is ever split or read as a marker.
- "then", "after that", "afterwards", "once done" (and the mishearing "than" after "and" or a comma) start a step that
  waits for everything before it. A sentence end and a comma before another action do the same.
- "and" before another action starts a step too, but only when the word after it is a known action and what came
  before has more than one word. "Tom and Jerry", "notes and calculator", "copy and paste" and "1,000" stay whole.
  Such a step may run together with the one before it, unless its action needs the earlier result (write, close ...)
  or it says "it"/"that"/"them", which point at the step before.
- English only. The word lists are plain data below; a language adds its own.

`depends_on` only ever holds earlier indexes. Step text is the speaker's own words, trimmed, never rewritten.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Words that start an action. A bare "and" splits only in front of one of these.
VERBS = frozenset(
    {"open", "close", "quit", "launch", "start", "search", "find", "look", "play", "pause", "stop", "write", "type", "enter"}
    | {"paste", "copy", "save", "click", "press", "tap", "scroll", "select", "delete", "remove", "go", "navigate", "visit"}
    | {"send", "create", "make", "set", "turn", "mute", "unmute", "show", "tell", "read", "take", "add", "check", "run"}
    | {"switch", "call", "email", "message", "ask", "bring", "put", "move", "rename", "download", "upload", "print"}
)
# Actions that work on what an earlier step produced, so an "and" before them is an order, not a parallel.
CONTEXT_VERBS = frozenset(
    {"write", "type", "enter", "paste", "save", "close", "quit", "click", "press", "tap", "scroll", "select", "delete"}
    | {"remove", "rename", "print", "copy", "pause", "stop"}
)
POINTERS = frozenset({"it", "that", "this", "them", "those", "these"})
JOINS = frozenset({"and", "an"})
# Words that may stand before an action without being a step ("hey glide, open notes").
FILLER = frozenset({"hey", "hi", "ok", "okay", "glide", "please", "yes", "so", "well", "um", "uh", "and", "also"})
ABBREVIATIONS = frozenset({"dr.", "mr.", "mrs.", "ms.", "vs.", "st.", "etc.", "e.g.", "i.e."})
# Marker phrases (lower case, apostrophes straight). Each one ends the clause before it and waits for it.
MARKERS = (
    ("once", "that's", "done"),
    ("once", "it's", "done"),
    ("once", "thats", "done"),
    ("when", "that's", "done"),
    ("after", "that"),
    ("after", "wards"),
    ("once", "finished"),
    ("once", "done"),
    ("when", "done"),
    ("afterthat",),
    ("afterwards",),
    ("then",),
)

# A double quote that never closes runs to the end: unsure means do not split.
_QUOTED = re.compile(r"\"[^\"]*(?:\"|$)|\u201c[^\u201d]*(?:\u201d|$)|\u2018[^\u2019]*\u2019|(?<!\w)'[^']*'(?!\w)")
_TOKEN = re.compile(r"\x00\d+\x00[^\s,;]*|\d+(?:[,.]\d+)+|[^\s,;]+|[,;]")
_HOLE = re.compile(r"\x00(\d+)\x00")


@dataclass(frozen=True)
class Step:
    index: int
    text: str
    depends_on: tuple[int, ...]  # earlier steps that must finish first; empty = may start at once
    refers_back: bool  # says "it"/"that"/...: the target is whatever the step before worked on


@dataclass(frozen=True)
class Plan:
    steps: tuple[Step, ...]

    def parallel_groups(self) -> list[tuple[int, ...]]:
        """Runs of neighbouring steps that wait for the same things, so may run together."""
        groups: list[list[int]] = []
        for step in self.steps:
            if groups and self.steps[groups[-1][0]].depends_on == step.depends_on:
                groups[-1].append(step.index)
            else:
                groups.append([step.index])
        return [tuple(g) for g in groups]


@dataclass
class _Tok:
    start: int
    end: int
    word: str  # lower case, edge punctuation gone

    def __post_init__(self) -> None:
        self.word = self.word.replace("\u2019", "'").strip('.,;:!?"').casefold()


def split_clauses(text: str) -> Plan:
    """The plan for one utterance: an empty plan for blank text, one step when nothing splits."""
    quotes: list[str] = []

    def hide(m: re.Match[str]) -> str:
        quotes.append(m.group(0))
        return f"\x00{len(quotes) - 1}\x00"

    masked = _QUOTED.sub(hide, text.replace("\x00", " "))
    toks = [_Tok(m.start(), m.end(), m.group(0)) for m in _TOKEN.finditer(masked)]
    raw = [masked[t.start : t.end] for t in toks]

    clauses: list[tuple[list[int], str | None]] = []  # token indexes, how it joins the one before
    cur: list[int] = []
    pending: str | None = None

    def close() -> None:
        nonlocal cur, pending
        while cur and toks[cur[-1]].word in JOINS:
            cur.pop()
        if cur:
            clauses.append((cur, pending if clauses else None))
            pending = None
        cur = []

    def word(k: int) -> str:
        return toks[k].word if k < len(toks) else ""

    def marker_at(k: int) -> int:
        for m in MARKERS:
            if tuple(word(k + d) for d in range(len(m))) == m:
                return len(m)
        # "than" is a mishearing of "then" only after a join and in front of an action
        if word(k) == "than" and word(k + 1) in VERBS and k > 0 and (word(k - 1) in JOINS or raw[k - 1] == ","):
            return 1
        return 0

    def splittable() -> bool:
        return len(cur) >= 2 and any(toks[c].word not in FILLER for c in cur)

    i = 0
    while i < len(toks):
        w = toks[i].word
        if n := marker_at(i):
            close()
            pending = "seq"
            i += n
        elif raw[i] in {",", ";"}:
            j = i + 1 + (word(i + 1) in {"and", "also"})
            if marker_at(j) or marker_at(i + 1):
                i += 1  # the marker after it does the splitting
            elif splittable() and word(j) in VERBS:
                close()
                pending = "seq"
                i = j
            else:
                i += 1
        elif w in JOINS and (marker_at(i + 1) or (word(i + 1) == "also" and marker_at(i + 2))):
            i += 1
        elif w == "and" and splittable() and word(i + 1) in VERBS:
            close()
            pending = "seq" if word(i + 1) in CONTEXT_VERBS else "par"
            i += 1
        else:
            cur.append(i)
            i += 1
            if raw[cur[-1]][-1:] in {".", "!", "?"} and raw[cur[-1]].casefold() not in ABBREVIATIONS and i < len(toks):
                close()
                pending = "seq"
    close()

    steps: list[Step] = []
    group: list[int] = []
    group_deps: tuple[int, ...] = ()
    for idx, (members, kind) in enumerate(clauses):
        start, end = toks[members[0]].start, toks[members[-1]].end
        body = _HOLE.sub(lambda m: quotes[int(m.group(1))], masked[start:end]).strip(" .,;!?")
        points = any(toks[c].word in POINTERS for c in members)
        if kind == "par" and not points:
            group.append(idx)
            deps = group_deps
        else:
            group_deps = tuple(group)
            group = [idx]
            deps = group_deps
        if points and idx:
            deps = tuple(sorted({*deps, idx - 1}))
        steps.append(Step(idx, body, deps, points and idx > 0))
    return Plan(tuple(steps))
