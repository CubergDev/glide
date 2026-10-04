"""What the router computes from the utterance and its context, with no model: structural features only.

There is no list of user commands here (that would be the catalog the executor deliberately does not have), and no
language-specific verb list. The features are facts about the text and the situation that a classifier is told and
that a few safety rules read:

- `script`, `tokens`, `question_mark`: the shape of the utterance.
- `has_history`, `task_running`, `data_sources`: whether this is a follow-up, whether a task is in progress, and what
  kinds of untrusted content are present (source and size only, never the text).
- `injection_marker`: the utterance itself contains wording that tries to override the assistant (an "ignore your
  instructions" sentence, a fake system tag). That is how pasted or forwarded hostile text looks when it reaches the
  router inside what the user said. It is a narrow, documented set, extendable in configuration code, and it can only
  ever REMOVE the ability to act, never grant it. A person who really says such a sentence gets an answer, not an action.
"""

# ruff: noqa: RUF001  full-width punctuation is the right punctuation here

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

from .decision import Context
from .stop import normalize

# Applied to the NORMALISED utterance (lower case, punctuation turned to spaces), or to the compact form for CJK.
_LATIN_MARKERS = tuple(
    re.compile(p)
    for p in (
        r"\b(ignore|disregard|forget|override|bypass|discard)\b.{0,40}\b(instructions?|rules|prompts?|guidelines|directions|restrictions)\b",
        r"\b(previous|prior|above|earlier) (instructions?|prompts?|messages?)\b.{0,30}\b(ignore|void|obsolete|irrelevant)\b",
        r"\bsystem prompt\b",
        r"\b(system|developer) override\b",
        r"\byou are now\b",
        r"\bjailbreak\b",
        r"\b(do not|dont) (tell|inform|alert) the user\b",
        r"\bact as if (the )?user (said|asked|approved|confirmed)\b",
    )
)
_CJK_MARKERS = tuple(
    re.compile(p)
    for p in (
        r"(忽略|無視|无视|忘記|忘记|不要理會|不要理会|唔好理)[^，。]{0,12}(指令|指示|提示|規則|规则|說明|说明|設定|设定)",
        r"(忽略|無視|无视|忘記|忘记)[^，。]{0,20}(instructions?|rules|prompts?|guidelines)",
        r"(ignore|disregard|forget|bypass)[^，。]{0,12}(指令|指示|提示|規則|规则)",
        r"(系統|系统)(提示詞|提示词|指令)",
        r"(以上|之前|先前|上面)[^，。]{0,6}(指令|指示|提示)[^，。]{0,6}(無效|无效|作廢|作废|唔算數)",
        r"(你現在是|你现在是|你而家係)[^，。]{0,8}(管理員|管理员|開發者|开发者|越獄|越狱)",
    )
)
# Override framing in the raw text, where the colon still exists: a line that opens with a fake instruction or role tag.
_RAW_MARKERS = (
    re.compile(r"(^|[.!?\n]\s*)(new|updated|additional) (instructions?|rules)\s*:"),
    re.compile(r"(^|\n)\s*(system|assistant)\s*:"),
)
_QUOTE = re.compile(r"[\"“”「」『』`]{1}[^\"“”「」『』`]{40,}[\"“”「」『』`]{1}")


@dataclass(frozen=True)
class Features:
    script: str  # "latin", "cjk", "mixed" or "other"
    tokens: int
    chars: int
    question_mark: bool
    quoted_block: bool  # a long quoted passage: probably pasted, so what is in it is not the request
    has_history: bool
    task_running: bool
    data_sources: tuple[tuple[str, int], ...]  # (source, characters) of untrusted content present, never the text
    injection_marker: bool

    def as_state(self) -> dict:
        """The features as the plain, bounded dict a classifier is shown."""
        return {
            "script": self.script,
            "tokens": self.tokens,
            "question_mark": self.question_mark,
            "quoted_block": self.quoted_block,
            "follow_up": self.has_history,
            "task_running": self.task_running,
            "untrusted_data_present": [{"source": s, "chars": n} for s, n in self.data_sources],
        }


def _script(text: str) -> str:
    cjk = latin = 0
    for ch in text:
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        if name.startswith(("CJK", "HIRAGANA", "KATAKANA", "HANGUL", "BOPOMOFO")):
            cjk += 1
        elif name.startswith("LATIN"):
            latin += 1
    if cjk and latin:
        return "mixed"
    return "cjk" if cjk else "latin" if latin else "other"


def has_injection_marker(text: str, extra: Iterable[str] = ()) -> bool:
    """Whether `text` contains wording that tries to override the assistant. `extra` are further phrases (substrings)."""
    norm = normalize(text)
    if any(p.search(norm) for p in _LATIN_MARKERS):
        return True
    compact = "".join(norm.split())
    folded = text.casefold()
    if any(p.search(folded) for p in _RAW_MARKERS):
        return True
    if any(p.search(folded) or p.search(compact) for p in _CJK_MARKERS):
        return True
    extra_norm = [e for e in map(normalize, extra) if e]
    return any(e in norm for e in extra_norm)


def features(text: str, context: Context, *, extra_markers: Iterable[str] = ()) -> Features:
    return Features(
        script=_script(text),
        tokens=len(text.split()),
        chars=len(text),
        question_mark=text.rstrip().endswith(("?", "？")),
        quoted_block=bool(_QUOTE.search(text)),
        has_history=bool(context.history),
        task_running=bool(context.running_task),
        data_sources=tuple((span.source, len(span.text)) for span in context.data),
        injection_marker=has_injection_marker(text, extra_markers),
    )
