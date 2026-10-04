"""One proposed UI action in, one verdict out. Pure: no clock, no screen, no I/O, no state.

The verdicts, mildest first: `allow`; `ask` (needs the user's explicit approval, with a reason class); `handoff` (a
personal detail the task never supplied: the user types it, the assistant never invents it); `refuse` (a secret
field: nothing is typed or submitted there, whatever the task says).

Rules, in the order they run for a click: a secret input is refused; a control whose words name a risky effect
(whole words only, see `lexicon.py`) is asked about with that effect as the reason; a control with no label, or a
role the tables do not know, is asked about rather than allowed. Pressing Return (or typing a line break) is a
click on whatever has focus, and in a message-like field it also needs send intent. Words are matched on the label,
the hint (tooltip, placeholder), the role and, for controls that are not inputs, the value, each alone and all
together, so a phrase split across them still matches. The words of an input's own value are the user's text and are
never scanned. There is no negation: "Do not delete" is asked about. Asking too often is the safe error.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
from enum import StrEnum

from .lexicon import Lexicon, load_lexicon, normalize

_LINE_BREAKS = frozenset("\r\n\u2028\u2029\x0b\x0c\x85")
# Most severe effect first: this is the reason given when a label names more than one.
_RISKS = ("destructive", "spending", "communication", "overwrite", "consent")


class Decision(StrEnum):
    ALLOW = "allow"
    ASK = "ask"
    HANDOFF = "handoff"
    REFUSE = "refuse"


class Reason(StrEnum):
    SPENDING = "spending"
    COMMUNICATION = "communication"
    DESTRUCTIVE = "destructive"
    OVERWRITE = "overwrite"
    CONSENT = "consent"
    SECRET_FIELD = "secret_field"
    PERSONAL = "personal"
    UNLABELED = "unlabeled"
    UNKNOWN = "unknown"  # a role or an action kind the tables do not list


_SEVERITY = {Decision.ALLOW: 0, Decision.ASK: 1, Decision.HANDOFF: 2, Decision.REFUSE: 3}


@dataclass(frozen=True)
class Action:
    """What is about to happen, described by the page. All text is untrusted data and none of it is ever echoed."""

    kind: str  # click, type, key, scroll ...
    label: str = ""  # the control's visible name
    role: str = ""  # AXButton, textbox, Edit ...
    hint: str = ""  # tooltip, description, placeholder
    value: str = ""  # the control's current value
    field_name: str = ""  # name/autocomplete attribute of an input
    field_type: str = ""  # input type
    text: str = ""  # what would be typed
    key: str = ""  # the key pressed, with modifiers ("cmd+Enter")
    task_text: str = ""  # the user's own request
    supplied: tuple[str, ...] = ()  # values the user configured (their email), which count as supplied
    secret: bool = False  # the observation already marked this input secret
    send_intent: bool = False  # the user asked for something to be sent


@dataclass(frozen=True)
class Verdict:
    decision: Decision
    reason: Reason | None = None
    term: str = ""  # the list phrase that decided it; never page text

    def as_dict(self) -> dict[str, str | None]:
        return {"decision": self.decision.value, "reason": self.reason.value if self.reason else None, "term": self.term}


ALLOW = Verdict(Decision.ALLOW)
_DEFAULT: Lexicon | None = None


def decide(action: Action, lexicon: Lexicon | None = None) -> Verdict:
    global _DEFAULT
    if lexicon is None:
        _DEFAULT = _DEFAULT or load_lexicon()
        lexicon = _DEFAULT
    kind = lexicon.kind(action.kind)
    if kind == "passive":
        return ALLOW
    if kind == "click":
        return _control(action, lexicon, as_input=False)
    if kind == "type":
        return _type(action, lexicon)
    if kind == "key":
        return _key(action, lexicon)
    return Verdict(Decision.ASK, Reason.UNKNOWN)


def _worst(*verdicts: Verdict) -> Verdict:
    return max(verdicts, key=lambda v: _SEVERITY[v.decision])


def _is_input(a: Action, lex: Lexicon, role: str) -> bool:
    return role in lex.input_roles or bool(a.field_type.strip() or a.field_name.strip())


def _field_words(a: Action) -> list[tuple[str, ...]]:
    return [normalize(t) for t in (a.label, a.hint, a.field_name, a.field_type)]


def _secret(a: Action, lex: Lexicon, role: str) -> Verdict | None:
    if a.secret or role in lex.secret_roles:
        return Verdict(Decision.REFUSE, Reason.SECRET_FIELD)
    for word in normalize(a.field_type):
        if word in lex.secret_types:
            return Verdict(Decision.REFUSE, Reason.SECRET_FIELD, word)
    term = lex.find("secret", _field_words(a))
    return Verdict(Decision.REFUSE, Reason.SECRET_FIELD, term) if term else None


def _risk(a: Action, lex: Lexicon, role: str) -> Verdict | None:
    parts = [normalize(a.label), normalize(a.hint)]
    if not _is_input(a, lex, role):
        parts.append(normalize(a.value))
    parts.append(normalize(a.role))
    parts.append(tuple(w for p in parts for w in p))  # all of it as one run, for a phrase split across fields
    for name in _RISKS:
        if term := lex.find(name, parts):
            return Verdict(Decision.ASK, Reason(name), term)
    return None


def _control(a: Action, lex: Lexicon, *, as_input: bool) -> Verdict:
    """Click, or Return pressed on the focused control."""
    role = lex.role(a.role)
    is_input = as_input or _is_input(a, lex, role)
    if is_input and (found := _secret(a, lex, role)):
        return found
    if not is_input and (found := _risk(a, lex, role)):
        return found
    named = any(normalize(t) for t in (a.label, a.hint, a.field_name)) or (not is_input and bool(normalize(a.value)))
    if not named:
        return Verdict(Decision.ASK, Reason.UNLABELED)
    if not is_input and role not in lex.known_roles:
        return Verdict(Decision.ASK, Reason.UNKNOWN)
    return ALLOW


def _submit(a: Action, lex: Lexicon) -> Verdict:
    """Return pressed, or a line break typed, with focus on the control `a` describes."""
    role = lex.role(a.role)
    if not _is_input(a, lex, role):
        return _control(a, lex, as_input=False)
    found = _control(a, lex, as_input=True)
    if found.decision is not Decision.ALLOW:
        return found
    words = _field_words(a)
    if lex.find("search", words):
        return ALLOW
    if not a.send_intent and (term := lex.find("message", words)):
        return Verdict(Decision.ASK, Reason.COMMUNICATION, term)
    return ALLOW


def _key(a: Action, lex: Lexicon) -> Verdict:
    role = lex.role(a.role)
    words = normalize(a.key)
    submits = bool(_LINE_BREAKS & set(a.key)) or bool(words and words[-1] in lex.submit_keys)
    if submits:
        secret = _secret(a, lex, role) if _is_input(a, lex, role) else None
        return secret or _submit(a, lex)
    if words and words[-1] in lex.neutral_keys:
        return ALLOW
    if _is_input(a, lex, role) and (secret := _secret(a, lex, role)):
        return secret
    return ALLOW


def _supplied(a: Action, text: str) -> bool:
    want = _plain(text)
    if any(want == _plain(s) for s in a.supplied):
        return True
    return bool(re.search(r"(?<!\w)" + re.escape(want) + r"(?!\w)", _plain(a.task_text)))


def _plain(text: str) -> str:
    """Spelling-sensitive but spacing- and case-insensitive: a look-alike letter is a different text."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _type(a: Action, lex: Lexicon) -> Verdict:
    role = lex.role(a.role)
    if found := _secret(a, lex, role):
        return found
    breaks = bool(_LINE_BREAKS & set(a.text))
    text = a.text.strip()
    verdicts = [ALLOW]
    if text and not _supplied(a, text):
        words = _field_words(a)
        if lex.find("search", words):
            pass
        elif (term := lex.find("personal", words)) and not lex.find("personal_except", words):
            verdicts.append(Verdict(Decision.HANDOFF, Reason.PERSONAL, term))
        elif not any(words):
            verdicts.append(Verdict(Decision.ASK, Reason.UNLABELED))
    if breaks:
        verdicts.append(_submit(a if _is_input(a, lex, role) else _as_field(a), lex))
    return _worst(*verdicts)


def _as_field(a: Action) -> Action:
    """Typing goes into a field whatever the page calls it, so a line break in it is Return in an input."""
    return replace(a, field_type=a.field_type or "text")
