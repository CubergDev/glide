"""The one list of stop phrases, and the deterministic test for them. No model, no network, no state.

Everything that has to recognise "stop" calls this: the router's first tier (`glide.routing.router`), the voice
loop's stop-only listen, the typed `/stop` line and the `[speech] stop_phrases` setting (which only ADDS phrases to
the built-in ones here). The matcher is moved here unchanged from `glide/assistant/router.py`; the old module
re-exports these names, and `tests/test_routing_stop.py` runs both through the same cases so the swap cannot
change behaviour.

The rule, in one line: the WHOLE utterance must be a stop and nothing else ("stop the music" is not one: "music" is
in neither word set, so it goes on to the router like any other request).
"""

from __future__ import annotations

import unicodedata


def normalize(text: str) -> str:
    """Lower case, full-width forms made ordinary, punctuation and symbols turned to spaces, spaces collapsed."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join("".join(" " if unicodedata.category(ch)[0] in "PSZC" else ch for ch in text).split())


# A whole utterance made only of these words, with at least one of the first set, is a stop. "stop the music"
# is not: "music" is in neither, so it goes to the router like any other request.
STOP_CORE = frozenset({"stop", "cancel", "abort", "halt", "enough", "quiet", "silence", "nevermind"})
STOP_FILLER = frozenset(
    {"please", "now", "it", "that", "this", "glide", "hey", "ok", "okay", "yes", "the", "task", "everything", "all"}
    | {"talking", "speaking", "right", "just", "thanks", "thank", "you"}
)
STOP_PHRASES = frozenset(
    normalize(p) for p in ("never mind", "shut up", "be quiet", "that's enough", "stop it", "quiet please", "forget it")
)

# Cantonese and Mandarin, compared without spaces and without the particles that end a sentence.
STOP_CJK = frozenset(
    {"停", "停止", "停下", "停低", "停下來", "停下来", "取消", "算了", "算啦", "算喇", "算數", "算数", "夠了", "够了", "夠喇"}
    | {"別說了", "别说了", "唔好講", "唔好說", "收聲", "閉嘴", "闭嘴", "安靜", "安静", "中止", "終止", "终止", "唔使", "不用了"}
    | {"唔好做", "不要了", "別做了", "别做了", "唔使喇", "唔好講嘢", "不要說了", "不要说了"}
)
CJK_POLITE_PREFIX = ("唔該", "唔该", "麻煩", "麻烦", "請", "请", "快", "即刻", "馬上", "马上")
CJK_PARTICLES = "啦喇吧呀啊嘅囉咯了喔哦嘛"


def _is_cjk_stop(compact: str) -> bool:
    for prefix in CJK_POLITE_PREFIX:
        compact = compact.removeprefix(prefix)
    compact = compact.rstrip(CJK_PARTICLES) if compact not in STOP_CJK else compact
    if compact in STOP_CJK:
        return True
    # repeated for emphasis: 停停停
    return any(p and len(compact) % len(p) == 0 and compact == p * (len(compact) // len(p)) for p in STOP_CJK if len(p) <= 2)


def stop_phrases(phrases) -> frozenset[str]:
    """Extra whole-utterance stop phrases (from configuration), normalised the way an utterance is."""
    return frozenset(filter(None, map(normalize, phrases)))


def is_stop(text: str, extra: frozenset[str] = frozenset()) -> bool:
    """Whether the whole utterance is a request to stop. Not a substring match: it must be nothing else.

    `extra` holds more phrases to treat as a stop, from `stop_phrases()`; each must be the whole utterance.
    """
    norm = normalize(text)
    if not norm:
        return False
    if norm in STOP_PHRASES or norm in extra:
        return True
    tokens = norm.split()
    if all(t in STOP_CORE or t in STOP_FILLER for t in tokens) and any(t in STOP_CORE for t in tokens):
        return True
    return _is_cjk_stop("".join(tokens))
