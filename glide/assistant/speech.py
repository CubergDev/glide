"""Turning a reply into speech: cut text into sentences as it arrives, and speak each one at once.

Speech starts after the first sentence, not after the last, so the splitter works on a stream: text goes
in as the model writes it and a sentence comes out the moment it is certain to be finished. `Speaker`
then synthesises each sentence on a thread of its own while the player is still playing the one before,
and can drop everything in one call when the user says stop.

Nothing here knows a vendor. The TTS is anything with `stream(text, *, voice, language)` yielding
`(pcm, sample_rate)` chunks (providers/tts.py), the player is anything with `play`, `cancel` and
`wait_idle` (audio_io.py).
"""

from __future__ import annotations

import contextlib
import queue
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

# Full-width marks look like their ASCII twins (hence the RUF001 waivers): the ideographic full stop, the
# full-width ! ? and ;, the ellipsis character, the ideographic and full-width comma, the full-width colon.
STRONG_END = "。！？；"  # noqa: RUF001  a sentence is over the moment one of these is written, no space needed
LATIN_END = ".!?…"  # over only when what follows is a space, a new line, or the end: "3.5" and "a.b" go on
CLOSERS = "\"'”’」』）)]】》»"  # noqa: RUF001  quotes and brackets that stay with their sentence
CLAUSE_CUTS = ",;:，、："  # noqa: RUF001  where a very long run-on may be cut when it has no end of its own
OPENERS = "\"'(“‘「『（[【《«"  # noqa: RUF001

MAX_CHARS = 140  # a sentence this long with no end of its own is cut at its last comma instead of waiting
MIN_CLAUSE = 24  # and never so early that the first piece is a word or two

# A period after one of these does not end a sentence. Lower case, without the period.
ABBREVIATIONS = frozenset(
    {
        *("mr", "mrs", "ms", "mx", "dr", "prof", "sr", "jr", "st", "mt", "ft", "vs", "cf", "approx", "dept", "est", "fig"),
        *("inc", "ltd", "co", "corp", "gen", "col", "lt", "sgt", "capt", "rev", "hon"),
        *("jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"),
        *("mon", "tue", "wed", "thu", "fri", "sat", "sun"),
    }
)
# A single capital is an initial ("J. K. Rowling"). "I" and "A" end sentences far more often than they are initials.
SENTENCE_LETTERS = frozenset("IA")

# Characters written in Cantonese and all but absent from standard written Mandarin.
CANTONESE_ONLY = frozenset("嘅咗唔嘢冇啲喺佢哋咁嚟睇諗搵嗰")

_MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_URL = re.compile(r"https?://\S+")
_EMPHASIS = re.compile(r"\*\*|__|~~|(?<![\w*])\*(?=\S)|(?<=\S)\*(?![\w*])")
_LIST_MARKER = re.compile(r"^\s{0,3}(?:#{1,6}\s+|[-*+•]\s+|\d{1,3}[.)]\s+|>\s+)")
_EMOJI = re.compile("[\U0001f000-\U0001faff\U00002600-\U000027bf\U0000fe0f\U0000200d]")


def has_content(text: str) -> bool:
    """Whether there is anything to say: a letter or a digit, in any script. `"`, `)` and `**` are not speech."""
    return any(ch.isalnum() for ch in text)


def clean_for_speech(text: str) -> str:
    """The sentence as it should be read out: no markdown, links, bullets or emoji, one space between words.

    The answer prompt asks for plain speech already, so this is the net under it, not the plan.
    """
    text = _MARKDOWN_LINK.sub(r"\1", text)
    text = _URL.sub("", text)
    text = _LIST_MARKER.sub("", text)
    text = _EMPHASIS.sub("", text.replace("`", ""))
    text = _EMOJI.sub("", text)
    return " ".join(text.split())


def detect_language(text: str) -> str:
    """A language code for text whose language nobody said: 'yue', 'zh', 'ja', 'ko', or 'en' for the rest.

    Script is all it looks at. Written Cantonese has a handful of characters of its own, so one of them
    is enough to call it 'yue'; other Chinese text is 'zh'. Latin text is called 'en' even when it is
    French: the TTS chain falls back to its default voice for a language it has no voice for either way.
    """
    han = kana = hangul = 0
    for ch in text:
        if "一" <= ch <= "鿿":
            han += 1
        elif "぀" <= ch <= "ヿ":
            kana += 1
        elif "가" <= ch <= "힣":
            hangul += 1
    if kana:
        return "ja"
    if hangul:
        return "ko"
    if han:
        return "yue" if any(ch in CANTONESE_ONLY for ch in text) else "zh"
    return "en"


# -- Splitting --------------------------------------------------------------------------------------


def _is_cjk(ch: str) -> bool:
    return "⺀" <= ch <= "鿿" or "豈" <= ch <= "﫿" or "＀" <= ch <= "￯"


def _word_before(buf: str, end: int) -> str:
    """The run of non-space characters that ends at `end`, without the brackets and quotes that open it."""
    start = end
    while start > 0 and not buf[start - 1].isspace():
        start -= 1
    return buf[start:end].lstrip(OPENERS)


def _is_abbreviation(word: str) -> bool:
    if word.lower() in ABBREVIATIONS:
        return True
    if "." in word:  # "e.g", "i.e", "U.S", "a.m", "Ph.D": every piece is one or two letters
        return all(0 < len(piece) <= 2 and piece.isalpha() for piece in word.split("."))
    return len(word) == 1 and word.isupper() and word not in SENTENCE_LETTERS  # an initial


def _period_ends(buf: str, start: int, end: int, after: int) -> bool:
    """Whether the run of periods buf[start:end] ends a sentence. `after` indexes the next word, or is len(buf)."""
    if end - start == 1:
        if buf[:start].strip().isdigit():
            return False  # "1. Open the app": a list number, not the end of a sentence
        if _is_abbreviation(_word_before(buf, start)):
            return False
    return after >= len(buf) or not buf[after].islower()  # "e.g. this" and "approx. five" go on


def _boundary(buf: str, final: bool) -> int | None:
    """Where the first finished sentence in `buf` ends, or None when it cannot be sure yet.

    A period, `?` or `!` at the very end of the buffer, or followed only by spaces, is undecided: the
    next character may be a digit ("3." then "5"), a lower-case word, or more punctuation. `final`
    says no more text is coming, so what is left is decided as it stands.
    """
    n = len(buf)
    i = 0
    while i < n:
        ch = buf[i]
        if ch == "\n":
            if buf[:i].strip():
                return i + 1
        elif ch in STRONG_END:
            j = i + 1
            while j < n and (buf[j] in STRONG_END or buf[j] in LATIN_END):
                j += 1
            while j < n and buf[j] in CLOSERS:
                j += 1
            return j
        elif ch in LATIN_END:
            j = i
            while j < n and buf[j] in LATIN_END:
                j += 1
            k = j
            while k < n and buf[k] in CLOSERS:
                k += 1
            if k >= n:
                if final:
                    return n
                return None
            if buf[k].isspace():
                after = k
                while after < n and buf[after] in " \t":
                    after += 1
                if after >= n and not final:
                    return None
                if any(c != "." for c in buf[i:j]) or _period_ends(buf, i, j, after):
                    return k
            elif _is_cjk(buf[k]):
                return k
            i = j
            continue
        i += 1
    return None


def _clause_cut(buf: str) -> int | None:
    """The end of the last clause in a long buffer, or None. Only commas that cannot be part of a number."""
    for p in range(len(buf) - 2, MIN_CLAUSE - 1, -1):
        ch = buf[p]
        if ch not in CLAUSE_CUTS:
            continue
        if ch in ",;:" and not buf[p + 1].isspace():
            continue  # "1,000", "10:30"
        return p + 1
    return None


class SentenceSplitter:
    """Sentences out of text that arrives in pieces of any size.

    `feed` returns the sentences finished by that piece, `flush` the rest once the text has ended. A
    sentence is returned as written (stripped); pieces with no letter or digit in them are dropped.
    Feeding a text in one piece or a character at a time gives the same sentences, except that a
    closing quote arriving after an ideographic full stop is left behind (and dropped), and a
    sentence longer than `max_chars` with no end is cut at a comma wherever the buffer stood.
    """

    def __init__(self, *, max_chars: int = MAX_CHARS) -> None:
        self._buf = ""
        self._max_chars = max_chars

    def feed(self, text: str) -> list[str]:
        self._buf += text
        return self._drain(final=False)

    def flush(self) -> list[str]:
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> list[str]:
        out: list[str] = []
        while self._buf:
            end = _boundary(self._buf, final)
            if end is None and not final and len(self._buf) >= self._max_chars:
                end = _clause_cut(self._buf)
            if end is None:
                break
            sentence, self._buf = self._buf[:end].strip(), self._buf[end:]
            if has_content(sentence):
                out.append(sentence)
        if final:
            rest, self._buf = self._buf.strip(), ""
            if has_content(rest):
                out.append(rest)
        return out


def split_sentences(text: str) -> list[str]:
    """All the sentences of a finished text. The streaming `SentenceSplitter` in one call."""
    splitter = SentenceSplitter()
    return [*splitter.feed(text), *splitter.flush()]


# -- Speaking ---------------------------------------------------------------------------------------


@dataclass
class _Lane:
    """The sentences queued since the last cancel, and the thread that voices them.

    A cancel does not wait for a thread that may be blocked in a network call: it abandons the lane and
    the next sentence gets a new one, so speech after a barge-in never queues behind the old request.
    """

    items: queue.Queue = field(default_factory=queue.Queue)
    pending: int = 0
    dead: bool = False


class Speaker:
    """Sentences in, audio out, cancellable at any moment.

    `say` returns at once. A thread synthesises the sentences in order and hands each chunk to the
    player as it arrives, so the next sentence is being made while the last one plays. `cancel` drops
    what is queued, abandons what is being made and silences the player; anything said afterwards is
    heard in full. A TTS failure after the chain has tried every voice is reported to `on_error` and
    the next sentence is tried: one lost sentence is better than a reply that goes silent.
    """

    def __init__(self, tts, player, *, on_error: Callable[[BaseException], None] | None = None, clock=time.monotonic) -> None:
        self._tts = tts
        self._player = player
        self._on_error = on_error
        self._clock = clock
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._lane: _Lane | None = None
        self.first_audio_at: float | None = None  # when the first chunk reached the player since `mark`

    def __repr__(self) -> str:
        return f"<Speaker pending={self._lane.pending if self._lane else 0}>"

    def mark(self) -> None:
        """Start timing: `first_audio_at` is cleared and set again by the next chunk that reaches the player."""
        self.first_audio_at = None

    def say(
        self, text: str, *, language: str | None = None, voice: str | None = None, only_if: Callable[[], bool] | None = None
    ) -> bool:
        """Queue one sentence. False when there was nothing to say or `only_if` (checked under the lock
        that `cancel` takes, so a cancel cannot slip between the check and the queueing) said no."""
        cleaned = clean_for_speech(text)
        if not has_content(cleaned):
            return False
        with self._lock:
            if only_if is not None and not only_if():
                return False
            if self._lane is None:
                self._lane = lane = _Lane()
                threading.Thread(target=self._run, args=(lane,), name="glide-speaker", daemon=True).start()
            lane = self._lane
            lane.pending += 1
            lane.items.put((cleaned, language, voice))
        return True

    def cancel(self) -> None:
        """Silence now: queued sentences are dropped, the one being made is abandoned, the player is cut."""
        with self._cond:
            lane, self._lane = self._lane, None
            if lane is not None:
                lane.dead = True
                lane.pending = 0
                while True:
                    try:
                        lane.items.get_nowait()
                    except queue.Empty:
                        break
                lane.items.put(None)
            self._player.cancel()
            self._cond.notify_all()

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until everything said so far has been played. False if `timeout` ran out first."""
        deadline = None if timeout is None else self._clock() + timeout
        with self._cond:
            while self._lane is not None and self._lane.pending > 0:
                remaining = None if deadline is None else deadline - self._clock()
                if remaining is not None and remaining <= 0:
                    return False
                self._cond.wait(remaining)
        remaining = None if deadline is None else max(0.0, deadline - self._clock())
        return self._player.wait_idle(remaining)

    def close(self) -> None:
        self.cancel()
        close = getattr(self._player, "close", None)
        if close is not None:
            close()

    # -- the lane's thread ----------------------------------------------------------------------

    def _run(self, lane: _Lane) -> None:
        while True:
            item = lane.items.get()
            if item is None:
                return
            text, language, voice = item
            try:
                self._speak(lane, text, language, voice)
            except Exception as exc:  # a provider's error, or a bug: either way the next sentence still gets its turn
                if not lane.dead and self._on_error is not None:
                    with contextlib.suppress(Exception):  # a broken reporter must not end the speech
                        self._on_error(exc)
            finally:
                with self._cond:
                    if not lane.dead:
                        lane.pending -= 1
                    self._cond.notify_all()

    def _speak(self, lane: _Lane, text: str, language: str | None, voice: str | None) -> None:
        chunks = self._tts.stream(text, language=language, voice=voice)
        try:
            for chunk in chunks:
                pcm, rate = chunk
                with self._lock:
                    if lane.dead:
                        return
                    if self.first_audio_at is None:
                        self.first_audio_at = self._clock()
                    self._player.play(pcm, rate)
        finally:
            close = getattr(chunks, "close", None)
            if close is not None:
                close()  # a cancelled sentence gives its HTTP connection back now
