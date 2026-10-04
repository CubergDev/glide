"""The `[speech]` table of glide.toml, validated once. Nothing here names a vendor, a model or a voice.

```toml
[speech]
headset = false          # true: the microphone stays open while Glide speaks, so you can talk over it
silence_ms = 600         # how long a pause ends a turn (200-2000)
merge_window_s = 0.0     # a pause shorter than this does not end the turn: "open Safari ... no, Chrome" is one request
idle_s = 0.0             # microphone off after this long with nothing happening (0 = never)
vad = "auto"             # "silero" needs the speech extra and a model; "energy" needs nothing; "auto" picks silero if configured
vad_model_path = ""      # the Silero ONNX file, its checksum, and where to fetch it from (https only)
vad_model_sha256 = ""
vad_model_url = ""
echo_canceller = "auto"  # speaker mode: take Glide's voice out of the microphone so you can talk over it. "webrtc" (aec extra), "nlms" (numpy), "none" (half duplex), "auto" (first that loads, and says which)
barge_min_voiced_ms = 190  # how long you must speak over Glide before it stops for you (96-1000); shorter sounds are backchannel
barge_margin_db = 8.0      # how far above the echo still left in the microphone your voice must be (3-30)
barge_min_erle_db = 6.0    # no interruption by voice until the canceller has removed at least this much echo (0-30)
stop_phrases = []          # whole-utterance phrases that stop Glide, besides the built-in ones ("stop", "never mind", ...)
```

Keys come from this table or from nowhere: there are no defaults for a model location or a checksum, because
a pinned download that was not configured is not a pinned download.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields

from ..assistant.router import normalize
from ..providers.config import ConfigError
from .echo import ECHO_CHOICES
from .vad import MAX_SILENCE_MS, MIN_SILENCE_MS

VAD_CHOICES = ("auto", "silero", "energy")
OUTPUT_RATES = (16000, 22050, 24000, 44100, 48000)


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _whole(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _text(v) -> bool:
    return isinstance(v, str)


# What each field's annotation allows, and the words for a value that is not one. A bool is not a number here.
_KINDS = {
    "bool": (lambda v: isinstance(v, bool), "must be true or false"),
    "int": (_whole, "must be a whole number"),
    "float": (_number, "must be a number"),
    "str": (_text, "must be text"),
    "str | None": (_text, "must be text"),
    "int | str | None": (lambda v: _whole(v) or _text(v), "must be a device number or name"),
    "tuple[str, ...]": (lambda v: isinstance(v, (list, tuple)) and all(map(_text, v)), "must be a list of text"),
}
# (key, lowest, highest or None for "no upper limit", what follows the range in the message), checked in this order
_RANGES = (
    ("silence_ms", MIN_SILENCE_MS, MAX_SILENCE_MS, ""),
    ("merge_window_s", 0, 5, " seconds"),
    ("idle_s", 0, None, ""),
    ("echo_tail_s", 0, 2, " seconds"),
    ("barge_min_voiced_ms", 96, 1000, ""),
    ("barge_margin_db", 3, 30, ""),
    ("barge_min_erle_db", 0, 30, ""),
)
_CHOICES = (("output_rate", OUTPUT_RATES), ("vad", VAD_CHOICES), ("echo_canceller", ECHO_CHOICES))


@dataclass(frozen=True)
class SpeechSettings:
    """Every `[speech]` key with its default. These defaults are the only ones: the loop, the gate and the device take
    theirs from here, so a number has one home."""

    headset: bool = False
    silence_ms: int = 600
    merge_window_s: float = 0.0
    idle_s: float = 0.0
    echo_tail_s: float = 0.3
    output_rate: int = 24000
    input_device: int | str | None = None
    output_device: int | str | None = None
    language: str | None = None
    vad: str = "auto"
    vad_model_path: str = ""
    vad_model_sha256: str = ""
    vad_model_url: str = ""
    echo_canceller: str = "auto"
    barge_min_voiced_ms: int = 190
    barge_margin_db: float = 8.0
    barge_min_erle_db: float = 6.0
    stop_phrases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, low, high, unit in _RANGES:
            value = getattr(self, name)
            if value < low or (high is not None and value > high):
                wanted = "not be negative" if high is None else f"be {low}-{high}{unit}"
                raise ConfigError(f"[speech] {name} must {wanted}, not {value}")
        for name, options in _CHOICES:
            if getattr(self, name) not in options:
                raise ConfigError(f"[speech] {name} must be one of {', '.join(map(str, options))}, not {getattr(self, name)!r}")
        if self.vad == "silero" and not self.silero_configured:
            raise ConfigError("[speech] vad = 'silero' needs vad_model_path and vad_model_sha256")
        if not all(isinstance(p, str) and normalize(p) for p in self.stop_phrases):
            raise ConfigError("[speech] stop_phrases must be a list of non-empty phrases")
        if self.vad_model_url and not self.vad_model_url.startswith("https://"):
            raise ConfigError("[speech] vad_model_url must be an https:// address")

    @property
    def silero_configured(self) -> bool:
        return bool(self.vad_model_path and self.vad_model_sha256)

    @classmethod
    def from_mapping(cls, table: Mapping | None) -> SpeechSettings:
        """The settings in a `[speech]` table. An unknown key is an error: a misspelt `headset` must not silently mean speakers."""
        table = dict(table or {})
        known = {f.name: f.type for f in fields(cls)}
        unknown = sorted(set(table) - set(known))
        if unknown:
            raise ConfigError(f"[speech] has unknown keys: {', '.join(unknown)} (known: {', '.join(sorted(known))})")
        values = {}
        for name, value in table.items():
            accepts, words = _KINDS[known[name]]
            if not accepts(value):
                raise ConfigError(f"[speech] {name} {words}")
            values[name] = float(value) if known[name] == "float" else tuple(value) if known[name].startswith("tuple") else value
        return cls(**values)
