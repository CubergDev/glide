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

VAD_CHOICES = ("auto", "silero", "energy")
MIN_SILENCE_MS, MAX_SILENCE_MS = 200, 2000
OUTPUT_RATES = (16000, 22050, 24000, 44100, 48000)


@dataclass(frozen=True)
class SpeechSettings:
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
        if not MIN_SILENCE_MS <= self.silence_ms <= MAX_SILENCE_MS:
            raise ConfigError(f"[speech] silence_ms must be {MIN_SILENCE_MS}-{MAX_SILENCE_MS}, not {self.silence_ms}")
        if not 0 <= self.merge_window_s <= 5:
            raise ConfigError(f"[speech] merge_window_s must be 0-5 seconds, not {self.merge_window_s}")
        if self.idle_s < 0:
            raise ConfigError(f"[speech] idle_s must not be negative, not {self.idle_s}")
        if not 0 <= self.echo_tail_s <= 2:
            raise ConfigError(f"[speech] echo_tail_s must be 0-2 seconds, not {self.echo_tail_s}")
        if self.output_rate not in OUTPUT_RATES:
            raise ConfigError(f"[speech] output_rate must be one of {', '.join(map(str, OUTPUT_RATES))}, not {self.output_rate}")
        if self.vad not in VAD_CHOICES:
            raise ConfigError(f"[speech] vad must be one of {', '.join(VAD_CHOICES)}, not {self.vad!r}")
        if self.vad == "silero" and not self.silero_configured:
            raise ConfigError("[speech] vad = 'silero' needs vad_model_path and vad_model_sha256")
        if self.echo_canceller not in ECHO_CHOICES:
            raise ConfigError(f"[speech] echo_canceller must be one of {', '.join(ECHO_CHOICES)}, not {self.echo_canceller!r}")
        if not 96 <= self.barge_min_voiced_ms <= 1000:
            raise ConfigError(f"[speech] barge_min_voiced_ms must be 96-1000, not {self.barge_min_voiced_ms}")
        if not 3 <= self.barge_margin_db <= 30:
            raise ConfigError(f"[speech] barge_margin_db must be 3-30, not {self.barge_margin_db}")
        if not 0 <= self.barge_min_erle_db <= 30:
            raise ConfigError(f"[speech] barge_min_erle_db must be 0-30, not {self.barge_min_erle_db}")
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
            kind = known[name]
            if kind == "bool" and not isinstance(value, bool):
                raise ConfigError(f"[speech] {name} must be true or false")
            if kind == "int" and (isinstance(value, bool) or not isinstance(value, int)):
                raise ConfigError(f"[speech] {name} must be a whole number")
            if kind == "float" and (isinstance(value, bool) or not isinstance(value, (int, float))):
                raise ConfigError(f"[speech] {name} must be a number")
            if kind == "str" and not isinstance(value, str):
                raise ConfigError(f"[speech] {name} must be text")
            if kind.startswith("tuple"):
                if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
                    raise ConfigError(f"[speech] {name} must be a list of text")
                value = tuple(value)
            values[name] = float(value) if kind == "float" else value
        return cls(**values)
