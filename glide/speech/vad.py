"""Who is speaking, and where a turn begins and ends, at 16 kHz in frames of 512 samples (32 ms).

A voice detector is any callable taking one frame of PCM and returning the probability that it holds speech.
`Silero` is the good one (it needs the `speech` extra: numpy and onnxruntime, and a model file whose
location and SHA-256 come from the `[speech]` table, never from this module). `EnergyProbability` needs
nothing and is the fallback: loudness against a floor that follows the room.

`TurnDetector` turns those probabilities into turns. Only silence ends a turn. It never commits early on a
shorter silence: an earlier version flushed the transcript proactively when the speaker paused, collided with
the end of the turn 154 ms later, and the transcriber refused the second commit (`commit_throttled`) and
closed. One utterance, one end, decided here and nowhere else.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import deque
from collections.abc import Callable
from pathlib import Path

from ..assistant.audio_io import SAMPLE_BYTES, SAMPLE_RATE, NoiseFloor, rms

FRAME_SAMPLES = 512
FRAME_BYTES = FRAME_SAMPLES * SAMPLE_BYTES
FRAME_S = FRAME_SAMPLES / SAMPLE_RATE  # 32 ms

START_PROBABILITY = 0.5  # a frame at least this likely to be speech starts a turn
STOP_PROBABILITY = 0.35  # a frame below this counts as silence; between the two it neither starts nor ends anything
PRE_ROLL_FRAMES = 8  # what was heard just before a turn started is kept, so the first syllable is not clipped
MAX_UTTERANCE_S = 60.0
MIN_SILENCE_MS, MAX_SILENCE_MS = 200, 2000  # how long a pause may be asked to end a turn

Probability = Callable[[bytes], float]
_SHA256 = re.compile(r"[0-9a-f]{64}")


class VadError(RuntimeError):
    """The voice detector could not be set up. The message never carries a key, a header or audio."""


class UtteranceTooLong(ValueError):
    """Speech ran on past the limit: the turn is discarded rather than submitted half-heard."""


class EnergyProbability(NoiseFloor):
    """Speech or not by loudness alone, as 1.0 or 0.0, against the room's floor (`NoiseFloor`, as `Endpointer` in
    assistant/audio_io.py uses it for push-to-talk)."""

    def __call__(self, pcm: bytes) -> float:
        level = rms(pcm)
        if self.is_speech(level):
            return 1.0
        self.learn(level)
        return 0.0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_sha256(value: str) -> str:
    value = value.strip().lower()
    if not _SHA256.fullmatch(value):
        raise VadError("the VAD model checksum must be 64 hexadecimal characters")
    return value


def _download(url: str) -> bytes:
    import httpx

    try:
        return httpx.get(url, timeout=30, follow_redirects=False).raise_for_status().content
    except httpx.HTTPError as exc:
        raise VadError(f"could not download the VAD model ({type(exc).__name__})") from exc


def install_model(path: Path, *, url: str, sha256: str, fetch: Callable[[str], bytes] | None = None) -> None:
    """Download the model to `path`, if and only if its SHA-256 is the configured one.

    `url` and `sha256` come from the `[speech]` table. Both are required: an install without a pinned
    checksum would trust whatever the address serves. The file is written beside its destination and renamed
    into place, so a crash never leaves a half-written model that a later run might load.
    """
    if not url.startswith("https://"):
        raise VadError("the VAD model must be fetched over https")
    sha256 = check_sha256(sha256)
    data = (fetch or _download)(url)
    if hashlib.sha256(data).hexdigest() != sha256:
        raise VadError("the downloaded VAD model does not match the configured checksum; nothing was saved")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(data)
    temporary.replace(path)


class Silero:
    """Silero VAD through onnxruntime. Call it with one frame of 512 samples; it keeps the model's state between calls.

    Needs `numpy` and `onnxruntime` (the `speech` extra), imported here and nowhere earlier. The file must
    match `sha256` or it is refused: a model that was swapped on disk is code that was swapped on disk.
    """

    def __init__(self, path: Path, sha256: str) -> None:
        sha256 = check_sha256(sha256)
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError as exc:
            raise VadError("Silero needs numpy and onnxruntime (the speech extra)") from exc
        if not path.is_file() or file_sha256(path) != sha256:
            raise VadError("the VAD model file is missing or does not match the configured checksum")
        options = ort.SessionOptions()
        options.inter_op_num_threads = options.intra_op_num_threads = 1
        self._np = np
        self._session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self) -> None:
        np = self._np
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)

    def __call__(self, pcm: bytes) -> float:
        np = self._np
        if len(pcm) != FRAME_BYTES:
            raise ValueError(f"Silero takes {FRAME_SAMPLES} samples per frame")
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32).reshape(1, -1) / 32768
        samples = np.concatenate((self._context, samples), axis=1)
        output, self._state = self._session.run(
            None, {"input": samples, "state": self._state, "sr": np.array(SAMPLE_RATE, dtype=np.int64)}
        )
        self._context = samples[:, -64:]
        return float(output[0, 0])


class TurnDetector:
    """Turns per-frame probabilities into turns. `feed(frame, probability)` returns `(started, pcm, commit)`.

    Before a turn, frames only fill a short pre-roll. The first frame at least `START_PROBABILITY` likely to be
    speech starts a turn and `pcm` is the whole pre-roll plus that frame, once. After that every frame is
    returned as it came, and the turn ends (`commit`) on the frame that makes `silence_ms` of quiet in a
    row. The commit frame is part of the turn. A turn longer than `MAX_UTTERANCE_S` raises `UtteranceTooLong`
    and ends: a command that long was not a command, and half of it must not be acted on.
    """

    def __init__(self, silence_ms: int = 600, *, max_utterance_s: float = MAX_UTTERANCE_S) -> None:
        if not MIN_SILENCE_MS <= silence_ms <= MAX_SILENCE_MS:
            raise ValueError(f"Silence threshold must be {MIN_SILENCE_MS}-{MAX_SILENCE_MS} ms.")
        self.silence_frames = math.ceil(silence_ms * SAMPLE_RATE / 1000 / FRAME_SAMPLES)
        self.max_frames = int(max_utterance_s * SAMPLE_RATE / FRAME_SAMPLES)
        self.pre_roll: deque[bytes] = deque(maxlen=PRE_ROLL_FRAMES)
        self.active = False
        self.silent = 0
        self.frames = 0

    def feed(self, pcm: bytes, probability: float) -> tuple[bool, bytes, bool]:
        started = False
        if not self.active:
            self.pre_roll.append(pcm)
            if probability < START_PROBABILITY:
                return False, b"", False
            self.active = started = True
            pcm = b"".join(self.pre_roll)
            self.pre_roll.clear()
            self.frames = self.silent = 0
        self.frames += 1
        self.silent = self.silent + 1 if probability < STOP_PROBABILITY else 0
        if self.frames > self.max_frames:
            self.active = False
            raise self.too_long()
        commit = self.silent >= self.silence_frames
        if commit:
            self.active = False
        return started, pcm, commit

    def too_long(self) -> UtteranceTooLong:
        return UtteranceTooLong(f"Utterance exceeded {self.max_frames * FRAME_S:.0f} seconds; it was not submitted.")

    def finish(self) -> bool:
        """End the turn now, as a silence would. True only if a turn was running, so a second call (or a call
        after the silence already ended the turn) ends nothing and commits nothing."""
        if not self.active:
            return False
        self.active = False
        return True

    def reset(self) -> None:
        self.pre_roll.clear()
        self.active = False
        self.silent = self.frames = 0
