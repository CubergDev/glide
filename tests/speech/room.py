"""A simulated room for the echo-cancellation and barge-in proofs. Needs numpy; the tests that use it skip without it.

Nothing here touches a sound card. The pieces:

- `speechlike(...)`: a speech-like signal (voiced harmonics with a moving pitch and formant-shaped spectrum, a
  syllable envelope, noise-like fricatives, pauses). It is a stand-in: real speech is richer, and a recorded
  corpus would be better evidence. What it does share with speech is what echo cancellers care about: strong
  correlation across short lags, wide dynamic range and no stationarity.
- `EchoPath`: what the speaker does to the microphone: a delay, a decaying random room response, a little
  loudspeaker colouring and (optionally) a soft clip, streamed block by block.
- `SimRoom`: a real `FullDuplexDevice`, driven by its own callbacks in lockstep with an injected clock. Each
  `step()` asks the output callback for one frame (what the speaker would play), passes it through the echo
  path, adds the person's voice and the room's noise, and gives the result to the input callback. `read()` then
  returns what the loop would get: the cleaned frame (and the device's canceller has seen the far end exactly as
  the real device hands it over).
"""

from __future__ import annotations

import numpy as np

from glide.speech.audio import FullDuplexDevice
from glide.speech.vad import FRAME_BYTES, FRAME_S, FRAME_SAMPLES, SAMPLE_RATE

FS = SAMPLE_RATE


def speechlike(seconds: float, seed: int, *, level: float = 3000.0, pause: float = 0.2) -> np.ndarray:
    """`seconds` of speech-like signal at the given RMS (16-bit scale), deterministic in `seed`."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    out = np.zeros(n)
    pos = 0
    while pos < n:
        seg = min(int(rng.uniform(0.15, 0.5) * FS), n - pos)
        if rng.random() < pause:
            pos += seg
            continue
        t = np.arange(seg) / FS
        if rng.random() < 0.2:  # a fricative: high-passed noise
            x = np.convolve(rng.standard_normal(seg), [1.0, -0.9])[:seg] * 0.5
        else:
            f0 = rng.uniform(90, 240)
            pitch = f0 * (1 + rng.uniform(-0.3, 0.3) * t / max(t[-1], 1e-3))
            phase = 2 * np.pi * np.cumsum(pitch) / FS
            formants = [rng.uniform(300, 900), rng.uniform(1000, 2500), rng.uniform(2500, 3500)]
            x = np.zeros(seg)
            for h in range(1, int(3800 / f0) + 1):
                weight = sum(np.exp(-0.5 * ((h * f0 - c) / (250 + 0.1 * c)) ** 2) for c in formants) + 0.02
                x += weight * np.sin(h * phase + rng.uniform(0, 2 * np.pi))
        out[pos : pos + seg] += x * np.sin(np.pi * np.arange(seg) / seg) ** 0.7
        pos += seg
    return out * (level / (np.sqrt(np.mean(out**2)) + 1e-9))


class EchoPath:
    """Speaker to microphone: `delay_ms` of nothing, then a decaying room response, scaled to `gain` (RMS of the echo
    over RMS of what was played, for white input). Streamed: `__call__` takes one block and returns one block."""

    def __init__(
        self, delay_ms: float, seed: int, *, gain: float = 0.5, rt60: float = 0.18, tail_ms: float = 350.0, soft_clip: float = 0.0
    ):
        rng = np.random.default_rng(seed)
        tail_n = int(tail_ms / 1000 * FS)
        delay = int(delay_ms / 1000 * FS)
        tail = rng.standard_normal(tail_n) * np.exp(-6.9 * np.arange(tail_n) / (rt60 * FS)) * 0.25
        tail[0] = 1.0
        ir = np.zeros(delay + tail_n)
        ir[delay:] = tail
        ir = np.convolve(ir, [0.5, 0.3, 0.2])[: len(ir)]  # loudspeaker and microphone colouring
        self.ir = ir * gain / np.sqrt(np.sum(ir**2))
        self.soft_clip = soft_clip
        size = 1 << int(np.ceil(np.log2(FRAME_SAMPLES + len(self.ir))))
        self._size = size
        self._h = np.fft.rfft(self.ir, size)
        self._carry = np.zeros(len(self.ir))

    def __call__(self, block: np.ndarray) -> np.ndarray:
        n = len(block)
        y = np.fft.irfft(np.fft.rfft(block, self._size) * self._h, self._size)[: n + len(self.ir) - 1]
        y[: len(self._carry)] += self._carry
        out, self._carry = y[:n].copy(), y[n:]
        if self.soft_clip:
            out = np.tanh(out * self.soft_clip / 10000) * 10000 / self.soft_clip
        return out


def pcm(x: np.ndarray) -> bytes:
    return np.clip(np.rint(x), -32768, 32767).astype("<i2").tobytes()


def samples(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float64)


class _Stream:
    def start(self): ...
    def stop(self): ...
    def close(self): ...


class SimRoom(FullDuplexDevice):
    """A full-duplex device whose microphone is a simulated room, driven in lockstep (16 kHz both ways)."""

    def __init__(self, canceller, path: EchoPath, *, user: np.ndarray | None = None, noise: float = 20.0, seed: int = 0, **kw):
        self.now = 100.0
        super().__init__(
            output_rate=SAMPLE_RATE,
            clock=lambda: self.now,
            input_factory=lambda cb: _Stream(),
            output_factory=lambda cb, rate: _Stream(),
            canceller=canceller,
            **kw,
        )
        self.path = path
        self.user = np.zeros(0) if user is None else user
        self.noise = noise
        self._rng = np.random.default_rng(seed)
        self.frame_no = 0
        self.played: list[bytes] = []  # what the speaker was handed, frame by frame
        self.heard: list[np.ndarray] = []  # the microphone, before cancelling
        self.cleaned: list[bytes] = []  # what `read()` returned
        self.cancels: list[int] = []  # the frame number at which `cancel()` was called
        self.on_step = None
        self._started = False
        self.stop_after: int | None = None

    def cancel(self) -> None:
        self.cancels.append(self.frame_no)
        super().cancel()

    def step(self) -> None:
        """One 32 ms of the room: the speaker plays, the microphone hears it and the person."""
        if not self._started:
            self._started = True
            self.start()
        k = self.frame_no
        far = self._on_output(FRAME_BYTES)
        self.played.append(far)
        echo = self.path(samples(far))
        user = np.zeros(FRAME_SAMPLES)
        part = self.user[k * FRAME_SAMPLES : (k + 1) * FRAME_SAMPLES]
        user[: len(part)] = part
        mic = echo + user + self._rng.standard_normal(FRAME_SAMPLES) * self.noise
        self.heard.append(mic)
        self.now += FRAME_S
        self._on_input(pcm(mic), False)
        self.frame_no += 1
        if self.on_step is not None:
            self.on_step(self)

    def read(self, timeout=None):
        if self.stop_after is not None and self.frame_no >= self.stop_after:
            self.loop_stop()
            return None
        self.step()
        frame = super().read(timeout=None)
        if frame is not None:
            self.cleaned.append(frame)
        return frame

    loop_stop = staticmethod(lambda: None)  # replaced by the test with the loop's stop


def erle_db(heard: np.ndarray, cleaned: np.ndarray) -> float:
    return 10 * np.log10(np.sum(heard**2) / max(np.sum(cleaned**2), 1e-9))
