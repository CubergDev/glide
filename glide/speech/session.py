"""Putting the voice stack together from `[speech]` settings. This is the only module that builds real hardware.

`build_voice(config, settings)` makes the device (with an echo canceller in speaker mode, so the person can talk
over Glide), the voice detector, an `Assistant` that speaks through the device, and a `VoiceLoop` over them. Each
part can be passed in, which is how tests (and any other front end) use it without a sound card.

Voice detector choice is visible: `vad = "silero"` that cannot be built is an error, and `vad = "auto"` that
falls back to loudness says so through `io.warn` instead of quietly listening worse. The echo canceller is chosen
the same way (`echo_canceller`, see echo.py): a named one that cannot be built is an error, and `auto` says which
fallback it took, or that speaker mode stays half duplex. A headset gets none: it hears no echo.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from ..assistant.core import IO, Assistant
from .audio import FullDuplexDevice
from .echo import make_canceller
from .settings import SpeechSettings
from .turns import VoiceLoop, format_status
from .vad import EnergyProbability, Probability, Silero, VadError


def make_vad(settings: SpeechSettings, warn=lambda message: None) -> Probability:
    if settings.vad == "energy" or (settings.vad == "auto" and not settings.silero_configured):
        return EnergyProbability()
    try:
        return Silero(Path(settings.vad_model_path).expanduser(), settings.vad_model_sha256)
    except VadError as exc:
        if settings.vad == "silero":
            raise
        warn(f"voice detection falls back to loudness: {exc}")
        return EnergyProbability()


def build_voice(
    config,
    settings: SpeechSettings,
    *,
    io: IO | None = None,
    act: bool = False,
    device=None,
    vad: Probability | None = None,
    on_idle=None,
    assistant_factory: Callable[..., Assistant] = Assistant,
) -> VoiceLoop:
    """A `VoiceLoop`, not yet started: call `start()` (or `run()`), and `stop()` then `loop.assistant.close()` to end.

    `act=False` keeps every computer task a dry run, as everywhere else. `assistant_factory(config, io=io)` makes the
    assistant: a front end that wants to watch what each request came to passes a subclass.
    """
    io = io or IO()
    owned = device is None  # a device handed in was started by whoever made it
    device = device or FullDuplexDevice(
        output_rate=settings.output_rate,
        headset=settings.headset,
        echo_tail_s=settings.echo_tail_s,
        input_device=settings.input_device,
        output_device=settings.output_device,
        canceller=None if settings.headset else make_canceller(settings.echo_canceller, io.warn),
    )
    io.player = device
    assistant = assistant_factory(config, io=io, extra_stop_phrases=settings.stop_phrases)
    try:
        loop = VoiceLoop(
            assistant,
            device,
            vad or make_vad(settings, io.warn),
            silence_ms=settings.silence_ms,
            merge_window_s=settings.merge_window_s,
            idle_s=settings.idle_s,
            language=settings.language,
            act=act,
            on_idle=on_idle,
            barge_min_voiced_ms=settings.barge_min_voiced_ms,
            barge_margin_db=settings.barge_margin_db,
            barge_min_erle_db=settings.barge_min_erle_db,
            confirm_tasks=settings.confirm_tasks,
            confirm_phrase=settings.confirm_phrase,
            confirm_timeout_s=settings.confirm_timeout_s,
        )
        if owned:
            device.start()
    except BaseException:
        assistant.close()  # closes the Speaker or, with none yet, the device through io.player
        raise
    return loop


def watch(loop: VoiceLoop, *, show=print, interval_s: float = 1.0, sleep=time.sleep, running=lambda: True) -> None:
    """Show `format_status(loop.status())` every `interval_s` while the loop runs and `running()` says so. For the live
    checks in docs/live-checks/voice.md: numbers and thresholds, never what was said. Ends when the loop's thread ends (`loop.ended`)."""
    while running() and not loop.ended:
        show(format_status(loop.status()))
        sleep(interval_s)
