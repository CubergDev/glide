"""Point at a control and ask about it from a terminal, without entering the action loop. Never clicks or types.

    python -m glide.assistant.point_cli                      a local preview: the item's accessibility text, nothing sent
    python -m glide.assistant.point_cli "what is this?" --allow-model [--with-image] [--at X Y] [--delay 3]
    python -m glide.assistant.point_cli --voice --allow-model [--headset]     spoken follow-ups about one pin

Without `--allow-model` nothing leaves this machine: the item under the pointer is read and shown. With it, the question,
the item's text and, with `--with-image`, a small crop go to the provider chains of glide.toml; the slots are printed
before anything is read, and every fallback is printed as it happens. The answer is on stdout, everything else on stderr.
Exit codes: 0 done, 2 nothing usable to read or bad options, 3 the provider failed, 130 cancelled with Ctrl-C.
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

from ..computer.config import load_dotenv, writer_vision
from ..computer.writer import make_writer, provider
from ..providers.config import ConfigError, load_config
from ..speech.settings import MAX_SILENCE_MS, MIN_SILENCE_MS
from .core import IO
from .point_ask import MAX_QUESTION, PointStopped, PointUnavailable, capture_point
from .point_session import PointSession
from .point_voice import PointAssistant

DEFAULT_QUESTION = "What is this, and what should I do next?"
REASON_CHARS = 160


def _bounded(low: float, high: float):
    def parse(raw: str) -> float:
        value = float(raw)
        if not math.isfinite(value) or not low <= value <= high:
            raise argparse.ArgumentTypeError(f"must be between {low:g} and {high:g}")
        return value

    return parse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="point_cli", description="Point at something on screen and ask what it means. Never clicks or types."
    )
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--delay", type=_bounded(0, 30), default=3, help="seconds to position the pointer (default: 3)")
    parser.add_argument("--at", nargs=2, type=float, metavar=("X", "Y"), help="use a fixed screen point instead of the pointer")
    parser.add_argument("--with-image", action="store_true", help="also read a small crop, including nearby visible content")
    parser.add_argument("--radius", type=_bounded(24, 240), default=160, help="crop radius in screen coordinates (default: 160)")
    parser.add_argument(
        "--allow-model", action="store_true", help="send the question and the selected context to the provider chains"
    )
    parser.add_argument("--voice", action="store_true", help="ask spoken follow-up questions about this one pin")
    parser.add_argument("--headset", action="store_true", help="allow spoken interruption during playback; use headphones")
    parser.add_argument(
        "--silence-ms", type=int, default=None, help="how long a pause ends a spoken question (default: [speech])"
    )
    parser.add_argument("--config", default=None, help="the glide.toml to use")
    return parser


def _say(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def _switch_line(event, config) -> str:
    reason = " ".join(config.scrub(str(event.reason)).split())
    if len(reason) > REASON_CHARS:
        reason = reason[: REASON_CHARS - 3] + "..."
    return f"fallback: {event.role} {event.from_slot} -> {event.to_slot or 'nothing left'} ({event.kind}: {reason})"


def _countdown(seconds: float) -> None:
    """Time to position the pointer. Ctrl-C ends it (a KeyboardInterrupt), and nothing has been read yet."""
    threading.Event().wait(seconds)


def _load(path: str | None):
    """The configuration, with every provider switch printed to stderr as it happens."""
    load_dotenv(Path.cwd() / ".env")
    config = load_config(path)
    config.on_switch(lambda event: _say(_switch_line(event, config)))
    return config


def main(argv: Sequence[str] | None = None, *, voice_factory: Callable | None = None, load=_load, capture=capture_point) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.question.strip() or len(args.question) > MAX_QUESTION:
        parser.error(f"question must contain 1-{MAX_QUESTION} characters")
    if args.at and not all(math.isfinite(v) for v in args.at):
        parser.error("point coordinates must be finite")
    if (args.with_image or args.voice) and not args.allow_model:
        parser.error("--with-image and --voice need --allow-model: they share the question and the selected context")
    if args.silence_ms is not None and not MIN_SILENCE_MS <= args.silence_ms <= MAX_SILENCE_MS:
        parser.error(f"--silence-ms must be between {MIN_SILENCE_MS} and {MAX_SILENCE_MS}")
    try:
        config = load(args.config)
    except (ValueError, OSError) as error:
        _say(f"could not load the configuration: {error}")
        return 2
    try:
        return _run(args, config, voice_factory, capture)
    except KeyboardInterrupt:
        _say("Point-to-ask cancelled.")
        return 130
    finally:
        close = getattr(config, "close", None)
        if close is not None:
            close()


def _writer(config):
    """The answer writer over the chains, or None; a slot of glide.toml that cannot be set up says why, without a key."""
    try:
        return make_writer(config)
    except ConfigError as error:
        raise PointUnavailable(" ".join(config.scrub(str(error)).split())) from error


def _run(args, config, voice_factory, capture) -> int:
    writer = None
    try:
        if args.allow_model:
            writer = _writer(config)
            if writer is None:
                raise PointUnavailable(
                    "Configure an answer provider first: the smart LLM chain in glide.toml has no usable slot."
                )
            if args.with_image and not writer_vision():
                raise PointUnavailable("The configured answer model is text-only; omit --with-image.")
            _say(f"Answer provider: {provider(writer)}")
            _say(
                "This request shares your question, the pointed item's text"
                + (" and a small image crop." if args.with_image else ".")
            )
        else:
            _say("Local preview: reads the pointed item's accessibility text. Nothing is sent to a model.")
        _say(f"Position the pointer. Reading in {args.delay:g} seconds; Ctrl-C cancels.")
        _countdown(args.delay)
        selection = capture(tuple(args.at) if args.at else None, with_image=args.with_image, radius=args.radius)
    except (PointUnavailable, PointStopped) as error:  # our own sentences
        _say(str(error))
        return 2
    except Exception:  # the adapters' errors can name the screen
        _say("Could not read this point. Check the desktop permissions.")
        return 2
    with selection:
        if writer is None:
            packet = selection.target.packet() if selection.target is not None else {}
            print("\n".join(f"{name}: {value}" for name, value in packet.items() if value))
            return 0
        return _ask(args, config, writer, selection, voice_factory)


def _ask(args, config, writer, selection, voice_factory) -> int:
    finished = threading.Event()
    outcome = {"code": 0}

    def emit(kind: str, **data) -> None:
        if kind == "answer":
            print(data["text"], flush=True)
            if data.get("uncertain"):
                _say("The selected context is incomplete. Point again or give more detail.")
            if not args.voice:
                finished.set()
        elif kind == "error":
            _say(data["text"])
            outcome["code"] = 3 if not data.get("closed") else 2
            if data.get("closed") or not args.voice:
                finished.set()
        elif kind == "stopped":
            finished.set()

    if args.voice:
        return _voice(args, config, writer, selection, voice_factory, emit, finished, outcome)
    session = PointSession(selection, writer, emit)
    try:
        session.ask(args.question)
        finished.wait()
    finally:
        session.close()
    return outcome["code"]


def _voice(args, config, writer, selection, voice_factory, emit, finished, outcome) -> int:
    if voice_factory is None:
        from ..speech.session import build_voice as voice_factory
    settings = config.speech
    settings = replace(settings, headset=args.headset or settings.headset)
    if args.silence_ms is not None:
        settings = replace(settings, silence_ms=args.silence_ms)
    loop = session = None
    try:
        io = IO(heard=lambda text: _say(f"you said: {config.scrub(text)}"), warn=lambda message: _say(config.scrub(message)))
        loop = voice_factory(
            config, settings, io=io, act=False, assistant_factory=lambda cfg, io=None: PointAssistant(cfg, io=io)
        )
        session = PointSession(selection, writer, emit, speak=loop.assistant.say, cancel_speech=loop.assistant.cut_voice)
        loop.assistant.bind(session)
        _say("Ask about this point, then ask follow-up questions. Say Stop to interrupt an answer. Ctrl-C ends.")
        loop.start()
        finished.wait()
    except Exception as error:  # a microphone that cannot open: what is missing, never a key
        _say(f"Voice input could not start ({type(error).__name__}): {config.scrub(str(error))}")
        outcome["code"] = 2
    finally:
        if session is not None:
            session.close()
        if loop is not None:
            loop.stop()
            loop.assistant.close()
    return outcome["code"]


if __name__ == "__main__":
    raise SystemExit(main())
