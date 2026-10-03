"""`glide computer` and `glide inspect`: the screen-driving loop, and a look at what it would send.

Both are reached through the one `glide` command (glide/cli.py), which loads glide.toml, prints every provider
switch, and hands the loaded configuration to `main`. `glide-computer` and `glide-inspect` are the same two
commands under their older names. Nothing here starts a run, opens a window or captures the screen until the
person has typed the command.
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

from . import config
from .actions import Context
from .control import RunControl
from .perception import capture, perceive
from .platform_adapter import desktop
from .report import annotate, ax_count, render_payload
from .runner import RunConfig, run
from .timing import format_timing
from .writer import make_writer, provider

WRITER_DISABLED = (
    "writer disabled: no usable llm provider in glide.toml; type_text, writer-proposed URLs and the final answer need one "
    "(run `glide doctor`)"
)
ABORTED, FAILED = (
    130,
    1,
)  # exit codes: stopped by the user (the shell's own value for Ctrl-C), and any run that did not do the job
FAILED_OUTCOMES = {"blocked", "unsupported", "crashed", "provider failure"}  # failures that may carry no `failure` text


def _fail(message: str, code: int = 2) -> int:
    print(f"glide computer: {message}", file=sys.stderr)
    return code


def ask_user(question: str) -> str:
    """Read the user's reply to a question the run has just printed. An empty reply declines to answer.

    The bell is for a user who is watching the browser, not this terminal.
    """
    try:
        return input("\a  > ")
    except EOFError:
        return ""


def main(argv: list[str] | None, glide_config) -> int:
    """Drive this computer toward a goal. `glide_config` is the loaded `GlideConfig`: its writer and classifier chains
    do the thinking, and every fallback between their slots is announced by the listener `glide` registered on it."""
    from ..providers.config import ConfigError

    parser = argparse.ArgumentParser(
        prog="glide computer", description="Drive this computer toward a goal: screen OCR, a classifier, deterministic actions."
    )
    parser.add_argument("goal", help="what you want done on this computer")
    parser.add_argument("--act", action="store_true", help="actually click and type (default: dry run, one step)")
    parser.add_argument(
        "--engine",
        choices=["legacy", "structured"],
        default="legacy",
        help="legacy: the screen loop. structured: planned effects checked after each action, through the "
        "provider chains of glide.toml (browser and desktop tasks; research and reasoning)",
    )
    parser.add_argument(
        "--readiness-timeout",
        type=float,
        default=config.DEFAULT_READINESS_TIMEOUT,
        help="structured engine: seconds a page or an effect may take to show before the run reports it (0-30)",
    )
    parser.add_argument("--steps", type=int, default=config.DEFAULT_STEPS, help="max actions before stopping")
    parser.add_argument("--min-confidence", type=float, default=config.DEFAULT_MIN_CONFIDENCE, help="stop below this confidence")
    parser.add_argument("--delay", type=float, default=config.DEFAULT_DELAY, help="seconds to wait after each action")
    parser.add_argument(
        "--handoffs",
        type=int,
        default=config.DEFAULT_HANDOFFS,
        help="times the writer may send a stopped run back to the classifier with a new focus (0: every stop is final)",
    )
    parser.add_argument("--out", type=Path, default=Path("runs") / time.strftime("%Y%m%d-%H%M%S"), help="run folder")
    parser.add_argument("--image", type=Path, help="replay a saved capture instead of the live screen (never acts)")
    parser.add_argument(
        "--record-content",
        action="store_true",
        help="also save the goal, the answer, history, screenshots and step files in the run folder (off by default)",
    )
    parser.add_argument("--app", help="frontmost app to report during replay")
    parser.add_argument("--url", help="browser URL to report during replay")
    args = parser.parse_args(argv)

    if args.act and not desktop.accessibility_trusted():
        return _fail("this terminal lacks Accessibility permission; grant it in System Settings > Privacy & Security")
    try:
        writer = make_writer(glide_config)
        config.writer_vision()  # a bad value stops the run here, not at its first stop
    except ValueError as e:  # a ConfigError is one
        return _fail(str(e))
    if writer is None:
        print(WRITER_DISABLED)
    else:
        print(f"writer: {provider(writer)}")

    cfg = RunConfig(
        goal=args.goal,
        out=args.out,
        act=args.act,
        steps=args.steps,
        min_confidence=args.min_confidence,
        delay=args.delay,
        handoffs=args.handoffs,
        image=args.image,
        app=args.app,
        url=args.url,
        record_content=args.record_content,
        engine=args.engine,
        execution_browser=config.browser(),
        readiness_timeout=args.readiness_timeout,
    )

    def ctx_factory(classifier, history):
        return Context(
            goal=args.goal,
            browser=config.browser(),
            email=config.email(),
            typesafe=classifier,
            writer=writer,
            history=history,
            ask=ask_user if sys.stdin.isatty() else None,
        )

    # The run prints nothing of its own; this terminal shows what it reports, and Ctrl-C reaches it as a stop.
    control = RunControl(str(uuid.uuid4()), lambda event: print(event.text) if event.text else None)
    try:
        state = run(cfg, ctx_factory, classifier_factory=glide_config.classifier, control=control)
    except ConfigError as e:  # no classifier slot is usable: the message names the variables to set
        return _fail(glide_config.scrub(str(e)))
    if state.outcome.startswith("aborted"):
        return ABORTED
    if state.failure or state.outcome in FAILED_OUTCOMES:
        return FAILED
    return 0


def inspect(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="glide inspect",
        description="Count down, capture the screen, and show exactly what the classifier would be sent.",
    )
    parser.add_argument("goal", nargs="?", default="(no goal given)")
    parser.add_argument("--countdown", type=int, default=3)
    parser.add_argument("--no-open", action="store_true", help="write files without opening them")
    parser.add_argument("--out", type=Path, default=Path("inspections") / time.strftime("%Y%m%d-%H%M%S"))
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    for n in range(args.countdown, 0, -1):
        print(f"{n}...", end=" ", flush=True)
        time.sleep(1)
    print("capture")

    browser = config.browser()
    timing: dict[str, float] = {}
    screen = capture(browser=browser, timing=timing)
    items = perceive(screen, config.MAX_OPTIONS, args.goal, timing)
    annotated = args.out / "annotated.png"
    text = args.out / "state.txt"
    screen.image.save(args.out / "raw.png")
    annotate(screen, items, chosen="", out=annotated)
    text.write_text(render_payload(args.goal, screen, items, [], browser, config.email()), encoding="utf-8")

    print(
        f"app={screen.app!r} url={screen.url!r} items={len(items)} ax={ax_count(items)} "
        f"offscreen={len(screen.offscreen)} field={screen.field.role if screen.field else None}"
    )
    print(format_timing(timing))
    print(f"  {annotated}\n  {text}")
    if not args.no_open:
        desktop.open_path(annotated)
        desktop.open_path(text, as_text=True)
    return 0
