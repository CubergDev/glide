"""The `glide` command, the one tree for everything Glide does.

    glide ask "what is the capital of France" [--speak]
    glide listen [--auto]          push-to-talk: Enter starts and stops, say "stop" or type /stop to cut in
    glide voice [--act]            hands-free: always listening, interruptible (needs the speech extra)
    glide chat [--speak]           a typed REPL: /pin /unpin /status /stop /act /help /quit
    glide doctor [--live]          what each provider slot can do (--live sends one tiny real request each),
                                   and how voice, memory, webhooks and mcp are set
    glide status                   the chains: slots, pins, resting slots, recent switches
    glide computer GOAL [--act]    drive the screen toward a goal (a dry run without --act); `glide-computer` too
    glide inspect [GOAL]           capture the screen and show what the classifier would be sent; `glide-inspect` too
    glide memory ...               local memory administration (off unless [memory] enabled = true)
    glide mcp ...                  serve Glide over MCP and show the MCP settings
    glide webhooks serve|work ...  the webhook listener and its worker (serve needs the webhooks extra)
    glide app-server [--socket PATH]  serve the SwiftUI app over a local Unix socket (never a network port)

`memory`, `mcp` and `webhooks` have their own options and help (`glide memory --help`); everything after the
command is theirs. Optional packages are imported only by the command that needs them, so `glide --help` and
the commands above that do not name an extra work with none installed.

Nothing here touches the machine unless `--act` is given (or `/act` typed in `chat`): a request to do
something on the computer is otherwise a dry run that looks at the screen and says what it would do. Every
provider switch is printed to stderr as it happens. Keys come from the environment named in glide.toml and
are never printed: text on its way to the terminal goes through the configuration's own `scrub`, and
through a sweep of every `*_KEY`, `*_TOKEN` and `*_SECRET` variable.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import queue
import sys
import threading
from collections.abc import Callable, Sequence
from pathlib import Path

from . import features
from .assistant.audio_io import AudioUnavailable, Endpointer, Microphone, Player
from .assistant.core import IO, Assistant, Reply
from .assistant.router import is_stop
from .assistant.tasks import DEFAULT_RUNS_DIR
from .computer.execution.reading import clean as printable

ACT_BANNER = (
    "ACT MODE: Glide will click and type on this Mac. Stop it by saying or typing stop, with Ctrl-C, "
    "or by moving the mouse into the top-left corner."
)
SECRET_SUFFIXES = ("KEY", "TOKEN", "SECRET")
MIN_SECRET = 8
REASON_CHARS = 160
FAILED_OUTCOMES = frozenset(  # a task that could not run
    {"provider failure", "generation unavailable", "desktop unavailable", "crashed", "not permitted", "not configured"}
)
POLL_S = 0.1  # how often the terminal loop looks up from waiting for a line
QUIT_WORDS = frozenset({"q", "quit", "exit", "/quit", "/exit"})


# -- Seams: everything that reaches outside the process goes through one of these, so tests replace it ----


def _dotenv() -> None:
    from .computer.config import load_dotenv

    load_dotenv(Path.cwd() / ".env")


def _load(path: str | None):
    """The configuration, with every provider switch printed to stderr as it happens.

    The listener is registered before any chain exists, so not even the first call can switch silently.
    """
    from .providers.config import load_config

    _dotenv()
    config = load_config(path)
    config.on_switch(lambda event: print(format_switch(event, config), file=sys.stderr, flush=True))
    return config


def _player(on_error: Callable[[BaseException], None]) -> Player:
    return Player(on_error=on_error)  # raises AudioUnavailable when `sounddevice` is not installed


def _microphone() -> Microphone:
    return Microphone()  # raises AudioUnavailable when `sounddevice` is not installed


def _doctor(config, live: bool) -> int:
    from .providers import doctor

    print(f"config: {config.source}")
    if config.defaulted:
        print(f"built-in chains in use for: {', '.join(config.defaulted)}")
    for warning in config.warnings:
        print(f"warning: {clean(warning, config)}")
    if live:
        print("live: one tiny request to each slot that has a key")
    rows = doctor.doctor(config, live=live)
    print(doctor.format_rows(rows))
    print("\nfeatures (nothing is started or downloaded to find this out)")
    report = features.feature_report(config)
    for name, _, line in report:
        print(f"  {name:<9}{clean(line, config)}")
    return 1 if doctor.failed(rows) or not all(ok for _, ok, _ in report) else 0


def _read_line(prompt: str = "") -> str:
    return input(prompt)


# -- Output -----------------------------------------------------------------------------------------


def redact(text: str) -> str:
    """`text` with the value of every secret-looking environment variable replaced by '***'."""
    for name, value in os.environ.items():
        if len(value) >= MIN_SECRET and name.upper().endswith(SECRET_SUFFIXES):
            text = text.replace(value, "***")
    return text


def clean(text: str, config=None) -> str:
    """Text safe to print: the configuration's own scrub first (it knows which variables hold keys), then the sweep,
    then every terminal escape sequence and control character removed (newlines and tabs stay): what an app, a page or
    a model wrote must never move the cursor, retitle the window or write the clipboard of this terminal."""
    scrub = getattr(config, "scrub", None)
    return printable(redact(scrub(text) if callable(scrub) else text), lines=True)


def format_switch(event, config=None) -> str:
    """One line for a provider switch: which job, from which slot to which, and why."""
    reason = " ".join(clean(str(event.reason), config).split())
    if len(reason) > REASON_CHARS:
        reason = reason[: REASON_CHARS - 3] + "..."
    if event.kind == "slow":
        return f"racing: {event.role} {event.from_slot} is slow ({reason}), also trying {event.to_slot or 'nothing else'}"
    target = event.to_slot or "nothing left"
    return f"fallback: {event.role} {event.from_slot} -> {target} ({event.kind}: {reason})"


def _warn(message: str, config=None) -> None:
    print(clean(message, config), file=sys.stderr, flush=True)


def _make_io(config, *, speak: bool, partial: Callable[[str], None] | None = None) -> IO:
    """The terminal's IO: text to stdout, notices to stderr, and a speaker when `speak` and one can be had."""
    player = None
    if speak:
        try:
            player = _player(lambda exc: _warn(f"playback failed: {exc}", config))
        except AudioUnavailable as exc:
            _warn(f"speech is off: {exc}", config)
    return IO(
        player=player,
        show=lambda text: print(clean(text, config), flush=True),
        partial=partial or (lambda text: None),
        heard=lambda text: print(f"you said: {clean(text, config)}", flush=True),
        warn=lambda message: _warn(message, config),
    )


def _timings(reply: Reply) -> str:
    return "timings: " + ", ".join(f"{name}={value:.2f}" for name, value in reply.timings.items())


# -- Commands ---------------------------------------------------------------------------------------


def cmd_ask(args: argparse.Namespace, config) -> int:
    assistant = Assistant(config, io=_make_io(config, speak=args.speak), runs_dir=args.runs)
    if args.act:
        print(ACT_BANNER, file=sys.stderr)
    try:
        reply = assistant.handle_text(" ".join(args.text), act=args.act, wait=True)
        assistant.wait_idle()
    except KeyboardInterrupt:
        assistant.stop()
        print("stopped", file=sys.stderr)
        return 130
    finally:
        assistant.close()
    if args.timings:
        print(_timings(reply), file=sys.stderr)
    result = reply.task.result if reply.task is not None else None
    return 1 if reply.error or (result is not None and result.outcome in FAILED_OUTCOMES) else 0


class LineReader:
    """Lines from the terminal on a thread of their own, so Enter and /stop are heard while Glide is busy.

    After the end of input every `get` returns None, so whoever reads the end first cannot hide it from the rest.
    """

    def __init__(self, read: Callable[[str], str]) -> None:
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._eof = False
        threading.Thread(target=self._run, args=(read,), name="glide-stdin", daemon=True).start()

    def _run(self, read: Callable[[str], str]) -> None:
        while True:
            try:
                self._lines.put(read(""))
            except Exception:  # EOFError, a closed stdin, or a terminal gone away: all are the end of input
                self._lines.put(None)
                return

    def get(self, timeout: float | None = None) -> str | None:
        """The next line, None at the end of input. Raises queue.Empty when `timeout` ran out."""
        if self._eof:
            return None
        line = self._lines.get(timeout=timeout)
        self._eof = line is None
        return line


def _spawn(work: Callable[[], object], config) -> threading.Thread:
    """Run one request on a worker thread, so the terminal stays free for Enter, /stop and Ctrl-C while it is handled."""

    def target() -> None:
        try:
            work()
        except Exception as exc:  # a bug must not leave the terminal waiting for an answer that will never come
            _warn(f"{type(exc).__name__}: {exc}", config)

    thread = threading.Thread(target=target, name="glide-turn", daemon=True)
    thread.start()
    return thread


def _idle(turn: threading.Thread | None) -> bool:
    return turn is None or not turn.is_alive()


def _prompt(text: str, state: dict, turn: threading.Thread | None) -> None:
    """Print the prompt once each time Glide is idle again, so it never lands in the middle of an answer."""
    if _idle(turn) and not state["prompted"]:
        print(text, end="", flush=True)
        state["prompted"] = True


def cmd_chat(args: argparse.Namespace, config) -> int:
    assistant = Assistant(config, io=_make_io(config, speak=args.speak), runs_dir=args.runs)
    act = args.act
    lines = LineReader(_read_line)
    print("glide chat. Type a request, /help for commands, /quit to leave.")
    if act:
        print(ACT_BANNER, file=sys.stderr)
    turn: threading.Thread | None = None
    state = {"prompted": False}
    interrupted = False
    try:
        while True:
            try:
                _prompt("you> ", state, turn)
                try:
                    line = lines.get(timeout=POLL_S)
                except queue.Empty:
                    continue
                interrupted = False
                state["prompted"] = False
                if line is None:
                    break
                line = line.strip()
                if not line:
                    continue
                if line.startswith("/"):
                    verdict = _slash(line, assistant, config)
                    if verdict == "quit":
                        break
                    if verdict == "act":
                        act = not act
                        print(ACT_BANNER if act else "act mode off: tasks are dry runs again")
                    continue
                if is_stop(line):  # heard here, not on a worker: stopping must not wait for a thread to start
                    print("stopped" if assistant.stop() else "stopped (nothing was running)")
                    continue
                assistant.interrupt_speech()  # a new request replaces the last answer, spoken or still being written
                turn = _spawn(lambda text=line, act=act: _chat_turn(assistant, text, act, args, config), config)
            except KeyboardInterrupt:
                assistant.stop()
                if interrupted:
                    break
                interrupted = True
                print("\nstopped (Ctrl-C again to leave)")
        _finish(assistant, turn)
    except KeyboardInterrupt:
        assistant.stop()
    finally:
        assistant.close()
    return 0


def _chat_turn(assistant: Assistant, text: str, act: bool, args: argparse.Namespace, config) -> None:
    reply = assistant.handle_text(text, act=act, wait=False)
    if args.timings:
        print(clean(_timings(reply), config), file=sys.stderr)


def _finish(assistant: Assistant, turn: threading.Thread | None = None) -> None:
    """On the way out: let the request being handled finish, then a running task (Ctrl-C stops it), then the voice."""
    if turn is not None:
        turn.join()
    task = assistant.task
    if task is not None and task.running:
        print("waiting for the running task to end (Ctrl-C stops it)")
        task.wait()
    assistant.wait_idle()


def _slash(line: str, assistant: Assistant, config) -> str | None:
    """One chat command. Returns "quit", "act" (toggle act mode) or None."""
    word, *rest = line.split()
    word = word.lower()
    if word in QUIT_WORDS:
        return "quit"
    if word == "/act":
        return "act"
    if word == "/stop":
        print("stopped" if assistant.stop() else "stopped (nothing was running)")
    elif word == "/status":
        print_status(config)
    elif word == "/pin":
        if len(rest) not in (2, 3) or (len(rest) == 3 and rest[2] != "strict"):
            print("usage: /pin <role> <slot or prefix> [strict]   roles: llm.fast llm.smart stt tts classifier")
        else:
            _pin(config, rest[0], rest[1], strict=len(rest) == 3)
    elif word == "/unpin":
        if len(rest) != 1:
            print("usage: /unpin <role>")
        else:
            try:
                config.unpin(rest[0])
                print(f"{rest[0]} is no longer pinned")
            except ValueError as exc:
                print(clean(str(exc), config))
    else:
        print(
            "/stop  cut speech and stop the task   /act  toggle act mode (default off: dry runs)   /status  provider chains\n"
            "/pin <role> <slot> [strict]  prefer one provider   /unpin <role>   /quit"
        )
    return None


def _pin(config, role: str, slot: str, *, strict: bool) -> None:
    try:
        name = config.pin(role, slot, strict)
    except ValueError as exc:  # ConfigError: it says which slots there are
        print(clean(str(exc), config))
        return
    print(f"{role} pinned to {name}" + (" (strict: nothing else is tried)" if strict else ""))


def cmd_listen(args: argparse.Namespace, config) -> int:
    try:
        mic = _microphone()
    except AudioUnavailable as exc:
        print(f"glide listen: {exc}", file=sys.stderr)
        return 2

    def partial(text: str) -> None:
        if sys.stderr.isatty():
            print(f"\r{clean(text, config)}", end="", file=sys.stderr, flush=True)

    assistant = Assistant(config, io=_make_io(config, speak=not args.text_only, partial=partial), runs_dir=args.runs)
    act = args.act
    lines = LineReader(_read_line)
    print("Press Enter to talk and Enter again to finish. Type a request to send text. /stop stops; q quits.")
    if act:
        print(ACT_BANNER, file=sys.stderr)
    turn: threading.Thread | None = None
    state = {"prompted": False}
    interrupted = False
    try:
        while True:
            try:
                _prompt("listen> ", state, turn)
                try:
                    line = lines.get(timeout=POLL_S)
                except queue.Empty:
                    continue
                interrupted = False
                state["prompted"] = False
                if line is None:
                    break
                typed = line.strip()
                if typed.lower() in QUIT_WORDS:
                    break
                if typed.lower() == "/stop" or is_stop(typed):
                    print("stopped" if assistant.stop() else "stopped (nothing was running)")
                elif typed:
                    assistant.interrupt_speech()
                    turn = _spawn(lambda text=typed, act=act: assistant.handle_text(text, act=act, wait=False), config)
                else:
                    turn = _record(assistant, mic, lines, args, act, config)
            except KeyboardInterrupt:
                assistant.stop()
                if interrupted:
                    break
                interrupted = True
                print("\nstopped (Ctrl-C again to leave)")
        _finish(assistant, turn)
    except KeyboardInterrupt:
        assistant.stop()
    finally:
        assistant.close()
    return 0


def _record(
    assistant: Assistant, mic: Microphone, lines: LineReader, args: argparse.Namespace, act: bool, config
) -> threading.Thread:
    """One push-to-talk utterance. Enter was pressed: cut the voice, listen until Enter or silence, and hand the
    speech to a worker thread that transcribes it, answers it and speaks. Returns that thread.

    The worker starts at once, so the transcript is being made while the person is still speaking; this thread
    only waits for the next Enter (or the end of the audio) and then goes back to the prompt, free to hear
    Enter again, `/stop` or Ctrl-C while the answer is still being written or spoken.
    """
    assistant.interrupt_speech()  # barge-in: speaking over Glide silences it at once
    stop, recorded = threading.Event(), threading.Event()
    endpointer = Endpointer() if args.auto else None

    def chunks():
        try:
            yield from mic.record(stop, endpointer=endpointer)
        finally:
            recorded.set()

    def work() -> None:
        reply = assistant.handle_audio(chunks(), act=act, wait=False, language=args.lang)
        if reply.route == "none" and not reply.error and not reply.heard:  # something heard and then cut is not silence
            print("(nothing heard)", flush=True)

    turn = _spawn(work, config)
    print("listening... (Enter to finish)", flush=True)
    while not recorded.is_set() and turn.is_alive():
        try:
            lines.get(timeout=POLL_S)  # any line, or the end of input, ends the recording
        except queue.Empty:
            continue
        stop.set()
    stop.set()
    print(file=sys.stderr)
    return turn


def _voice_loop(config, io: IO, act: bool):
    """The voice loop over the real sound device, not yet started (`build_voice` sets `io.player` to that device)."""
    from .speech.session import build_voice

    return build_voice(config, config.voice, io=io, act=act)


def cmd_voice(args: argparse.Namespace, config) -> int:
    from .speech.audio import DeviceFault
    from .speech.vad import VadError

    io = _make_io(config, speak=False)  # the device is the speaker: a second one would fight it for the output
    try:
        loop = _voice_loop(config, io, args.act)
    except (AudioUnavailable, VadError, DeviceFault) as exc:
        print(
            f"glide voice: {clean(str(exc), config)} (the speech extra has the audio and voice packages: uv sync --extra speech)",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:  # a sound card that will not open: sounddevice's PortAudioError is none of the kinds above
        print(
            f"glide voice: the audio device could not be opened ({type(exc).__name__}: {clean(str(exc), config)})",
            file=sys.stderr,
        )
        return 2
    print('glide voice: listening. Just talk; say "stop" to stop a task, press Ctrl-C to leave.')
    if args.act:
        print(ACT_BANNER, file=sys.stderr)
    try:
        loop.run()
    except KeyboardInterrupt:
        loop.assistant.stop()
    finally:
        loop.stop()
        loop.assistant.close()
    return 1 if loop.failure else 0


def cmd_doctor(args: argparse.Namespace, config) -> int:
    return _doctor(config, args.live)


# -- Commands that are other modules' own: everything after the command is theirs ---------------------------


def _forward(args: argparse.Namespace) -> list[str]:
    """The words after the command, led by the glide.toml the person named (their own `--config` is glide.toml too)."""
    return ["--config", args.config, *args.rest] if args.config else list(args.rest)


def _help_asked(words: Sequence[str]) -> bool:
    return any(word in ("-h", "--help") for word in words)


def cmd_memory(args: argparse.Namespace) -> int:
    from .memory import cli as memory

    return memory.main(_forward(args))


def cmd_mcp(args: argparse.Namespace) -> int:
    from .mcp import cli as mcp

    return mcp.main(_forward(args))


def cmd_webhooks_serve(args: argparse.Namespace) -> int:
    """`glide webhooks serve`: its `--config` is the webhook JSON file, so the glide.toml only says where that is by default."""
    from .memory.settings import SettingsError, find_config

    words = list(args.rest)
    if not _help_asked(words):
        lacking = features.missing(features.WEBHOOKS_MODULES)
        if lacking:
            print(f"glide: {features.extra_message('glide webhooks serve', 'webhooks', lacking)}", file=sys.stderr)
            return 2
        if not any(word == "--config" or word.startswith("--config=") for word in words):
            try:
                words = ["--config", str(features.webhooks_file(os.environ, find_config(os.environ, path=args.config))), *words]
            except SettingsError as exc:
                print(f"glide webhooks: {exc}", file=sys.stderr)
                return 2
    from .webhooks import cli as webhooks

    return webhooks.main(words)


def cmd_webhooks_work(args: argparse.Namespace) -> int:
    from .webhooks import worker

    return worker.main(_forward(args))


def cmd_computer(args: argparse.Namespace, config=None) -> int:
    from .computer import cli as computer

    return computer.main(args.rest, config)


def cmd_app_server(args: argparse.Namespace, config=None) -> int:
    from .app_server import cli as app_server

    return app_server.main(args.rest, config)


def cmd_inspect(args: argparse.Namespace) -> int:
    from .computer import cli as computer

    _dotenv()
    return computer.inspect(args.rest)


def cmd_status(args: argparse.Namespace, config) -> int:
    print_status(config)
    print(
        "Switches are remembered per process, so a new `glide status` starts with none; /status inside `glide chat` shows them."
    )
    return 0


def print_status(config) -> None:
    """Each chain: its slots (pinned, resting, calls, failures, latency, last error), and its recent switches."""
    from .providers.config import ROLES, ConfigError

    print(f"config: {config.source}")
    for role in ROLES:
        print(role)
        try:
            chain = config.chain(role)
        except ConfigError as exc:
            print(f"  unavailable: {clean(str(exc), config)}")
            continue
        for row in chain.status():
            marks = ["pinned"] if row["pinned"] else []
            if row["resting_s"]:
                marks.append(f"resting {row['resting_s']}s")
            latency = "-" if row["avg_latency_s"] is None else f"{row['avg_latency_s']:.2f}s"
            line = f"  {row['name']}  calls {row['calls']}  failures {row['failures']}  avg {latency}"
            print(line + (f"  [{', '.join(marks)}]" if marks else ""))
            if row["last_error"]:
                print(f"      last error: {clean(row['last_error'], config)[:REASON_CHARS]}")
        slots = getattr(config, "slots", None)
        for info in slots(role) if callable(slots) else []:
            if info.state == "skipped":
                print(f"  {info.name}  skipped: {clean(info.reason, config)}")
        for event in list(chain.events)[-5:]:
            print(f"  {format_switch(event, config)}")


# -- Entry ------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="glide",
        description="A voice and text assistant that can also drive your Mac. Without --act it never clicks or types.",
    )
    parser.add_argument(
        "--config", help="the glide.toml to read (default: $GLIDE_CONFIG, ./glide.toml, ~/.config/glide/glide.toml)"
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    work = argparse.ArgumentParser(add_help=False)
    work.add_argument(
        "--act", action="store_true", help="really click and type on this Mac (default: a dry run that says what it would do)"
    )
    work.add_argument("--runs", type=Path, default=DEFAULT_RUNS_DIR, help="where a computer task writes its run folder")
    work.add_argument(
        "--timings", action="store_true", help="print how long the route, first word and first audio took, to stderr"
    )

    ask = commands.add_parser("ask", parents=[work], help="answer one request, or do it on the computer")
    ask.add_argument("text", nargs="+", help="the request")
    ask.add_argument("--speak", action="store_true", help="say the answer aloud as well as printing it")
    ask.set_defaults(handler=cmd_ask)

    listen = commands.add_parser("listen", parents=[work], help="push-to-talk voice loop")
    listen.add_argument("--auto", action="store_true", help="also end a recording when the speaker goes quiet")
    listen.add_argument("--text-only", action="store_true", help="print answers instead of speaking them")
    listen.add_argument("--lang", help="language of the speech (default: detected)")
    listen.set_defaults(handler=cmd_listen)

    chat = commands.add_parser("chat", parents=[work], help="a typed conversation; /help lists its commands")
    chat.add_argument("--speak", action="store_true", help="say answers aloud as well as printing them")
    chat.set_defaults(handler=cmd_chat)

    doctor = commands.add_parser("doctor", help="what each provider slot can do right now")
    doctor.add_argument("--live", action="store_true", help="send each slot one tiny real request (spends a few tokens each)")
    doctor.set_defaults(handler=cmd_doctor)

    status = commands.add_parser("status", help="the provider chains, pins, resting slots and recent switches")
    status.set_defaults(handler=cmd_status)

    voice = commands.add_parser("voice", help="hands-free voice loop: always listening, interruptible (speech extra)")
    voice.add_argument("--act", action="store_true", help="really click and type on this Mac (default: tasks are dry runs)")
    voice.set_defaults(handler=cmd_voice)

    # The rest take their own options. Their parsers live in their own modules and are imported only when the command
    # runs, so nothing is parsed here: `parse_known_args` hands the words after the command to `main` as `args.rest`.
    def passthrough(parent, name: str, handler, help: str, *, config: bool = False, loads: bool = False):
        command = parent.add_parser(name, add_help=False, help=help)
        if config:  # `glide memory --config x ...` and `glide --config x memory ...` mean the same
            command.add_argument("--config", default=argparse.SUPPRESS)
        command.set_defaults(handler=handler, passthrough=True, loads_config=loads)
        return command

    passthrough(
        commands, "computer", cmd_computer, "drive the screen toward a goal (`glide computer --help`)", config=True, loads=True
    )
    passthrough(
        commands,
        "app-server",
        cmd_app_server,
        "serve the SwiftUI app over a local socket (`glide app-server --help`)",
        config=True,
        loads=True,
    )
    passthrough(commands, "inspect", cmd_inspect, "capture the screen and show what the classifier would be sent")
    passthrough(commands, "memory", cmd_memory, "local memory administration (`glide memory --help`)", config=True)
    passthrough(commands, "mcp", cmd_mcp, "serve Glide over MCP, show the MCP settings (`glide mcp --help`)", config=True)
    webhooks = commands.add_parser("webhooks", help="the webhook listener and its worker (`glide webhooks serve --help`)")
    parts = webhooks.add_subparsers(dest="webhooks_command", required=True, metavar="serve|work")
    passthrough(parts, "serve", cmd_webhooks_serve, "receive authenticated webhooks and queue agent requests (webhooks extra)")
    passthrough(parts, "work", cmd_webhooks_work, "consume queued requests, one at a time", config=True)
    return parser


def computer_main(argv: Sequence[str] | None = None) -> int:
    """The `glide-computer` command: `glide computer`, under its older name."""
    return main(["computer", *(sys.argv[1:] if argv is None else argv)])


def inspect_main(argv: Sequence[str] | None = None) -> int:
    """The `glide-inspect` command: `glide inspect`, under its older name."""
    return main(["inspect", *(sys.argv[1:] if argv is None else argv)])


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, rest = parser.parse_known_args(argv)
    if rest and not getattr(args, "passthrough", False):
        parser.error(f"unrecognized arguments: {' '.join(rest)}")
    args.rest = rest
    if getattr(args, "passthrough", False) and (not args.loads_config or _help_asked(rest)):
        if not _help_asked(rest):
            _dotenv()  # the documented .env holds keys and settings these commands read from os.environ themselves
        return args.handler(args)  # theirs to load, if they load anything; asking for help needs no configuration
    try:
        config = _load(args.config)
    except (ValueError, OSError) as exc:  # ConfigError is a ValueError: the file is wrong, or missing
        print(f"glide: {clean(str(exc))}", file=sys.stderr)
        return 2
    try:
        return args.handler(args, config)
    finally:
        with contextlib.suppress(Exception):
            config.close()


if __name__ == "__main__":
    raise SystemExit(main())
