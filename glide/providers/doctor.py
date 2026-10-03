"""`glide doctor`: what each configured slot can do right now, one row per slot.

Offline (the default) it only checks what needs no network: that the file parses, that each slot's key
variable is set (the value is never read out, only whether it is there), and that its adapter builds. Live it
also sends every slot ONE tiny real request, so the user learns which model ids and keys actually work before
a conversation depends on them. Live is for the user to run: it spends a few tokens per slot and reaches the
network, so nothing starts it but `--live`.

A live probe goes through the real facade (`LLM`, `STT`, `TTS`, the classifier) with the slot pinned strictly,
so it exercises the same code a real call does, reaches that slot and no other, and a failure is that slot's
own. Because it uses the real chain, the result is recorded there like any call: a refused key rests that slot,
which is true and is what the first real call would have found out. The pin the user had is restored afterwards.

Statuses: `ready` (offline: key present and adapter built), `ok` (live: it answered), `failed(<kind>)` (live: the
error kind, as in errors.py), `skipped(<why>)` (no key, no voice, ...) and `error` (the configuration itself is
unusable: a role with no usable slot, a pin that names nothing, options an adapter refuses).

No key is ever shown. Rows carry variable NAMES, and every message is passed through `GlideConfig.scrub`.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from typesafe_sdk import Choice

from glide.computer.config import load_dotenv

from .base import Audio
from .config import ROLES, ConfigError, GlideConfig, SlotInfo, load_config
from .errors import AllProvidersFailed, ProviderError

PROBE_PROMPT = "Reply with the single word: ok"
PROBE_TOKENS = 5  # a few tokens: the point is that the slot answers, not what it says
PROBE_WORD = "Hello"
PROBE_STATE = "A traffic light is lit. Its colour is green."
PROBE_OPTIONS = ("yes", "no")  # two or more: a one-option question is settled without asking anyone
SILENCE_RATE = 16000
DETAIL_WIDTH = 110
TRUNCATED = "token limit"  # llm.py's wording for a reply that ran out of tokens before saying anything


@dataclass(frozen=True)
class Row:
    """One slot: `status` as described in the module docstring, `latency_s` for a live probe."""

    role: str
    slot: str
    status: str
    detail: str
    latency_s: float | None = None


def doctor(
    config: GlideConfig,
    *,
    live: bool = False,
    roles: Sequence[str] | None = None,
    timeout: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> list[Row]:
    """One row per slot of every role (or of `roles`), in chain order, the skipped slots included.

    `live=False` sends nothing anywhere. `live=True` sends each ready slot one tiny request (a five token chat, one
    second of silence, one short word, one two-option question), timed with `clock`; `timeout` limits each, in
    seconds, and the adapters' own limit applies when it is left out.
    """
    rows: list[Row] = []
    for role in roles or ROLES:
        rows.extend(_role_rows(config, role, live, timeout, clock))
    return rows


def _role_rows(config: GlideConfig, role: str, live: bool, timeout: float | None, clock: Callable[[], float]) -> list[Row]:
    try:
        infos = config.slots(role)
    except ConfigError as e:
        return [Row(role, "-", "error", config.scrub(str(e)))]
    ready = [i for i in infos if i.state == "ready"]
    problem = ""
    try:
        config.chain(role)  # builds it, applies GLIDE_PIN_*, and says why a role cannot be used
    except ConfigError as e:
        problem = config.scrub(str(e))
    pinned = None if problem or not ready else config.pinned(role)
    rows: list[Row] = []
    for info in infos:
        if info.state == "skipped":
            rows.append(Row(role, info.name, f"skipped({info.short})", config.scrub(info.reason)))
        elif problem or not live:
            rows.append(Row(role, info.name, "ready", _ready_detail(info, pinned)))
        else:
            rows.append(_probe(config, role, info, timeout, clock))
    if problem or not ready:
        rows.append(Row(role, "-", "error", problem or "no usable provider"))
    return rows


def _ready_detail(info: SlotInfo, pinned: tuple[str, bool] | None) -> str:
    if not info.provider:
        text = f"classifier prompt over the {info.name} chain; adapter built"
    elif info.env_var:
        text = f"{info.env_var} is set; adapter built"
    else:
        text = "no key needed; adapter built"
    if pinned and pinned[0] == info.name:
        text += "; pinned (strict)" if pinned[1] else "; pinned"
    return text


# -- live probes ---------------------------------------------------------------------------------


def _probe(config: GlideConfig, role: str, info: SlotInfo, timeout: float | None, clock: Callable[[], float]) -> Row:
    before = config.pinned(role)
    started = clock()
    try:
        config.pin(role, info.name, strict=True)
        started = clock()
        detail = _ask(config, role, info, timeout)
    except Exception as e:  # a probe reports its slot's failure, whatever it was, and goes on to the next slot
        root = _root(e)
        status = f"failed({root.kind if isinstance(root, ProviderError) else type(root).__name__})"
        return Row(role, info.name, status, config.scrub(_line(str(root))) or type(root).__name__, clock() - started)
    else:
        return Row(role, info.name, "ok", config.scrub(detail), clock() - started)
    finally:
        config.unpin(role)
        if before is not None:
            with contextlib.suppress(ConfigError):
                config.pin(role, before[0], before[1])


def _root(error: BaseException) -> BaseException:
    """A strict pin turns a slot's failure into AllProvidersFailed(exhausted); the slot's own error is what to report."""
    if isinstance(error, AllProvidersFailed) and error.errors:
        return error.errors[-1][1]
    return error


def _ask(config: GlideConfig, role: str, info: SlotInfo, timeout: float | None) -> str:
    family = role.split(".")[0]
    if family == "llm":
        return _ask_llm(config, role, info, timeout)
    if family == "stt":
        transcript = config.stt().transcribe(Audio(pcm=bytes(2 * SILENCE_RATE), sample_rate=SILENCE_RATE), timeout=timeout)
        said = f"{len(transcript.text)} characters" if transcript.text else "no text"
        language = f", language {transcript.language}" if transcript.language else ""
        return f"transcribed one second of silence: {said}{language}"  # empty text is a fine answer to silence
    if family == "tts":
        speech = config.tts().synthesize(PROBE_WORD, timeout=timeout)
        if not speech.pcm:
            raise ProviderError(f"{info.name} returned no audio", kind="content", provider=info.name)
        return f"{len(speech.pcm) / 2 / speech.sample_rate:.1f} s of audio at {speech.sample_rate} Hz"
    return _ask_classifier(config, info, timeout)


def _ask_llm(config: GlideConfig, role: str, info: SlotInfo, timeout: float | None) -> str:
    messages = [{"role": "user", "content": PROBE_PROMPT}]
    try:
        result = config.llm(role).chat(messages, max_tokens=PROBE_TOKENS, temperature=0.0, hedge=False, timeout=timeout)
    except Exception as e:
        root = _root(e)
        if isinstance(root, ProviderError) and root.kind == "content" and TRUNCATED in str(root):
            # A 200 with no text because a reasoning model spent its five tokens thinking: the key, the model id
            # and the request all worked, which is what is being asked. Reported as reached, with the reason.
            return f"reached, but the reply was cut at {PROBE_TOKENS} tokens, as a model that thinks first does{_effort(info)}"
        raise
    text = _line(result.text)[:40]
    usage = f" ({result.usage.input_tokens} in, {result.usage.output_tokens} out)" if result.usage else ""
    return f"replied {text!r}{usage}{_effort(info)}"


def _effort(info: SlotInfo) -> str:
    """Whether the slot's `reasoning_effort` was accepted, from what the client learned: the endpoint may refuse it."""
    wanted = info.options.get("reasoning_effort")
    settings = getattr(info.client, "settings", None)
    if not wanted or not isinstance(settings, Mapping):
        return ""
    return f"; reasoning_effort={wanted} " + ("accepted" if settings.get("reasoning_effort") else "refused, so it is dropped")


def _ask_classifier(config: GlideConfig, info: SlotInfo, timeout: float | None) -> str:
    classifier = config.classifier()
    question = Choice(
        instructions="Is the traffic light green?",
        criteria={"yes": "the light is green", "no": "the light is not green"},
    )
    extra = {} if timeout is None else {"timeout": timeout}
    reply = classifier.system_one(state=PROBE_STATE, questions={"q": question}, **extra)
    answer = reply.answers.get("q") if hasattr(reply, "answers") else None
    choice = getattr(answer, "choice", None)
    if choice not in PROBE_OPTIONS:
        raise ProviderError(
            f"{info.name} answered with something that is not one of the options", kind="content", provider=info.name
        )
    confidence = getattr(answer, "confidence", None)
    shown = f" (confidence {confidence:.2f})" if isinstance(confidence, int | float) else ""
    return f"chose {choice!r}{shown}"


# -- the table -----------------------------------------------------------------------------------


def _line(text: str, width: int | None = None) -> str:
    """`text` on one line, shortened to `width` with an ellipsis."""
    one = " ".join(str(text).split())
    return one if width is None or len(one) <= width else one[: width - 1] + "…"


def format_rows(rows: Sequence[Row], *, detail_width: int | None = DETAIL_WIDTH) -> str:
    """A table, one line per row, for a terminal. `detail_width=None` leaves long details whole."""
    if not rows:
        return "nothing is configured"
    header = ("ROLE", "SLOT", "STATUS", "TIME", "DETAIL")
    body: list[tuple[str, ...]] = []
    previous = None
    for row in rows:
        body.append(
            (
                row.role if row.role != previous else "",
                row.slot,
                row.status,
                "-" if row.latency_s is None else f"{row.latency_s:.2f}s",
                _line(row.detail, detail_width),
            )
        )
        previous = row.role
    widths = [max([len(h), *(len(line[i]) for line in body)]) for i, h in enumerate(header)]
    lines = ["  ".join(cell.ljust(width) for cell, width in zip(line, widths, strict=True)).rstrip() for line in (header, *body)]
    return "\n".join(lines)


def failed(rows: Sequence[Row]) -> bool:
    """Whether any row is a live failure or an unusable configuration."""
    return any(r.status.startswith("failed") or r.status == "error" for r in rows)


def main(argv: Sequence[str] | None = None, *, env: Mapping[str, str] | None = None) -> int:
    """The `glide doctor` command: 0 when nothing failed, 1 when something did, 2 when the file cannot be loaded.

    `.env` in the working directory is read into the environment first (only the real environment: with `env`
    given, nothing is touched), the way the other commands do.
    """
    parser = argparse.ArgumentParser(prog="glide doctor", description="Check the providers in glide.toml.")
    parser.add_argument("--live", action="store_true", help="send each slot one tiny real request (spends a few tokens each)")
    parser.add_argument(
        "--config", help="the glide.toml to read (default: $GLIDE_CONFIG, ./glide.toml, ~/.config/glide/glide.toml)"
    )
    parser.add_argument("--role", action="append", choices=ROLES, help="check only this role (repeatable)")
    parser.add_argument("--timeout", type=float, help="seconds to allow each live request")
    args = parser.parse_args(argv)
    if env is None:
        load_dotenv(Path.cwd() / ".env")
    try:
        config = load_config(args.config, env)
    except ConfigError as e:
        print(f"glide doctor: {e}", file=sys.stderr)
        return 2
    try:
        print(f"config: {config.source}")
        if config.defaulted:
            print(f"built-in chains in use for: {', '.join(config.defaulted)}")
        for warning in config.warnings:
            print(f"warning: {warning}")
        if args.live:
            print("live: one tiny request to each slot that has a key")
        rows = doctor(config, live=args.live, roles=args.role, timeout=args.timeout)
        print(format_rows(rows))
        return 1 if failed(rows) else 0
    finally:
        with contextlib.suppress(Exception):
            config.close()


if __name__ == "__main__":
    raise SystemExit(main())
