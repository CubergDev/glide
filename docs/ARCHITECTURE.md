# Architecture

A short map of how Glide is built, written from the code of the integration branch. Every statement here was checked
against the source; where something is planned, or lives on a branch that is not merged, it is under "Not in this tree
yet" and nowhere else. Setup and commands are in the [README](../README.md); the rules every change follows are in
[AGENTS.md](../AGENTS.md); what only a person can verify is in [LIVE_CHECKS.md](LIVE_CHECKS.md).

## Package map

| Path | What it is |
|---|---|
| `glide/cli.py` | The one `glide` command tree. Optional packages are imported only by the command that needs them. |
| `glide/features.py` | What `glide doctor` reports for voice, memory, webhooks and mcp: standard library only, starts nothing. |
| `glide/providers/` | Swappable vendors. `config.py` reads `glide.toml` into one chain per role (`llm.fast`, `llm.smart`, `stt`, `tts`, `classifier`, plus optional `llm.planner` and `llm.research`). `chain.py` is the failover engine, `errors.py` the shared `ProviderError` vocabulary, `base.py` the shapes an adapter speaks, `interrupt.py` how an adapter lets a cancel reach its connection, `doctor.py` what `glide doctor` measures. Adapters: `llm.py`, `stt.py`, `tts.py`, `classifier.py`, and `writer_client.py` (the neutral writer over the chains). |
| `glide/assistant/` | Turns text or speech into an answer or a computer task and speaks the result: `core.py` (`Assistant`), `answer.py` (the streamed answer's prompt and messages), `clarifier.py` (the channel for a `clarify` decision), `speech.py` and `audio_io.py` (sentence-by-sentence speaking, push-to-talk device), `tasks.py` (`ComputerTask`, one at a time, on a worker thread). |
| `glide/computer/` | The screen-driving loop. `runner.py` (the legacy loop: perceive, classify, act, verify, with run folders), `perception.py` and `ax_walk.py` (OCR and accessibility), `decide.py` and `actions.py`, `macos.py` and `windows.py` behind `platform_adapter.py`, `control.py` (`RunControl`), `diagnostics.py` (recording), `generation.py` (writer request and result shapes), `desktop_access.py` (permissions). |
| `glide/computer/execution/` | The structured engine: contracts, planning, policy, progress, query forms, routing (`decide`), the research supervisor and page reading, and the backends `dom.py` (CDP, over the client in `glide/computer/browser/cdp.py`: a session to a browser the person already started, never a launcher), `native.py`, `obscura.py`, `playwright_cli.py` (with `playwright_driver.js`). `providers.py` picks the backend from `[browser]`; `spawn.py` is the one place a process may be started. |
| `glide/speech/` | Hands-free voice: `vad.py` (who is speaking), `audio.py` (one full-duplex device with the echo guard and headset mode), `turns.py` (`VoiceLoop`: turns, self-correction, barge-in, idle), `session.py` (builds the real thing), `settings.py` (the `[speech]` table), `elevenlabs.py` (an optional realtime TTS adapter). |
| `glide/memory/` | Opt-in memory, context planning and a bounded tool harness. Off by default. [memory.md](memory.md). |
| `glide/mcp/` | Glide as an MCP server (`server.py`, a small JSON-RPC core) and as a client (`client.py`, `bridge.py`, `transport.py`). [mcp.md](mcp.md). |
| `glide/webhooks/` | Webhook ingress, a durable queue and a separately started worker. Off by default. [its README](../glide/webhooks/README.md). |
| `tests/` | Offline only. `conftest.py` refuses any call that would reach the real machine; each feature adds its refusals in `tests/guards_<key>.py`, which `conftest.py` registers by path. |

## Not in this tree yet

These exist on other branches and are **not** part of the code described above. Whoever merges them should delete the
matching line and update the pages named in the last column.

| Piece | Branch | What it adds | Pages to update |
|---|---|---|---|
| `glide/ui/` (PySide6 pet) and point-to-ask (`glide/assistant/point_*.py`) | `consolidation/pet-point` | The pet over the in-process assistant; read-only "point and ask"; `docs/live-checks/pet-point.md` | README, this page, LIVE_CHECKS, DECISIONS (D5, D11) |
| Echo cancellation and barge-in through speakers (`glide/speech/echo.py`, the `aec` extra) | `consolidation/barge-in` (already merged into `consolidation/06-barge-in`) | Interrupting by voice without a headset; `docs/voice-echo-cancellation.md`; a longer `docs/live-checks/voice.md` | README (extras, limits), LIVE_CHECKS, this page |
| `app/` (SwiftUI menu bar app, wire protocol) | `consolidation/app-swiftui` | A scaffold that builds and passes its own tests; it has never been launched, and the core's socket server that it talks to is not written anywhere | README, LIVE_CHECKS, DECISIONS (D11) |
| Rewritten CI and review setup | `consolidation/ci-cd` | Workflows, templates, `REVIEW_SETUP.md` | DECISIONS (D18) |

## Request flow

```
text (ask, chat, listen) ----------------------------+
speech (glide voice): microphone -> VAD -> VoiceLoop -+--> Assistant
                      push-to-talk (glide listen) ----+      |
                                                             v
        1. stop phrase?  router.fast_path: no model, no network  --> stop everything
        2. one fast-LLM call (router.Router), JSON with a schema --> "answer", "computer" or "stop"
           any failure of that call lands on "answer", never on "computer"
                 |                                  |
          answer (streamed, spoken a             computer task: ComputerTask on a worker thread
          sentence at a time)                    (a dry run unless --act or /act)
                                                    |
                                          runner.run: the legacy loop
                                          perceive (screen, OCR, accessibility)
                                          -> classifier chain picks one action -> act -> look again
                                          -> writer chain composes the final answer
```

- **Speech in** is `Assistant.handle_audio`; the transcriber is `config.stt()`, a chain like any other, so the voice
  loop owns no vendor. **Speech out** is a `Player` (`glide listen`, `ask --speak`) or the full-duplex device
  (`glide voice`).
- **The structured engine** (`computer/execution/engine.py`) is a second way to run a task: scope (`routing.decide`:
  execute, research, reason or clarify), plan as milestones that are effects to observe, then observe, wait for
  readiness, let the classifier bind one offered action, dispatch it through `RunControl.dispatch`, re-observe, and
  count a milestone only when a fresh observation shows its effect. `research` and `reason` go to the supervisor
  (`research.py`): the model plans and writes, the browser only reads and navigates, and every citation must quote a
  page read in this run. The backend comes from `providers.make_backend` (CDP, native, Obscura or Playwright CLI).
  **Which engine runs is one setting, `engine` = `legacy` or `structured`, and the default is `legacy`.** One resolver
  (`computer.config.engine`, through `features.engine_for`) reads `--engine` (on `ask`, `chat`, `listen`, `voice`,
  `computer`, and the pet), else the app's or the pet's toggle, else `GLIDE_ENGINE`, else `[computer] engine` in
  `glide.toml`, else `legacy`; a bad value is a one-line error. A front end puts its choice on the configuration
  (`engine_choice`) and `ComputerTask` turns it into `RunConfig.engine` (with the `[browser]` and `[research]` settings
  for the structured one), so every path that runs a task, webhooks included, uses the same answer. The default stays
  `legacy` because the structured engine has not been qualified live, and the audit rated the legacy loop weaker on
  untrusted screen text; flipping the default is the user's decision after the live checks in `docs/LIVE_CHECKS.md`.
  `glide doctor`, the `glide computer` header, the app's settings and the pet's badge say which engine is active, and
  `run.json` records it (never content).   The engine's own `decide` now runs only when nothing routed the request first (`glide-computer` run directly): the assistant asks
  the one router, `glide/routing/` ([ROUTER.md](ROUTER.md): stop phrase, classifier chain, fast-LLM JSON call; a failed or
  unsure router means answer), and passes its `execute`/`research`/`reason` decision to the engine as `RunConfig.route`.
  `glide chat` sets `clarify=True`, so a `clarify` decision asks one question and the next typed line is the answer; every
  other front end keeps it off and says what is needed.
- **Webhooks** do not go through the assistant: the worker takes a queued `agent.task.requested` call, and only with
  `--allow-desktop`, a yes for that run, and (to click or type) `--act` plus the source's `allow_actions`, hands it to a
  `ComputerExecutor`.
- **MCP**: `glide mcp serve` serves over stdio and offers no tools unless `server_memory` is `read` or `write` (then memory tools). The client code
  (`MCPClient`, `MCPBridge`, `open_stdio_client`) is a library; no command or assistant path calls it yet.

## RunControl and cancellation

One `RunControl` (`computer/control.py`) is the cancel token of a request, a task or the voice. It travels in a context
variable, so no signature carries it.

- `Assistant.stop()` bumps an epoch, cancels every request in flight, then the task, then cuts the speech.
  `Assistant.interrupt_speech()` is the barge-in hook: it cuts the voice and drops any request still being heard. A
  stop phrase works from any front end because it is matched before any model is called.
- A task checks the control before each step, before each model reply is used, and inside every input call (between two
  typed characters). The mouse in the top-left corner also aborts (`macos.py`), through the older `abort_on` hook that
  `ComputerTask.stop()` sets together with the control.
- `dispatch` marks a write `in_flight` until a fresh observation clears it. A stop that lands mid-action therefore
  leaves an **unknown** outcome: the runner reads the screen once, read-only, reports "completion unknown" and stops. It
  never replays the write.
- Every provider call runs through `control.interruptible` (`providers/chain.py`): the caller returns at once with a
  `ProviderError` of kind `cancelled`, and a cancel is never a provider fault, never retried, never a `SwitchEvent`.

Limits, stated plainly: cancelling is not rolling back, so an action already sent stays done. A request still waiting
for its first byte cannot be closed from outside (`providers/interrupt.py` finds its connection in httpcore's pool by
reading two private attributes and does nothing if they moved); its thread ends on the request's own deadline and its
answer is thrown away. In speaker mode the microphone hears silence while Glide speaks (plus `echo_tail_s` after), so you
cannot interrupt by voice through speakers in this tree; a headset can. None of this has been shown against a real
provider, microphone or screen: see LIVE_CHECKS.

## Provider chains and SwitchEvent

A `Chain` holds the slots of one role in order. A call goes to the first healthy slot; a failure moves on, a slot that
keeps failing rests for a cooldown, a user can pin a slot (`/pin`, `GLIDE_PIN_*`, strict means nothing else is tried),
and a slow slot can be raced against the next (`hedge_after_s`). The chain reacts only to `ProviderError`; any other
exception is a bug and propagates. **Every switch is a `SwitchEvent`**, kept on `chain.events` and handed to the
listener the CLI registers before any chain exists, so it is printed to stderr as it happens. The browser provider
fallback (`[browser] fallback`) is the same rule: it applies only before anything has been done, never swaps a backend
mid-run, never includes `native`, and is reported as a `SwitchEvent` too. Adapters raise only `ProviderError`; its
messages carry a status and a short snippet, never a key, header or request body. Keys come from the environment
variables `glide.toml` names; model ids, endpoints and voice ids are configuration, and `tests/test_no_hardcoded_literals.py`
scans for them.

## Recording

Stored run data omits utterances, typed text, captured content and raw URLs unless detailed recording is opted into.

- Computer runs: `record_content` is off by default. Off: `run.json` holds counts, timings, the outcome and a scrubbed
  `failure`, and no goal, answer or history; URLs are cut to scheme and host and an exception is recorded by type. On
  (`glide computer --record-content`): the goal, answer, history, screenshots and `events.jsonl`, `diagnostic.json` and
  `failure.md` are also written, as 0600 files in 0700 folders. No other command sets it in this tree.
- Memory: off until `[memory] enabled = true`; stores only what you save, plus run outcomes and allowlisted event metadata.
  It is not read by the assistant.
- Webhooks: the queue keeps identifiers only unless `"record_content": true` in the webhook file.

## Where the decisions landed

[DECISIONS.md](DECISIONS.md) lists D0-D18 with what each says, where it landed in the code, and whether it is
implemented, partly done, not yet done, or waiting for live proof. The original table is kept in
[history/HANDOFF.md](history/HANDOFF.md) section 2. The two decisions with the most left to do are D9 (the router's live calibration) and D11
(the SwiftUI app and its socket server).
