# Glide

Glide is a voice and text assistant that can also drive your Mac. You talk or type; it answers, or does the task on
the screen and tells you what it did. Every model, speech and vision vendor is configuration in `glide.toml`, never code.
It never clicks or types unless you pass `--act`, and it asks first before anything that takes over this machine.
Failures are never silent: a provider that fails over is printed as a switch, and an action is only called done after
a fresh look at the screen confirms it. Python 3.12 or later, managed with uv.

Status: the offline test suite passes except for known work in progress (see [CURRENT.md](CURRENT.md)). **Nothing in
this repository has been proven live**: voice, browsers, the research pipeline and the webhook and MCP servers are
tested only against fakes. [docs/LIVE_CHECKS.md](docs/LIVE_CHECKS.md) lists what you have to check yourself.

## Install

```sh
uv sync                    # the core only
uv sync --extra speech     # one extra
uv sync --all-extras       # everything below
```

The core stays small. Every other feature is an extra that is imported only by the command that needs it, and a missing
one is reported with the command to install it.

| Extra | Installs | Used by |
|---|---|---|
| `speech` | numpy, onnxruntime, sounddevice, websockets | `glide voice` (microphone, speaker, Silero voice-activity model you supply) |
| `aec` | an echo canceller (WebRTC AEC3 through livekit) | `[speech] echo_canceller = "webrtc"`: talking over Glide through speakers. Without it `auto` falls back to a numpy filter from `speech` |
| `ui` | PySide6 and Cocoa bindings | the pet window and point-to-ask (`python -m glide.ui`, `python -m glide.assistant.point_cli`) |
| `webhooks` | fastapi, uvicorn, pydantic, pyjwt | `glide app-server [--socket PATH]` | Serve the SwiftUI app over a local Unix socket (never a network port) | **Binds a socket**; once the app approves it, it can run computer tasks and use the microphone | yes to bind; the app asks per task |
| `glide webhooks serve` |
| `mcp` | the MCP Python SDK | real MCP transports (`glide mcp serve` itself needs no SDK) |
| `keyring` | keyring | an optional OS-keychain secret source for webhooks. The core never reads keys from it |
| `memory` | nothing (standard library only) | `glide memory`; names the feature. MCP-facing parts also need `mcp` |
| `all` | all of the above | |

## First run: `glide setup`

```sh
glide setup           # a page on this computer only (127.0.0.1, random port, a link that works once)
glide setup --text    # the same questions as plain prompts
```

It asks which features to turn on (computer control, webhooks and memory start off; confirmations are always on), which
providers to use per role (one key is enough, and a fallback chain is visible), shows the variable each provider reads and
whether it is set (never its value), and prints the exact `glide.toml` before writing it. An existing file is backed up
first. A key you paste is held in the wizard's memory for the optional test and for starting Glide, and is never written
anywhere. A test sends one tiny request, spends a few tokens, and only runs when you click it. The "where to get a key"
links live in `glide/setup/keys.toml`; check the provider's site, pages move.

## The control panel: `glide panel`

```
glide panel [--no-open] [--config PATH] [--idle-minutes N] [--runs DIR]
```

A local page (127.0.0.1, random port, a link that works once, then a session header; Host and Origin are checked, changes
are POST-only, no external assets) that stays open after `glide setup`. Tabs: Providers (the built-ins plus your own
OpenAI-compatible or classifier endpoints, key status by presence only, a Test button that spends tokens and only runs on
a click), Roles and chains (ordered slots with free-text model ids and the chain policy, the loader's one-line error
shown inline), Features and safety (computer control and detailed recording are OFF by default, the engine is legacy until
you choose otherwise), Chat / Ask (the real assistant; computer tasks start as a dry run that you approve, a real run
needs a second confirm, Stop reaches the running task, and an uncertain outcome is shown as "completion unknown; nothing
was retried"), Files (off by default; plan, preview, approve by typing the plan hash, run, undo) and Status (offline
doctor, last runs, and the unresolved-run marker with how to clear it after you have looked).

Changes go to glide.toml only after a visible diff and a confirm; the old file is backed up first, and comments in the file
are not kept. The panel shows whether it is editing the project-local or the user-level file. A key pasted into the page is
held in this process's memory only and is never written or echoed. Panel-only switches live in a `[panel]` table
(`glide/panel/settings.py`). Nothing here is a substitute for the live checks in `docs/LIVE_CHECKS.md`.

## Set up

```sh
cp glide.toml.example glide.toml        # edit the chains; the model ids in it are unverified starting points
export OPENAI_API_KEY=...               # name the variables your chosen providers use
glide doctor                            # reads the file, shows each slot and what is switched on
glide doctor --live                     # also sends each slot with a key one tiny real request (spends a few tokens)
```

Keys are never written to `glide.toml`: a provider names the environment variable that holds its key, and a slot whose
variable is unset is skipped with a warning. Put them in your shell or in a `.env` file in the folder you run from
(`.env` is git-ignored). The built-in providers and their variables: `OPENAI_API_KEY`, `OPENROUTER_API_KEY`,
`DEEPSEEK_API_KEY`, `GEMINI_API_KEY`, `ELEVENLABS_API_KEY`, `TYPESAFE_API_KEY`; any OpenAI-style server of your own is a
`kind = "openai_compat"` provider with its own variable. Other variables Glide reads:

| Variable | Meaning |
|---|---|
| `GLIDE_CONFIG` | the `glide.toml` to read (default: `./glide.toml`, then `~/.config/glide/glide.toml`) |
| `GLIDE_PIN_LLM_FAST`, `GLIDE_PIN_LLM_SMART`, `GLIDE_PIN_STT`, `GLIDE_PIN_TTS`, `GLIDE_PIN_CLASSIFIER`, `GLIDE_PIN_LLM_PLANNER` | prefer one provider slot; a trailing `!` means nothing else is ever tried |
| `GLIDE_BROWSER_PROVIDER`, `GLIDE_BROWSER_ENDPOINT`, `GLIDE_BROWSER_TARGET`, `GLIDE_PLAYWRIGHT_SESSION`, `GLIDE_PLAYWRIGHT_CLI` | the browser provider for one run (see `[browser]` in the example file) |
| `GLIDE_MEMORY`, `GLIDE_MEMORY_AUTO`, `GLIDE_DATA_DIR` | memory on/off, auto capture, data folder |
| `GLIDE_WEBHOOK_CONFIG`, `GLIDE_AGENT_TOKEN` | the webhook file, and the worker's agent token |

## The command tree

One command, `glide`. The first column is what it does; the second is **what it takes control of**, which is what needs your
yes before an agent (or you, if you are being careful) runs it. Commands marked "no" need no approval: they touch no
screen, input, microphone or app.

| Command | What it does | Takes control of | Approval |
|---|---|---|---|
| `glide ask TEXT [--speak] [--engine E]` | One answer, or one computer task | Answer: nothing (`--speak` plays audio). A computer request is a **dry run that still captures the screen**; with `--act` it clicks and types | yes if it may route to the computer, or with `--act` or `--speak` |
| `glide chat [--speak] [--act] [--engine E]` | A typed conversation; `/stop /act /pin /unpin /status /help /quit` | As `ask`. `/act` turns real clicking and typing on | yes, as `ask` |
| `glide listen [--auto] [--text-only] [--act] [--engine E]` | Push-to-talk: Enter starts and stops | The **microphone**, the speaker, and as `ask` for tasks | yes |
| `glide voice [--act] [--engine E]` | Hands-free: always listening, interruptible (`speech` extra) | The **microphone** and speaker for as long as it runs; with `--act`, the screen, keyboard and mouse | yes, always |
| `glide computer GOAL [--act] [--engine legacy\|structured] [--record-content]` | Drive the screen toward a goal. `glide-computer` is the same command | Without `--act`: **captures the screen** once and says what it would do. With `--act`: mouse, keyboard, apps. `--engine structured` with a browser provider attaches to the browser you started | yes, always |
| `glide inspect [GOAL]` | Count down, capture the screen, show what the classifier would be sent. `glide-inspect` is the same | **Captures the screen** and, unless `--no-open`, opens two files | yes, always |
| `glide doctor [--live]` | What each provider slot can do, and what is switched on | Nothing without `--live`. `--live` sends real requests (spends money) and reaches any local model server you configured | `--live` only |
| `glide status` | The chains: slots, pins, resting slots, recent switches | Nothing | no |
| `glide memory ...` | Local memory administration (off unless `[memory] enabled = true`) | Files in your data folder, only when memory is on | no |
| `glide mcp status`, `glide mcp serve` | Show the MCP settings; serve Glide over stdio | `serve` reads stdin and writes stdout until it closes; it offers no tools unless you set `server_memory`. It starts no other program | no for `status`; ask for `serve` |
| `glide webhooks serve` | Receive authenticated webhooks and queue agent requests (`webhooks` extra) | **Binds a port** (loopback by default). `--check-config` builds and starts nothing | yes to bind a port |
| `glide webhooks work [--allow-model] [--allow-desktop [--act]]` | Consume queued requests, one at a time | Calls your server and, with `--allow-model`, your LLM. With `--allow-desktop`, **the screen and apps**, and each run still needs your yes on that terminal. `--act` lets an approved run click and type | yes; and a yes per run |

**Which engine drives a computer task.** `legacy` (the screen loop) is the default everywhere. `structured` (planned effects
checked after each action, the research route and the `[browser]` providers) is opt-in from every front end: `--engine
structured` on `ask`, `chat`, `listen`, `voice` and `computer`, `GLIDE_ENGINE`, `[computer] engine` in `glide.toml`, the
app's `computer.engine` setting, or the toggle in the pet. The flag beats the environment, which beats the file. It stays
opt-in because the structured engine is not live-qualified, and the audit rated the legacy loop weaker on untrusted screen
text, so flipping the default is your decision after the live checks in `docs/LIVE_CHECKS.md`. `glide doctor` and the
`glide computer` header print the engine in force.

Older names still work: `glide-computer`, `glide-inspect`, `glide-webhooks` (`glide webhooks serve`) and
`glide-webhook-worker` (`glide webhooks work`).

## Interrupting Glide

Stopping is matched before any model is called, so it works the same typed or spoken and costs no network:

- say or type **stop** (English, Cantonese and Mandarin phrases are built in; `/stop` in `chat` and `listen`);
- **Ctrl-C** (a second Ctrl-C leaves `chat` and `listen`);
- on macOS, **move the mouse into the top-left corner** of the screen during an `--act` run.

A stop cancels the request in flight (the connection of each model, transcription and speech call is closed and the
caller returns at once), the voice, and the task. A task stops before its next action, and between two typed characters.
Talking over Glide while it speaks cuts the voice and takes your new request, with a **headset** (`headset = true` in
`[speech]`).

Limits, which are real:

- **Cancelling is not undoing.** An action that was already sent stays done. If a stop lands while an action is in flight,
  Glide reads the screen once, reports "completion unknown" and stops. It never repeats a write whose outcome it cannot
  confirm; you reconcile it.
- **Talking over Glide through speakers depends on echo cancellation.** `[speech] echo_canceller` picks one (`auto` takes
  the first that loads and says which; `none` is half duplex: the microphone hears silence while Glide speaks and for a
  short tail after). How well it works on real speakers has not been measured: see
  [docs/LIVE_CHECKS.md](docs/LIVE_CHECKS.md). A headset avoids the question.
- A model request still waiting for its first byte cannot be closed from outside; its answer is thrown away and its thread
  ends on the request's own deadline.
- None of this has been shown against a real provider, microphone or screen yet: see [docs/LIVE_CHECKS.md](docs/LIVE_CHECKS.md).

## Safety model

- **Approval first.** [AGENTS.md](AGENTS.md) lists what needs a yes, for that exact command and once only: running the
  screen-driving loop, moving the mouse, pressing keys, clicking, typing, AppleScript, launching or quitting an app or
  browser, opening a URL or file, any screen capture, and requests to local model servers. The "Takes control of" column
  above is that list applied to each command. Webhook-triggered desktop runs are off unless the worker is started with
  `--allow-desktop` and need a yes for every run; approval is never cached.
- **Dry by default.** Without `--act`, nothing clicks or types.
- **Recording is opt-in.** Stored run data omits utterances, typed text, captured content and raw URLs unless detailed
  recording is turned on (`glide computer --record-content`; memory and webhook content have their own switches, all off).
- **Keys stay in the environment.** They are never written to a file, a log or a prompt; text on its way to the terminal is
  scrubbed.
- **Untrusted text.** Page, task and provider text is data, never instructions.
- **Offline tests.** `tests/` never reaches the real machine: `tests/conftest.py` refuses input events, AppleScript, screen
  capture, accessibility, `subprocess.Popen` and non-loopback network. Do not weaken it; a new call that reaches the machine
  adds its refusal in `tests/guards_<name>.py` in the same change.

Checks: `uv run --python 3.13 pytest -q` and `uv run --python 3.13 ruff check . && uv run --python 3.13 ruff format .`

## Where to read next

| | |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Package map, request flow, cancellation, provider chains, recording |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Decisions D0-D18 and where each stands |
| [docs/LIVE_CHECKS.md](docs/LIVE_CHECKS.md) | Everything that cannot be proven offline, as checklists |
| [CURRENT.md](CURRENT.md) | One-screen status |
| [AGENTS.md](AGENTS.md), [CLAUDE.md](CLAUDE.md) | The rules for agents working on this repository |
| [glide.toml.example](glide.toml.example) | Every setting, with comments |
| [docs/memory.md](docs/memory.md), [docs/mcp.md](docs/mcp.md), [glide/webhooks/README.md](glide/webhooks/README.md) | The optional features |
| [docs/history/](docs/history/) | The consolidation notes. Historical: they describe the workspace as it was before integration |

`main` is protected: work goes on a branch and a pull request.
