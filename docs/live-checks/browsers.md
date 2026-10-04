# Browser providers: live verification checklist

Glide can drive a browser through four providers: **CDP**, **native** (the macOS or Windows desktop), **Obscura** and the
**Playwright CLI** (D7). Everything offline passes, but offline tests use fakes. Per `docs/history/HANDOFF.md` section 8, only CDP was
ever qualified against a real browser, and that was a report, not something this branch re-checked. **None of the four
has been run live on this branch.** This list is for you to run, one provider at a time. Nobody else should run it for
you, because every step takes over a browser or this Mac.

Each step says what to do and what you should see. If you see something else, stop and note which step; do not retry a
step that changed a page.

## Before you start

- Use a throwaway browser profile and harmless pages. Do not type anything private. Glide never needs a password.
- Nothing below should be pasted anywhere with page text, URLs or typed text in it. The run records omit those unless
  you turn on content recording. A failure message should name a setting or a port, never a page.
- Settings live in `glide.toml` (see `glide.toml.example`, section `[browser]`) and can be overridden per run by
  `GLIDE_BROWSER_PROVIDER`, `GLIDE_BROWSER_ENDPOINT`, `GLIDE_BROWSER_TARGET`, `GLIDE_PLAYWRIGHT_SESSION` and
  `GLIDE_PLAYWRIGHT_CLI`. Endpoints must be `http://127.0.0.1:<port>` or `http://localhost:<port>`.
- Glide never starts a browser or Obscura, and never installs the Playwright CLI. You start them.

### A small read-only probe

This snippet uses only the backends, read-only, and prints counts. It makes the same calls the engine makes, so run it only
where you accept that. The end-to-end command is `glide computer GOAL --engine structured --act` (section 6):

```
uv run --python 3.13 python - <<'PY'
from glide.computer import browser_settings
from glide.computer.execution import providers

browser_settings.use({"provider": "cdp", "cdp": {"endpoint": "http://127.0.0.1:9222"}, "target": ""})
backend = providers.make_backend(act=False)
try:
    seen = backend.inspect()
    print(backend.transport, "tabs:", len(seen.tabs), "ready:", seen.ready, "elements:", len(seen.elements))
finally:
    backend.close()
PY
```

Change the `provider` and its table for the other providers. It prints counts only, never page text.

## 1. CDP (a Chromium browser you started)

Start the browser yourself with remote debugging on a loopback port and a profile directory that is not your everyday one
(recent Chrome ignores the debugging port on the default profile). Check the browser's own documentation for the current
flags. Then:

| Step | Expect |
|---|---|
| Run the probe with one tab open | `cdp tabs: 1 ready: True elements: N` with N above zero on a page with links |
| Open a second tab and run it with `target = ""` | An error that names `GLIDE_BROWSER_TARGET`, and no action taken |
| Set `target` to the tab id | The probe reads that tab |
| Set `target` to `new` | No tabs are shown and no page is touched; the first navigation creates a tab of its own |
| Stop the browser and run the probe | `Cannot connect to cdp at http://127.0.0.1:9222. Start or reconnect ...` and nothing else |
| Navigate the tab to a slow page and run the probe while it loads, several times | Never "Cannot connect"; either an observation or "Couldn't read the selected page. Check the page before retrying." (this is the false-disconnect fix, D7) |
| With a harmless page: click a button, type into a search box, scroll, press Enter | Each effect shows in the next observation. Typing into a password field is refused |
| Create a tab and close it | The tab appears, then is gone, and the selected tab is a remaining one |
| A page with a `<video>`: start it, then observe | `media` shows `paused: False` and a growing `current_time` |
| Press Stop while a click is in flight | The mouse release still goes out (no held button) and the run reports the outcome as unknown, not failed |

## 2. Obscura (its loopback server, started by you)

Start Obscura as its documentation says, with its CDP port on loopback. Then run the probe with `provider = "obscura"`
and `[browser.obscura] endpoint = ...`.

| Step | Expect |
|---|---|
| Probe | `obscura tabs: ... ready: ...` (Obscura refuses a browser `Origin`; Glide sends none) |
| Type into an `<input>`, then into a `contenteditable` | The input is replaced (select by range, then insert). The editable is refused: "Obscura typing outside input/textarea" |
| Send a key other than Enter, Delete or Escape, or any shortcut | `Unsupported capability required: Obscura keyboard shortcut ...` and no key sent |
| Server not running | `Cannot connect to obscura at ... Start or reconnect ...` |

## 3. Playwright CLI (an already open session)

Install Microsoft's `@playwright/cli` yourself and open a session by the name you configure (default `glide`), as that
CLI's documentation says. Glide never opens the browser and does not search the working directory for the program.

First live check, and the one most likely to fail: **the exact way Glide calls the CLI has never run against a real
CLI.** It runs `<playwright-cli> -s=<session> run-code --filename=<tmp file> --raw --json`, with a command file that
contains the fixed driver and one JSON request, and expects JSON on stdout that looks like `{"result": "<JSON text>"}`
(or `{"isError": true, ...}` on failure). If your CLI version does not accept those flags or prints another shape, every
Playwright step fails with "Invalid Playwright CLI response" or "Playwright operation failed". Record the version and
what it printed. That is a driver change, not a configuration problem.

| Step | Expect |
|---|---|
| Probe with `act=False` (the probe above) | `Unsupported capability required: Playwright CLI inspection requires Allow computer actions; use CDP for passive inspection`, and no process started. Its inspection can recreate a tab, so it is refused when actions are off. (With a fallback configured, the same refusal is the message `Playwright CLI is not enabled: it needs Allow computer actions ...` and the next provider is tried.) |
| Probe with `act=True` and the session open | `playwright-cli tabs: ... ready: True elements: N` |
| CLI not installed (rename it) | `Playwright CLI is not installed: install @playwright/cli, or set its path ...` before anything is spawned |
| Session not open | `Cannot connect to playwright-cli at playwright:glide. Open the selected Playwright CLI session and try again.` |
| Navigate while a probe reads | No "Cannot connect": the CLI's own "Execution context was destroyed" text is treated as a page change and the read is retried. The exact wording varies by engine and version, so note any read that fails as a connection error during navigation |
| Type, click, press a shortcut | One operation each; a key or button is never left held. A request that arrives after Stop is refused by the gate |
| Press Stop mid-write | The call waits for the write to settle, then reports Stop. The outcome is unknown, never replayed |
| Firefox or WebKit sessions | `Glide.selectAll` uses `ControlOrMeta+A`; a CDP session is unavailable there and Glide does not need it |

Optional, needs only Node and no browser: the driver's own logic (cancellation gate, key and button release after Stop,
the refusal to overlap an unresolved write) is JavaScript. `playwright_driver_world.cjs` here runs the driver against
fake Playwright objects. From the repository root:

```
node docs/live-checks/playwright_driver_world.cjs glide/computer/execution/playwright_driver.js 10
```

Expect `{"quantity":10,"verified":true}`; try 1 and 17 as well. This harness was ported by renaming `Permit.*` to
`Glide.*` and has not been run since.

## 4. Native desktop (macOS or Windows)

This is the only provider that moves your pointer and keyboard. Grant Accessibility and Screen Recording to the
application that launches Glide when asked, and nothing else.

| Step | Expect |
|---|---|
| Probe with `provider = "native"` | Counts for the front app; no browser is launched, and the URL of a browser that is not running is never queried |
| Open a browser with several tabs | Tabs and the active tab are read without walking the whole UI (scoped observation) |
| Type into a text field | The field is focused, then written by value, falling back to keystrokes only when the value cannot be set |
| Press Stop during typing | No further character is typed |

## 5. The optional fallback

Only if you want it. In `glide.toml`:

```
[browser]
provider = "cdp"
fallback = ["obscura"]
```

| Step | Expect |
|---|---|
| Both running | CDP is used; no switch |
| Stop CDP only | Obscura is used, and the switch is reported (`browser_provider_switched`, and a `SwitchEvent` for whoever listens). It is never silent |
| Stop both | An error naming the last provider tried, and no action taken |
| Stop CDP **during** a task | The task fails with a connection error. It does **not** move to Obscura: a switch happens only before anything is done |
| `fallback = ["native"]` | Refused at start: the native desktop is never a fallback |

## 6. End to end, through the engine

The engine is merged. It is reached only by `glide computer`, not by the assistant (`ask`, `chat`, `listen`, `voice` still use the
legacy loop). Repeat sections 1 to 4 end to end, once per provider, with
`GLIDE_BROWSER_PROVIDER=<provider> glide computer "<task>" --engine structured --act --out runs/<name>` (add the endpoint, target
or session variables from `glide.toml.example`). Without `--act` the run stops after one proposed step with outcome `dry run`.
Add: a task that opens a new tab on a loopback test page and reads it; the zero-action refusal when the browser is down (the run
must report "blocked" with the setting to fix, `steps_taken` of 0 and exit status 1); and `glide doctor` for the model providers,
which does not test browsers. [../LIVE_CHECKS.md](../LIVE_CHECKS.md) indexes this with every other live check.
