# Glide app (SwiftUI, macOS)

The native Mac app for Glide (decision D11): a menu bar item, a Settings window, a voice status view and a raccoon pet. It is a separate program from the Python core and talks to it over a local Unix socket with the JSON-lines protocol in [PROTOCOL.md](PROTOCOL.md). It has no Python in it and no provider, model or voice names: all of those come from the core's settings payload.

Status: **builds and passes its tests; has never been launched.** Nothing here was run on screen, with a microphone or against a real core (the core's socket server does not exist yet). The checklist at the end is for you.

## Layout

| Path | What |
|---|---|
| `Sources/GlideProtocol` | Wire models, codec, line framing. Foundation only. |
| `Sources/GlideClient` | Unix socket transport, reconnecting client (`GlideClient`), in-memory transport for tests. |
| `Sources/GlideAppKit` | `AppModel` (state), pet state machine (`PetPose.swift`), SwiftUI views. No `@main`. |
| `Sources/GlideApp` | The executable: `MenuBarExtra` and `Settings` scenes. |
| `Tests/` | offline tests (protocol, client, app model and pet). |
| `scripts/test.sh` | Runs the tests (see below for why a script). |

## Build and test

Needs a Swift 6 toolchain and macOS 14 or later. It was built with Swift 6.4 and the Command Line Tools only (no Xcode).

```sh
cd app
swift build            # compiles all targets, including the executable. Does not run anything.
./scripts/test.sh      # swift test, with the Testing macro plugin path when Xcode is not installed
```

Use `./scripts/test.sh` (or `swift test` with Xcode selected), not `swift build --build-tests`.

Toolchain notes, found by building here:

- With the Command Line Tools alone, `swift test` fails with `plugin for module 'TestingMacros' not found` unless `-Xswiftc -plugin-path .../usr/lib/swift/host/plugins/testing` is given. `scripts/test.sh` adds it.
- The SwiftUI `@State` and `#Preview` macros need Xcode's macro plugins, which the Command Line Tools lack. The code uses neither: the two bits of view state (the text field draft and the switches disclosure) live in `AppModel`, and `GlideApp` holds the model in a `let`. If Xcode is installed later, `#Preview` blocks can be added.
- The `ld: warning: search path ... not found` lines in build output are harmless.

The tests do not touch the screen, input, microphone, other apps or the network. Two of them use a Unix socket in a temp directory inside the test process (a `socketpair`, and a bound socket under `/tmp`), which is local to the process. If a sandbox refuses the `bind`, that test reports an issue saying so.

## Running it (you, not an agent)

```sh
swift build
.build/debug/GlideApp --socket /path/to/core.sock     # or set GLIDE_SOCKET
```

The core must already be listening on that socket. The app is a menu-bar-only app (`NSApp.setActivationPolicy(.accessory)`); there is no Dock icon. The built executable is unbundled. A proper `.app` bundle (Info.plist, signing, login item) is not part of this change.

The app never opens the microphone or the screen itself. Those permissions belong to the Python core process.

## How it behaves

- **Menu bar window:** pet and state, approval cards, task line with the attempted/verified distinction, hands-free status with a level meter, the last few lines of conversation, a text field, Interrupt and Stop, a Hands-free switch, provider switches, Settings, Quit.
- **Approvals:** the exact command as plain text, `Approve once` or `Deny`, a countdown. No "always". Only a click answers. At the deadline the card disappears (the core has already denied).
- **Settings:** General (core status, socket path, allow screen control), Voice (hands-free, headset, silence), Providers (each role's chain and key-variable status from the core, pinning), Privacy (record content, off by default). Every change goes to the core and the screen then shows what the core reports.
- **Connection:** reconnects with backoff. On a drop it clears approvals, warns if a task was running, and never resends anything.
- **Pet moods:** idle, listening, typing, thinking, talking, question, happy, sad, sleeping, the same set as the PySide6 raccoon. Sleeping = not connected. Question = an approval is waiting or a write's outcome is unknown. Happy or sad for four seconds after a task ends. Drawn from placeholder shapes; Reduce Motion shows a still frame.

## Not done (decisions or later work)

- The core's socket server (Python). PROTOCOL.md says what it must do.
- A floating desktop pet (needs an `NSPanel`; the pet currently lives in the menu bar window).
- App bundle, signing, notarization, launching or supervising the core from the app.
- Windows. The PySide6 pet stays as the fallback until this one is verified live (the open D11 question; this default matches the handoff).
- Whether the menu bar icon should flash on an approval when the window is closed. It changes symbol; a notification would be a new permission.

## Live verification checklist

Tick these on your Mac once the core serves the protocol. None can be proven offline.

**Connection**
- [ ] Start the core, then the app. The menu bar icon appears and no Dock icon does. The window says "Connected".
- [ ] Quit the core. The window says "Not connected" within a moment and the pet sleeps. Restart the core: the app reconnects by itself and settings reload.
- [ ] Start the app with no core: it keeps retrying quietly and does not crash. With `--socket` pointing at a regular file, it says the path is unusable and does not retry.
- [ ] Make the socket owned by another user (or a different file type): the app refuses it.
- [ ] Core replies with another protocol `v`: the app shows "unsupported protocol version" and does not loop.

**Voice and conversation**
- [ ] Hands-free switch in the window and in Settings both reach the core and the screen follows the core's answer.
- [ ] While you speak: the pet listens, the meter moves, the partial transcript updates in place and becomes final.
- [ ] While Glide speaks: the pet talks. `Interrupt` cuts the speech and the meter drops.
- [ ] Mute and Unmute change the icon and the core's mic.
- [ ] Type a request and press Return: it is sent once, the field empties, the pet shows typing while text is in the field.
- [ ] `Stop` ends a running task and its speech.

**Approvals and tasks**
- [ ] An approval request shows the exact command, kind and countdown; the menu bar icon changes. `Approve once` runs that command once. A second similar command asks again.
- [ ] `Deny` does not run it. Letting it expire does not run it, and the card disappears.
- [ ] Quit the app while an approval is open: the core treats it as denied.
- [ ] A task shows "attempted, not yet checked" before "checked". An unconfirmed result is flagged.
- [ ] Force an unknown-outcome write (`reconcile_required`): the app warns, the pet asks a question, and nothing is repeated.
- [ ] Task end: happy for about four seconds on success, sad on failure.

**Providers and settings**
- [ ] Settings > Providers lists the core's roles and slots exactly as `glide.toml` names them, with "VARIABLE is set / is not set" and no key value anywhere.
- [ ] Break a key on purpose: a provider switch appears under "Provider switches" with the core's reason.
- [ ] Pin a role and unpin it: the core's pin changes.
- [ ] Change silence length: the core's value changes within its allowed range; an out-of-range change shows the core's refusal.
- [ ] Change a setting from the core's own CLI at the same time: the app picks up the push, and a stale change is refused and re-read.
- [ ] Kill the connection during a settings change: the app says the outcome is unknown, shows the core's current value, and does not repeat the change.

**Privacy**
- [ ] With `record_content` off, the core stores no utterances, typed text or captured content, and the app shows no "Recording content" mark.
- [ ] Turning it on shows the mark. Turning it off removes it.
- [ ] The app writes no transcript, summary or command to disk (check `~/Library` for new Glide files after a session: only the socket if you chose that default).

**Interface**
- [ ] Settings opens and takes focus from the menu bar window (an accessory app can be fussy about this).
- [ ] Reduce Motion makes the pet still. VoiceOver reads the status and the buttons.
- [ ] Light and dark mode both read well.
