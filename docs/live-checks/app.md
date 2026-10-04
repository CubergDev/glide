# The app and the core together: live verification checklist

Nothing below can be proven offline. The tests use socket pairs, fake assistants and a fake voice stack; they show the protocol and the
safety rules are right, not that the SwiftUI app, the microphone or macOS behave. Do these once on the Mac you will use. A failed box
is a bug to report.

**What takes over your Mac, and when.** Starting the server (`glide app-server`) takes over nothing: it listens on a socket and waits.
Launching the app puts an icon in the menu bar. A computer task looks at your screen (and clicks and types, if you enabled that) only
after YOU press `Approve once` on a card, and only for that one task. The microphone opens only when you switch hands-free on. Each of
those is your action; none happens by itself.

## 0. Before you start

You need the core's dependencies (`uv sync`), a `glide.toml` (`cp glide.toml.example glide.toml`, then set the environment variables it
names; `glide doctor` shows what can be reached), and the Swift app from the `consolidation/app-swiftui` branch (`app/`). If that
branch is not merged into yours, check it out in a second worktree and use its `app/` directory below. Needs a Swift 6 toolchain and
macOS 14 or later.

```sh
cd app && swift build            # compiles only; nothing runs
```

## 1. Start the core

Terminal 1, in the repository:

```sh
uv run glide app-server
```

Expected: one line, `glide app-server: listening on /Users/<you>/Library/Application Support/Glide/glide.sock (Ctrl-C to stop)`.
(`--socket PATH` or `GLIDE_SOCKET` changes it; use a short path, a Unix socket path is limited to 103 bytes.)

- [ ] In terminal 2: `ls -ld ~/Library/Application\ Support/Glide` shows `drwx------` and you as owner, and
      `ls -l ~/Library/Application\ Support/Glide/glide.sock` shows `srw-------` and you as owner.
- [ ] `lsof -i -P | grep -i glide` shows nothing: there is no network listener.
- [ ] Start a second `uv run glide app-server`. Expected: it refuses ("a core is already listening on this socket") and exits 2; the
      first keeps running.
- [ ] Make the directory open (`chmod 755 ...`) and start again: it refuses ("must be mode 0700"); restore with `chmod 700`.
- [ ] Replace the socket with a symlink (`ln -s /tmp/x ...glide.sock` after stopping the core): it refuses and leaves the link alone.
      Remove the link afterwards.

## 2. Protocol by hand (optional, 1 minute)

With the core running, in terminal 2 (replace the path if you changed it). Each line you type is one message:

```sh
nc -U ~/Library/Application\ Support/Glide/glide.sock
{"v":1,"type":"hello","id":"a1","data":{"protocol":1,"client":"nc","client_version":"0"}}
```

- [ ] The core answers a `hello` line with `"recording_content":false`, then a `state` line (`idle`). Nothing before your hello.
- [ ] Send `{"v":1,"type":"settings_get","id":"a2"}`: a `settings` line whose role, slot, provider and model names are exactly the ones in
      your `glide.toml`, `key_env` names (not values), and `key_present` true or false. **No key value anywhere.** Search the output for
      the first characters of one of your keys: it must not be there.
- [ ] Send `{"v":1,"type":"hello","data":{"protocol":2}}` on a new connection: `error` with `"fatal":true`, then the connection closes.
- [ ] Send `not json`: an `error` with code `malformed` that does not repeat what you sent; the connection stays.

## 3. Start the app

Terminal 2 (or the same path as in step 1 through `--socket`/`GLIDE_SOCKET`):

```sh
cd app && .build/debug/GlideApp
```

- [ ] A menu bar icon appears and no Dock icon. The window says "Connected" (the core terminal shows no error).
- [ ] Quit the core with Ctrl-C: the app says "Not connected" and the pet sleeps. `ls` the socket path: the file is gone (clean shutdown).
      Start the core again: the app reconnects by itself and loads the settings.
- [ ] Kill the core hard (`kill -9`, find it with `pgrep -f "glide app-server"`), then start it again: it starts (a dead leftover socket of
      yours is removed) and the app reconnects.

## 4. Typing, and what the app is told

- [ ] Type a plain question ("what is two plus two") and press Return. An answer arrives.
- [ ] With recording off (the default), the app shows that something was said and how long it was, but **not the words**, in your line or
      in Glide's. The window does not show a "Recording content" mark. (If the app shows the words anyway, that is a bug on one of the
      two sides: look at the raw lines with `nc` as in step 2.)
- [ ] Settings > Privacy: switch "Record content" on. The mark appears, and new transcript lines now carry words. Switch it off: the
      mark goes and the words stop. (This lasts until the core exits; restarting starts with it off. `glide app-server --record-content`
      starts with it on and prints that.)
- [ ] Press Interrupt while Glide is speaking a long answer (needs a speaker: switch hands-free on first, step 6): the voice stops.

## 5. Approvals: the computer is never taken without a click

Use something harmless: "open the Notes app and tell me what's on screen" or any request that routes to the computer.

- [ ] A card appears with the exact command (`Look at this Mac's screen, without clicking or typing, to: <your goal>`), kind `screen`, and a
      countdown (30 s). The menu bar icon changes. **Nothing has looked at the screen yet.**
- [ ] Press `Deny`. The card goes. Glide says nothing was done (a warning line: "the task was not approved"). No screenshot permission
      prompt, no window changes.
- [ ] Ask again and let the countdown run out: the card disappears, nothing runs.
- [ ] Ask again and quit the app while the card is open: the core's terminal shows no task starting. Relaunch the app: no card is waiting
      and nothing runs.
- [ ] Ask again and press `Stop` while the card is open: the card goes; nothing runs.
- [ ] Ask again and press `Approve once`. The task runs as a dry run (it looks, it does not click or type) and the app shows
      "attempted, not yet checked" and "checked" steps, then the end. macOS may ask for Screen Recording permission for the terminal
      running the core: that is the core's permission, not the app's.
- [ ] Ask the same thing a second time: it asks again. One approval is one task.
- [ ] Settings > General: allow screen control, ask again. The card now says kind `input` and `Control this Mac (clicks and typing) to: ...`.
      Deny it. (Approve only a harmless request, and only if you are ready to watch it: it will click and type.)
- [ ] While an approved task runs, press `Stop`: it stops before its next action. If the action in flight cannot be confirmed, the app
      shows the "unknown outcome" warning and the pet asks a question; nothing is repeated.

## 6. Hands-free voice

- [ ] Settings > Voice: switch hands-free on. macOS asks for microphone access the first time (the core's process, not the app). The state
      becomes "listening". Deny the permission once and check that the setting stays off and the core's reason is shown, not a hang.
- [ ] Speak a question: the pet listens, the transcript line appears (redacted, with a length), Glide answers aloud. A typed request now
      also gets a spoken answer.
- [ ] Mute in the app: the microphone indicator in the menu bar of macOS goes out. Unmute: it returns.
- [ ] Change the silence length or headset while hands-free is on: the loop restarts with the new value (a brief gap) and keeps listening.
- [ ] Switch hands-free off: the microphone indicator goes out and state is "idle".
- [ ] Known gap: talking over Glide shows its speech as "finished", not "interrupted" in the app.

## 7. Providers and settings

- [ ] Settings > Providers lists the roles and slots exactly as `glide.toml` names them.
- [ ] Unset one key variable in the shell that runs the core, restart the core: that slot shows "not set" and "skipped".
- [ ] Make a key wrong on purpose and ask something: a provider switch appears under "Provider switches" with the core's reason, and the
      same switch is printed in the core's terminal. No key appears in either.
- [ ] Pin a role to a slot in the app, then unpin. The Providers page shows the pin appear and go, and a request for that role then goes to
      the pinned slot first (a provider switch line names it if it fails). Pins live in the running core only.
- [ ] Change silence length to 250 and to 5000: the first is taken, the second is refused with the core's range and nothing else in that
      change is applied.
- [ ] Open the file `glide.toml` afterwards: it is unchanged (the app never writes it).

## 8. Files and logs

- [ ] With recording off, run `ls ~/Library/Application\ Support/Glide`: only the socket. The app wrote nothing to disk.
- [ ] A task's run folder (`runs/<timestamp>/run.json`) has counts and an outcome but not your goal or what was on the screen (unless
      you turned recording on).
- [ ] The core's terminal shows no transcript, goal, URL or answer text.

## What to report

Anything that happened without a click or a switch of yours; a card that did not match what ran; any word of a transcript, goal or
answer in the app, the terminal or a run folder while recording was off; a key anywhere; a task that started after `Deny`, an expiry, a
`Stop` or a disconnect.
