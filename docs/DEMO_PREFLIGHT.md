# Demo preflight

Do this before you record ([DEMO.md](DEMO.md)). Part 1 is an offline script that only reads. Part 2 is a checklist of what only
you can do. Nothing in this page was run live by whoever wrote it.

## 1. The offline check

```sh
uv sync --all-extras
uv run --no-sync python scripts/demo_preflight.py --config ~/glide-demo/glide-demo.toml
```

It reads, and nothing else:

- the Python version, which extras are installed (`speech`, `aec`, `ui`; looked up, not imported), which of `uv`, `swift`, `ffmpeg`,
  `node` are on `PATH` (`which` only), whether `glide.toml` and `.env` exist, and whether `app/` (the SwiftUI app) is on this branch;
- `glide.toml` through Glide's own loader, so a file the loader refuses is refused here with the same message;
- every provider slot of every role: ready, or skipped with the **name** of the variable to set;
- which key variables are set. **Names only: no value is printed, ever.** It also reads the variable names in `./.env` (or
  `--env-file`), as `glide` does; the values are dropped on the spot;
- the `[speech]` settings (headset, echo canceller, confirm phrase) and what they mean for scene b;
- the `[browser]` settings as Glide resolves them, whether the endpoint is loopback, and `search_url`.

It **never connects** to anything: not the browser's debugging port, not a provider, not a socket. It never starts a program and never
touches the screen, microphone or a macOS permission. For the browser it prints two commands for **you** to run (both take over this
machine, so they are on the approval list in DEMO.md): the line that starts Chrome or Brave with remote debugging on loopback and a
throwaway profile, and `curl -s http://127.0.0.1:9222/json/version`, which should answer with JSON that has a `Browser` field.

Exit status 0 means nothing blocks the demo offline (warnings allowed); 1 means a `[FAIL]` line must be fixed first. A passing run is
not a live check: it says the setup is plausible, not that a provider, browser or microphone works. Keys: the check cannot tell a
wrong key from a right one. `glide doctor --live` can (it spends a few tokens, and needs your yes).

Reading the output: `[ok]`, `[info]`, `[warn]` (the demo will be weaker or a scene will fall back), `[FAIL]` (a scene cannot run).

## 2. Manual checklist

Tick each one yourself. Do it on the Mac and the user account you will record on.

### Permissions (System Settings > Privacy & Security)

The terminal you run Glide from needs these. macOS asks on first use, which on camera is a bad surprise: trigger the prompts earlier
with throwaway runs (each such run needs your yes, per AGENTS.md).

- [ ] **Accessibility**: the terminal. Needed for `--act` (Glide refuses with "this terminal lacks Accessibility permission" without it)
      and for point-to-ask.
- [ ] **Screen Recording**: the terminal, only for the legacy loop, `glide inspect`, a point crop, and the screen recorder itself. The
      recorder (QuickTime Player, or the terminal running `screencapture`) needs it too, and macOS may ask you to quit and reopen the app.
- [ ] **Microphone**: the terminal, for `glide voice` and `glide listen`. Denying it makes `glide voice` exit with status 2 and a message.
- [ ] After granting, quit and reopen the terminal once; check each toggle is on.

### Audio

- [ ] **Headset (recommended for scene b)**: wired or a Bluetooth one you have tested. A Bluetooth headset may change its sample
      rate when the microphone opens, which Glide has never been checked against; if answers are garbled, use a wired one.
      Set `[speech] headset = true`.
- [ ] **Or speakers** (scene b, the harder case): the `aec` extra installed (the script says), `headset = false`,
      `echo_canceller = "auto"`. At start Glide prints which canceller it took. Volume at a normal level and the laptop still.
      Expect it to fail more often than the headset.
- [ ] Input device is the one you want (System Settings > Sound). The recording will have your microphone, not Glide's voice, if you
      use a headset; plan on-screen text for Glide's side.
- [ ] A short test that is yours to do: record ten seconds with the recorder and play it back.

### Screen

- [ ] **Do Not Disturb or a Focus mode on**, notifications off (Notification Center previews off too). A banner with a message
      in the recording is a privacy leak, and it can cover the area Glide is about to click.
- [ ] Menu bar clean: quit apps whose icons show names, hours or counts you do not want shown; hide the clock seconds if you like.
- [ ] Desktop clean, no personal files visible, a neutral wallpaper. Close Mail, Messages, chat apps. Hide the Dock if it shows
      private apps.
- [ ] One display, scaling set to the resolution you will record (see [DEMO_RECORDING.md](DEMO_RECORDING.md)).
- [ ] Terminal: large font, a plain prompt (no hostname, no path with your name if you can avoid it: `PS1='$ '`), the working
      directory the repository folder, the command history out of reach.
- [ ] Screen saver, display sleep and the lock screen off for the length of the take (`caffeinate -d` in a spare terminal needs no
      approval, but it keeps the Mac awake: close it after).
- [ ] Battery above 50 percent or on power.

### Browser

- [ ] A **clean profile**: Glide demo uses `--user-data-dir="$HOME/glide-demo-profile"`, not your everyday profile (recent Chrome ignores
      the debugging port on the default profile, and your everyday profile has your tabs, bookmarks, passwords and sign-ins).
      Start with the profile folder deleted for a fresh take: `rm -rf ~/glide-demo-profile` (it holds only demo data).
- [ ] Sign in to nothing. No extensions, no bookmarks bar, no autofill, no password manager prompt.
- [ ] Exactly one window and one tab (the demo uses `target = "new"`, so Glide opens its own tab).
- [ ] The staging site is served on loopback and the pages exist (`~/glide-demo/site`, see DEMO.md "Staging").
- [ ] The browser's debugging port answers on loopback only: run the `curl` line the script printed. Do not expose the port; the
      flags bind `127.0.0.1`.

### Keys and configuration

- [ ] The key variables for your first `llm.fast`, `llm.smart`, `stt`, `tts` and `classifier` slots are exported in the shell or in
      `.env` (never in `glide.toml`). The script's `keys` lines show which are set.
- [ ] A spare key with a spending limit if you have one. Glide never needs a password.
- [ ] You know which variable belongs to the first `llm.fast` slot (scene a breaks it on purpose with a wrong value on one command).
- [ ] ElevenLabs needs a voice id in `options.voice`; without it the slot is skipped and macOS `say` speaks (fine for a demo, and said so
      by the preflight).
- [ ] Recording of content stays **off** (no `--record-content`, `record_content` not set anywhere). Scene g depends on it.
- [ ] `glide doctor` (no `--live`): every role you need is `ready`. Optional, with your yes: `glide doctor --live` to see that each slot
      answers.

### The engine setting

- [ ] There is **no engine setting in the config, the chat, the voice loop, the pet or the app** on this branch. The structured engine
      is used only through `glide computer ... --engine structured` (scenes c and d). Say it on camera as "opt in per command". Do not
      show or promise an engine toggle or badge.
- [ ] `[browser] provider = "cdp"` in the demo config (the script warns if not), and `search_url` set if a research task names no site.

### The rehearsal

- [ ] Rehearse once, scene by scene, off camera, each command with its own yes. Write down what differed from DEMO.md and fix the script
      before the take rather than during it.
- [ ] Keep a second terminal ready with `Ctrl-C` in mind and know the three stops: say or type `stop`; Ctrl-C; the mouse into the
      top-left corner of the screen.
- [ ] Close the demo when finished: quit the browser, stop the web server (Ctrl-C), delete `/tmp/glide-demo.sock`, `unset` any
      `OPENROUTER_API_KEY=invalid-demo-key` you exported (the DEMO.md form sets it for one command only).
