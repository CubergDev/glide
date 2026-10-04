# Recorded demo: script

A 5 minute 45 second screen recording of Glide, in seven scenes. Written from the code of branch `consolidation/demo`
(the integration branch plus these documents). Read [DEMO_PREFLIGHT.md](DEMO_PREFLIGHT.md) first and run
`scripts/demo_preflight.py`; read [DEMO_RECORDING.md](DEMO_RECORDING.md) for how to record and what to check before sharing.

**Honesty first.** [README.md](../README.md) says "nothing in this repository has been proven live", and that is still true.
Every scene below has only ever run against fakes. Treat the first full pass as a rehearsal that may find bugs, not a
recording. Each scene has a fallback that shows something real without the risky part, and the script says out loud what
Glide cannot do. Do not cut the limits out of the final video.

**Approval.** [AGENTS.md](../AGENTS.md): every command marked **TAKES OVER THE MACHINE** needs a yes for that exact command,
once. The full list is at the end ("Approval list"). Nobody but the person doing the demo runs these.

## Facts that shape the script

The code differs from some pages of the docs (README, ARCHITECTURE, LIVE_CHECKS still say the pet, echo cancellation and the
app server are "not in this tree"; they are in `glide/`). What the code does, checked for this script:

- **Engine.** `--engine structured` exists on `glide computer` only. There is no engine setting in `glide.toml`, in
  `ask`, `chat`, `listen`, `voice`, the pet or the app server, and the SwiftUI app has no engine badge. Tasks started by
  voice or chat run the legacy loop, which moves the real mouse and cannot verify through a browser. The browser scenes
  therefore use `glide computer`.
- **Confirmation.** Only `glide voice --act` (and the app's approval card) has a confirmation step. For `ask`, `chat`,
  `listen` and `computer`, the dry run is the default and "confirming" means running again with `--act`
  (or `/act` in `chat`).
- **SwiftUI app.** `app/` is **not on this branch**. It is on `consolidation/app-swiftui`, which has never been launched.
  The core's side (`glide app-server`) is here.
- **Point-to-ask** is `python -m glide.assistant.point_cli`, not a `glide` subcommand.
- **Run commands** as `uv run --no-sync ...` after one `uv sync --all-extras` (plain `uv run` re-syncs the environment, and
  may drop extras you installed one at a time).

## Staging (once, before the take; nothing here takes over the machine)

Everything lives outside the repository, in `~/glide-demo`:

```sh
mkdir -p ~/glide-demo/site ~/glide-demo/runs
cd ~/glide-demo/site
cat > index.html <<'HTML'
<!doctype html><title>Glide demo shop</title><h1>Glide demo shop</h1>
<p>Harmless test pages served from this Mac.</p>
<a href="report.html">Open the report</a> | <a href="ferry.html">Harbor Ferry</a> | <a href="tram.html">Hill Tram</a>
<p><button id="b">Ask me about this button</button></p>
HTML
echo '<!doctype html><title>Report</title><h1>Quarterly report</h1><p>The demo report page loaded.</p>' > report.html
echo '<!doctype html><title>Harbor Ferry</title><h1>Harbor Ferry</h1><p>The Harbor Ferry carries 120 passengers and the adult fare is 4 coins.</p>' > ferry.html
echo '<!doctype html><title>Hill Tram</title><h1>Hill Tram</h1><p>The Hill Tram carries 80 passengers and the adult fare is 3 coins.</p>' > tram.html
```

The config: copy your `glide.toml` to `~/glide-demo/glide-demo.toml` and append what the scenes need (the model ids are yours;
nothing here is a claim about a model):

```toml
[speech]
headset = true            # scene b with a headset. For speakers: headset = false and the aec extra (see scene b)
merge_window_s = 0.0

[browser]
provider = "cdp"
target = "new"
search_url = "http://127.0.0.1:8765/"

[browser.cdp]
endpoint = "http://127.0.0.1:9222"

[research]
calls = 12
```

Every command below is run from the repository folder (it holds `.env`) and starts with
`G="uv run --no-sync glide --config $HOME/glide-demo/glide-demo.toml"` (so `$G ask ...`).

Start the web server and the browser (both take over the machine, see the approval list), each in its own terminal:

```sh
# TAKES OVER THE MACHINE: opens a loopback port
cd ~/glide-demo/site && python3 -m http.server 8765 --bind 127.0.0.1
# TAKES OVER THE MACHINE: launches a browser with remote debugging on loopback only (the exact line is printed by the preflight)
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --remote-debugging-port=9222 --remote-debugging-address=127.0.0.1 --user-data-dir="$HOME/glide-demo-profile" --no-first-run about:blank
# TAKES OVER THE MACHINE: connects to that port. Expect JSON with a "Browser" field.
curl -s http://127.0.0.1:9222/json/version
```

Put the browser, the terminal and (for scene f) the menu bar on screen at once, at the resolution in DEMO_RECORDING.md. Large font in
the terminal. Say the on-screen text aloud when it matters; with a headset the recording will not contain Glide's voice.

## Time budget

| Scene | What | Start | Length |
|---|---|---|---|
| a | A fast answer, typed and spoken, with the provider chain and a visible fallback | 0:00 | 0:45 |
| b | Talking over Glide, and a spoken stop | 0:45 | 0:45 |
| c | A browser task: dry run, then for real, verified | 1:30 | 1:15 |
| d | Research with cited sources (structured engine) | 2:45 | 1:00 |
| e | Point and ask (read only) | 3:45 | 0:30 |
| f | The menu-bar app: state, an approval request | 4:15 | 0:30 |
| g | The safety story | 4:45 | 1:00 |
| | Total | | 5:45 |

If a scene is going wrong, stop it (say "stop", Ctrl-C), say one sentence about what happened, switch to its fallback, and keep the
clock. A visible failure handled safely is part of the story.

---

## Scene a: a fast answer, a visible chain, a visible fallback (0:00, 45 s)

**Show.** Answers come from a chain of swappable providers; a failure is printed, never hidden. **Status: never run live.**

**Commands**

1. `$G status` (no machine use). Expect: `config: <path>`, then each role (`llm.fast`, `llm.smart`, `stt`, `tts`, `classifier`) with
   its slots, `calls 0 failures 0`, and any `skipped: <VARIABLE> is not set`. 10 s.
2. `$G ask --timings "In two sentences, why is the sky blue?"` (**TAKES OVER THE MACHINE** only if the router sends it to the
   computer; that is why it is on the approval list). Expect: the answer on stdout, then on stderr
   `timings: route=..., first_word=..., first_audio=...` (the names come from the reply; read the numbers aloud, do not promise
   them). 10 s.
3. The fallback, forced on purpose: `OPENROUTER_API_KEY=invalid-demo-key $G ask --timings "Name three primary colours."`
   (use the variable of the **first** `llm.fast` slot in your file; the preflight lists it). The key is set but wrong, so the first
   slot fails with `auth` and the chain moves on. Expect on stderr, before or with the answer:
   `fallback: llm.fast <first slot> -> <second slot> (auth: <short reason, never a key>)`, then the answer. 15 s. A key that is
   merely unset only skips the slot when the file is loaded (a warning, no switch), so it does not make this line.
4. Hands-free, if the speech extra and keys are ready (**TAKES OVER THE MACHINE**: microphone and speaker): `$G voice`.
   Expect `glide voice: listening. Just talk; say "stop" to stop a task, press Ctrl-C to leave.` Say: "What is the capital of
   France?" Expect `you said: ...` and a spoken answer. 10 s. Ctrl-C to leave.

**Say.** "Every model, speech and vision vendor is configuration. When a provider fails, Glide tells you and uses the next one.
Nothing is silent." Do not claim a latency; read what `--timings` printed.

**Expected result.** One fallback line, one correct answer, nothing printed that looks like a key.

**If it fails.** No answer at all: run `$G doctor` (no `--live`) and read the `skipped` rows aloud. No `fallback:` line:
the first slot may not be the variable you broke, or `hedge_after_s` raced the second slot (a `racing:` line appears instead,
which is also a visible switch). Voice does not start: `glide voice: ... (the speech extra ...)` exits 2; skip step 4.

**Safe fallback.** Steps 1 to 3 need no microphone and no screen. If keys are missing too, show `$G doctor` and say what it
would show with keys.

**Budget.** 0:00 to 0:45: status 0:10, ask 0:20, fallback 0:35, voice 0:45 (cut voice first if running late).

---

## Scene b: talk over Glide, and a spoken stop (0:45, 45 s)

**Show.** You can interrupt it, by voice. **Status: never run live; the echo cancellation was proved only on simulated rooms.**

**Setup.** With a headset: `[speech] headset = true` (above). With speakers: `headset = false`, `echo_canceller = "auto"` and
the `aec` extra installed (`uv sync --all-extras`); start-up then prints which canceller was taken, or
`no echo canceller is available, so speaker mode is half duplex (...)`. Without the extra, speaker mode may be half duplex and
this scene will show nothing happening: use the headset. The preflight says which case you are in.

**Commands** (**TAKES OVER THE MACHINE**: microphone and speaker): `$G voice`

1. Say: "Explain in many sentences how a refrigerator works." While it speaks (after about three seconds) say, clearly:
   "Wait, what time is it?" Expect: the voice cuts within about half a second, `you said: ...` shows your new question
   starting at "Wait", and it is answered.
2. Ask the refrigerator question again. While it speaks say: "Stop." Expect: it stops and nothing more is said. (Stop phrases are
   matched before any model is called, so this works with no network.)

**Say.** "Talking over it cuts the voice and takes the new question. A spoken stop is matched before any model is called."
Then, honestly: "Cancelling is not undoing: an action already sent stays done."

**If it fails.** It answers itself or interrupts itself: speakers are the problem. Switch to the headset and re-take; do not tune
live on camera. It does not cut when you talk over it: say "stop" after it finishes, then use the fallback. The microphone is
denied: it exits 2 with a message.

**Safe fallback.** In text: `$G chat`, ask for the long answer, type `stop` as soon as text appears. Expect `stopped`, no `fallback:`
line, no further text. `/quit` to leave. (Takes over nothing without `--speak` or `--act`.)

**Budget.** 0:45 to 1:30: question 0:10, talk over 0:20, spoken stop 0:35, limits sentence 0:45.

---

## Scene c: a browser task, dry run first, then for real (1:30, 75 s)

**Show.** Nothing clicks until you ask for it, and an action is only "done" when a fresh look at the page shows the effect.
**Status: never run live. Only the CDP provider was ever qualified, and that was a report.**

**Setup.** The staging web server and the browser from "Staging" are running; the terminal has Accessibility permission (`--act`
checks it first).

**Commands** (all **TAKES OVER THE MACHINE**: they attach to your browser; the second also clicks):

1. The dry run:
   `GLIDE_BROWSER_PROVIDER=cdp GLIDE_BROWSER_ENDPOINT=http://127.0.0.1:9222 GLIDE_BROWSER_TARGET=new $G computer "open http://127.0.0.1:8765/ and click the link Open the report" --engine structured --out $HOME/glide-demo/runs/c1`
   Expect: `writer: <provider>`, `browser: CDP · http://127.0.0.1:9222 · target new`, then
   `dry run: nothing was done; it would <the first move>`. Nothing in the browser changes beyond the tab Glide opens. Exit status 0.
2. Say: "That was a preview. Now the same request for real." Then the same command with `--act` and `--out .../c2`.
   Expect progress lines from the run, the "Report" page opening in the browser, and the run ending with Glide's own sentence that
   it checked the page. In `run.json` (opened in step 3) `outcome` is `done` and `steps_taken` is above 0. Exit status 0 (`echo $?`).
3. Verify out loud: `cat $HOME/glide-demo/runs/c2/run.json | head -40` (content recording is off, so there is no goal, no
   page text and no full URL in it; point at that).

**Spoken variant (optional, riskier, not on the clock).** `$G voice --act`, say "open the calendar". The loop runs a dry run and
the screen shows `To do this for real, say "confirm and run it" within 10 seconds.` Say exactly `confirm and run it`.
Anything else, silence, a stop, or talking over Glide gives `Not confirmed, so nothing was done.` This runs the legacy loop on your
real screen, not through the browser provider: use a throwaway window.

**Say.** "Without --act, Glide only describes its first move. With --act it clicks, and it counts a step as done only after it sees
the effect. Stop any time: say stop, press Ctrl-C, or throw the mouse into the top-left corner."

**If it fails.** `Cannot connect to cdp at http://127.0.0.1:9222`: the browser is not up; start it and retry (nothing was done).
`this terminal lacks Accessibility permission`: grant it and retry. Outcome `blocked`, `unsupported` or a stuck run: say it plainly,
do not retry a step that changed the page; close the tab. A click whose outcome is unknown is reported as "completion unknown";
never replay it. Mid-run: Ctrl-C (exit 130).

**Safe fallback.** Run only step 1 (the dry run), say that the live step is the one not yet proven, and move on. A second
fallback that does not use the browser: `$G computer "open the calendar" --image <a screenshot you saved earlier>` replays a saved
capture, never acts and does not capture the screen (it still sends the capture to your providers, so use a harmless image).

**Budget.** 1:30 to 2:45: dry run 0:25, narration 0:10, live run 0:25, run.json 0:15.

---

## Scene d: research with sources that were actually read (2:45, 60 s)

**Show.** Sources are opened and read in this run, and the answer's quotes must come from them. This is the opt-in
structured engine (`--engine structured`). **Status: never run live. The research pipeline has never been validated live.**

**Commands** (**TAKES OVER THE MACHINE**: browser; `--act` is how the live checks run it):

`GLIDE_BROWSER_PROVIDER=cdp GLIDE_BROWSER_ENDPOINT=http://127.0.0.1:9222 GLIDE_BROWSER_TARGET=new $G computer "Compare the Harbor Ferry at http://127.0.0.1:8765/ferry.html and the Hill Tram at http://127.0.0.1:8765/tram.html: which carries more people and which fare is lower? Cite both pages." --engine structured --act --out $HOME/glide-demo/runs/d1`

Expect: the browser opens each of the two pages; the answer says the ferry carries more (120 against 80) and the tram's fare is
lower (3 against 4), cites `[1]` and `[2]`, and lists exactly the two addresses. Both quoted sentences are visibly on the pages
(say: "Check the quotes against the page"). The time and model-call count are in `run.json` (`research`). A claim that is not on
a page read in this run fails the run before it is spoken.

**Say.** "The model plans and writes; the browser only reads and navigates; every citation has to quote a page it opened in this
run. Page text is data, never instructions." Be careful not to promise accuracy: "I am showing one run, not a benchmark."

**If it fails.** `needs evidence` or a budget message: say so, that is the safe outcome. A login wall or a refusal is one plain
sentence. A long wait over about a minute may be the known provider disconnect (LIVE_CHECKS L3); stop with Ctrl-C and use the
fallback. A wrong or uncited answer: show it and say that is why the checks exist.

**Safe fallback.** The reasoning route needs no browser: `$G computer "Summarize this text: Glide is an assistant that can also drive the Mac. It asks before it acts." --engine structured`
(**TAKES OVER THE MACHINE** only because the command's own rule is to ask; the route is one model call with no browser; expect no tab opens
and `research.route = "reason"` in `run.json`). If keys fail too, show the scene-c dry run again.

**Budget.** 2:45 to 3:45: command 0:05, reading and answer 0:35, quotes check 0:15, limits 0:05.

---

## Scene e: point and ask, read only (3:45, 30 s)

**Show.** Ask about whatever is under the pointer; the feature has no way to click or type. **Status: never run live.**

**Commands** (**TAKES OVER THE MACHINE**: reads the accessibility text of the item under the pointer; needs Accessibility):

1. The local preview, nothing sent: `uv run --no-sync python -m glide.assistant.point_cli --delay 5 --config $HOME/glide-demo/glide-demo.toml`
   Expect on stderr `Local preview: reads the pointed item's accessibility text. Nothing is sent to a model.` and
   `Position the pointer. Reading in 5 seconds; Ctrl-C cancels.` Move the pointer over the "Ask me about this button" button in
   the browser. Expect its role and label on stdout. 15 s.
2. With a model, text only: add `--allow-model "What does this button probably do?"` and keep the pointer on the button. Expect on stderr
   `Answer provider: <slot>` and `This request shares your question, the pointed item's text.`, then a short answer on stdout. 15 s.
   (`--with-image` also shares a crop of the screen and needs Screen Recording; leave it out.)

**Say.** "Read only. Without --allow-model nothing leaves the machine; with it, it tells you what it shares first."

**If it fails.** Exit 2 `Could not read this point. Check the desktop permissions.` (Accessibility); exit 2 for a protected
(password) field is correct behaviour. Exit 3 means the provider could not be reached. Ctrl-C exits 130.

**Safe fallback.** Step 1 only. Or the pet: see scene f.

**Budget.** 3:45 to 4:15.

---

## Scene f: the menu-bar app (4:15, 30 s)

**Show.** State, a pet, an approval card ("Approve once" or "Deny", no "always"). **Status: never run live; may not build.**
The app has no engine badge (nothing on the branch implements one); do not say it has.

**Setup (not on the clock).** `app/` is on `consolidation/app-swiftui`, not here. In a separate clone of that branch, with Swift 6 and
macOS 14 or later:

```sh
cd app && swift build            # compiles only. Not attempted for this document.
./scripts/test.sh                # its own tests; no approval needed
```

**Commands** (**TAKES OVER THE MACHINE**: a menu-bar item, and a socket):

1. Terminal 1, the core: `uv run --no-sync glide --config $HOME/glide-demo/glide-demo.toml app-server --socket /tmp/glide-demo.sock`
   Expect `glide app-server: listening on /tmp/glide-demo.sock (Ctrl-C to stop)` (the socket is mode 0600, no network port).
2. Terminal 2, from the app clone: `.build/debug/GlideApp --socket /tmp/glide-demo.sock`. Expect a menu-bar icon, no Dock icon, "Connected".
3. Type a plain question in the app: an answer arrives; with recording off the transcript shows that something was said and its
   length, not the words. Type "open the calendar": an approval card appears with the exact command and a countdown. Click **Deny**.
   Expect nothing happens on screen. 

**Say.** "The app is a client; the core decides. Every computer task, even a dry run, asks first, once."

**If it fails.** The build fails or the app crashes: use the fallback, say it has never been launched.

**Safe fallback A (core only, no app).** `nc -U /tmp/glide-demo.sock`, then send
`{"v":1,"type":"hello","id":"a1","data":{"protocol":1,"client":"nc","client_version":"0"}}` and
`{"v":1,"type":"settings_get","id":"a2"}`. Expect `"recording_content":false`, then settings that name key variables and whether each
is set, never a value. **Safe fallback B (the older window).** `uv run --extra ui python -m glide.ui --workspace $HOME/glide-demo`
(`ui` extra): a raccoon window, settings with "Allow computer actions" and "Record task and page content" unticked.

**Budget.** 4:15 to 4:45: core 0:05, app 0:10, question 0:05, approval and Deny 0:10.

---

## Scene g: the safety story (4:45, 60 s)

**Show.** Why it is reasonable to let this near your Mac. No takeover commands in this scene; it reads and runs offline tests.
**Status: the facts below are from code and offline tests; none is a live proof.**

1. **Recording is off by default** (0:15). `grep -r "Harbor" $HOME/glide-demo/runs/d1 || echo "not recorded"` prints `not recorded`:
   the run folder holds counts, timings and the outcome, not the goal, page text or full URLs. (`--record-content` turns it on; it
   is the only thing that does.) (Run files are 0600 in 0700 folders when content is recorded.)
2. **Approvals default to no** (0:15). Say the three: without `--act` nothing clicks; the app's approval card is "Approve once", no
   "always"; `glide voice --act` needs the confirm phrase, and any other answer is a no. A webhook-triggered desktop run needs a yes
   on the worker's terminal every time. Show [AGENTS.md](../AGENTS.md)'s "Never take over the machine without approval".
3. **Offline tests and guards** (0:15). `uv run --no-sync pytest -q tests/test_cli_tree.py` (about 15 s, offline). Show the head of
   `tests/conftest.py`: it refuses input events, AppleScript, screen capture, accessibility, `subprocess.Popen` and non-loopback
   network. Say plainly that these use fakes: "they show the logic, not that a vendor, a microphone or macOS behaves".
4. **The review trail** (0:15). `git log --oneline | head -12` shows the merges of the review fixes; `ls docs/reviews docs/history`;
   `docs/DECISIONS.md` (D0-D18, with what is not done, such as D9: two routers still exist); `docs/LIVE_CHECKS.md` ("nothing in this page
   was run by the agent that wrote it").

**Say last.** "None of this has been proven live yet. That is what the next pass is for."

**If it fails.** `pytest` slow or red: show the file instead and say the number from the last full run (do not quote a number you
did not see). `grep` finds the word: recording was on, stop and say so; do not publish that take.

**Safe fallback.** The documents alone: README "Safety model" and `docs/LIVE_CHECKS.md` index.

---

## What is not ready (say it, do not hide it)

- Nothing was ever run live: voice, browsers, research, echo cancellation, point-to-ask, the pet and the app server run only against
  fakes. The SwiftUI app has never been launched and is not on this branch.
- Speaker-mode barge-in needs the `aec` extra and was proved only on simulated rooms. A headset is the reliable case.
- No engine setting for ask, chat, voice, the pet or the app; no engine badge. D9 (two routers) is open.
- Spoken and chat tasks use the legacy loop, which moves the real mouse and verifies from the screen, not the page.
- The full offline suite is not green on this branch. Measured while writing this: 16 failed, 5622 passed, 17 skipped (about 5.5 minutes),
  plus `tests/test_webhooks_worker_mutants.py` failing to import (`Control` is not in `glide.webhooks.worker`). The failures are in
  `tests/test_errors_mutants.py` and `tests/test_voice_gate_mutants.py` among others. Do not run the whole suite on camera.
- The model ids in `glide.toml.example` are unverified starting points; the research route may overstate; a request still waiting
  for its first byte cannot be cancelled from outside; the roughly 60 s provider disconnect is unexplained.

## Approval list (ask once, before the take)

These are all the commands the demo runs that take over the machine, in order. Replace nothing; a yes covers a command once.

1. `cd ~/glide-demo/site && python3 -m http.server 8765 --bind 127.0.0.1`
2. `"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --remote-debugging-port=9222 --remote-debugging-address=127.0.0.1 --user-data-dir="$HOME/glide-demo-profile" --no-first-run about:blank`
3. `curl -s http://127.0.0.1:9222/json/version`
4. `$G ask --timings "In two sentences, why is the sky blue?"` and `OPENROUTER_API_KEY=invalid-demo-key $G ask --timings "Name three primary colours."`
   (the second sends a deliberately wrong key to one real provider; no money is spent on a refused request)
5. `$G voice` (microphone and speaker), once for scene a and once for scene b
6. `GLIDE_BROWSER_PROVIDER=cdp GLIDE_BROWSER_ENDPOINT=http://127.0.0.1:9222 GLIDE_BROWSER_TARGET=new $G computer "open http://127.0.0.1:8765/ and click the link Open the report" --engine structured --out $HOME/glide-demo/runs/c1`
7. The same with `--act` and `--out $HOME/glide-demo/runs/c2`
8. The scene d command (with `--act`, `--out .../d1`) and, as its fallback, `$G computer "Summarize this text: ..." --engine structured`
9. `uv run --no-sync python -m glide.assistant.point_cli --delay 5 --config $HOME/glide-demo/glide-demo.toml`, and the same with `--allow-model "What does this button probably do?"`
10. `uv run --no-sync glide --config $HOME/glide-demo/glide-demo.toml app-server --socket /tmp/glide-demo.sock`, then `nc -U /tmp/glide-demo.sock`,
    then (if the app was built) `.build/debug/GlideApp --socket /tmp/glide-demo.sock`
11. Optional: `$G voice --act`; `uv run --extra ui python -m glide.ui --workspace $HOME/glide-demo`; `$G doctor --live` (spends tokens)
12. The screen and audio recording itself: see DEMO_RECORDING.md (`screencapture -v -g -x ...` or QuickTime Player, and `ffmpeg` on the saved file only)

`$G` is `uv run --no-sync glide --config $HOME/glide-demo/glide-demo.toml`. Commands that need no approval: `$G status`, `$G doctor`
(without `--live`), `pytest`, `git`, `ls`, `grep`, `cat` on files in `~/glide-demo`, `scripts/demo_preflight.py`.
