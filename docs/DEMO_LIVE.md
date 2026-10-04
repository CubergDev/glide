# Live demo: what to run, in what order

Written after real runs on this machine (2026-10-04). Nothing here is claimed beyond what ran. Keys come from your own shell
environment and are never written to a file.

## Setup, once

```bash
cd ~/glide && git checkout main && git pull
uv sync --all-extras
export VERCEL_AI_GATEWAY_KEY=...   # the gateway
export ELEVENLABS_API_KEY=...      # speech
export TYPESAFE_API_KEY=...        # jev, the screen-action classifier
G="uv run --no-sync glide --config $HOME/glide-demo/glide-demo.toml"
$G doctor --live                   # every row should be ok (spends a few tokens)
```

`~/glide-demo/glide-demo.toml` is the config used for the live runs (gateway fast and smart chains, ElevenLabs speech, jev).
Reasoning effort is a per-slot option (`reasoning_effort`) and a dropdown in `glide panel`.

## Scenes that worked live

1. **Fast answer, visible chain.** `$G ask --timings "In two sentences, why is the sky blue?"` answered in about 1.8 s
   (first word 1.5 s). Try it again with a wrong key in one provider's variable to show the fallback line.
2. **jev picks screen actions.** Visible in the browser scene below (`jev-1.13.0` classifier calls at about 0.3 to 0.5 s each).
3. **Browser task with the structured engine.** Start the local page and a loopback-only Chrome:
   ```bash
   cd ~/glide-demo/site && python3 -m http.server 8765 --bind 127.0.0.1 &
   "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --remote-debugging-port=9222 \
     --remote-debugging-address=127.0.0.1 --user-data-dir="$HOME/glide-demo-profile" --no-first-run about:blank &
   GLIDE_BROWSER_PROVIDER=cdp GLIDE_BROWSER_ENDPOINT=http://127.0.0.1:9222 GLIDE_BROWSER_TARGET=new \
     $G computer "open http://127.0.0.1:8765/ and click the link Open the report; success means the address is http://127.0.0.1:8765/report.html" \
     --engine structured --act --out $HOME/glide-demo/runs/take1
   ```
   Say the success condition in the request: the engine verifies from the page, and a vague outcome ("the report page is
   shown") cannot be observed, so the planner may ask a question or re-plan. Dry run first: leave `--act` off.
   Reliability measured: 2 of 3 identical runs finished in about 5 s; one blocked after a re-plan (about 20 s). Rehearse
   it several times and keep the takes that finish.
4. **File planner: plan, preview, approve, undo.** In a folder under your home:
   ```bash
   F="uv run --no-sync python -m glide.files"
   $F plan ~/glide-demo/messy --intent type --out ~/glide-demo/plan.json
   $F preview ~/glide-demo/plan.json          # prints the hash; nothing moves
   $F apply ~/glide-demo/plan.json --approve <hash>
   $F undo ~/.glide/file-runs/<manifest>.json  # put everything back
   ```
   This ran live: 7 files sorted into 5 folders, then restored exactly.
5. **Speech check without a microphone.** `$G doctor --live` shows ElevenLabs speech-to-text and text-to-speech working.
6. **Control panel.** `$G panel` prints a one-time local link. Providers (add your own gateway), chains with a reasoning-effort
   dropdown per slot, features and safety toggles, Chat, Files, Status. Chat with a computer task is a dry run until you approve.

## Not proven live yet, so say so or skip

Hands-free voice (`glide voice`; needs a headset and `uv sync --extra speech`), point-and-ask (`python -m glide.assistant.point_cli`,
needs Screen Recording permission), the menu-bar apps, webhooks, and anything on Windows.

## If a take goes wrong

`ls ~/glide-demo/runs/<take>`: `run.json` has the outcome and failure code, `failure.md` the observed facts. A run that
ends "uncertain" leaves a marker in the runs folder; `glide doctor` names it and how to clear it.
