# 3-minute demo script (measured latencies, side-by-side with a general assistant)

Numbers below are what ran on this machine on 2026-10-04: `ask` 1.8 s total (first word 1.5 s), a compiled browser task
1.5 to 2.0 s (6 of 6 runs), a read-and-quote task 17 s, one jev action choice 0.3 to 0.5 s, speech synthesis probe 0.3 to 0.4 s.
Nothing about the other assistant is measured here: record its timer on screen and say only what the timer shows.

## Fair comparison rules (say them out loud, once)
- Same page (local, loopback), same goal text, same machine, both start from a blank state.
- Show a visible clock (screen recorder timestamp or a stopwatch window) for both. Do not speed up either side; if you
  cut for length, keep a clock in frame.
- Glide is told the success condition ("success means the address is ..."); give the other assistant the same sentence.

## Before you press record
Setup per docs/DEMO_LIVE.md. Local page up, loopback Chrome up. Chrome window and the other assistant's window side by side,
Glide's terminal below. Run each Glide command once off camera so the connections are warm; delete the old tabs.

## Timeline

| Time | Scene | You say | On screen | Measured / expected |
|---|---|---|---|---|
| 0:00 to 0:15 | Hook | "Same request, two assistants. One clock." | Both windows, clock visible | |
| 0:15 to 0:40 | Fast answer | "First, a question. No tools." | `glide ask --timings "In two sentences, why is the sky blue?"` | answer in about 1.8 s, `timings:` line shows route, first word, total |
| 0:40 to 0:55 | Visible fallback | "If a model fails, Glide says so." | same command with one wrong key | a `fallback:` line, the answer still arrives |
| 0:55 to 1:35 | Race 1: act in the browser | "Open the page, click the link, success means this address." | Left: Glide `computer ... --engine structured --act`. Right: the other assistant given the same sentence | Glide about 2 s; read the other clock aloud, whatever it is |
| 1:35 to 2:05 | Race 2: read and quote | "Now read a page and quote the evidence." | Glide `computer "Read http://127.0.0.1:8765/report.html and tell me how many widgets were sold, quoting the page"` | about 17 s, answer 42 with the quote and a source line |
| 2:05 to 2:30 | Safe by default | "Anything that changes things asks first." | `glide files plan/preview` (nothing moves), then apply with the hash, then undo | preview instant, 7 files into 5 folders, undo restores them |
| 2:30 to 2:45 | Control panel | "Everything above is configurable here." | `glide panel`: Providers (gateway), Chains with reasoning-effort dropdown, Features toggles | one click through, no typing of keys |
| 2:45 to 3:00 | Close | "Fast where it can be, honest where it can't." | Show a run folder: run.json outcome, failure.md | say the limits: voice and menu-bar apps are not in this cut |

## What to skip or say plainly
- Hands-free voice and the menu-bar apps: not proven live; leave out.
- Do not claim a speed multiple over the other assistant unless the on-screen clocks show it; read the numbers as they are.
- If a take fails, cut to the next take; the outcome is in `run.json`.

## Rehearse
Run Race 1 six times in a row; it finished 6 of 6 here. Keep the compiled wording exactly:
`open http://127.0.0.1:8765/ and click the link Open the report; success means the address is http://127.0.0.1:8765/report.html`
