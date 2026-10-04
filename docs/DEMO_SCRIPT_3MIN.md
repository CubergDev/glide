# 3-minute demo script (every number measured on this machine, 2026-10-04)

Measured: `ask` 1.8 s total (first word 1.5 s); jev picks a screen action in 0.3 to 0.5 s; a stated-address browser task
1.5 to 2.0 s (6 of 6); a chain of two link clicks across three pages 2.2 s (5 of 5); read-a-page-and-quote 16 to 18 s;
a loosely worded browser request 16 to 25 s (about seven model calls of 1.4 to 3 s each, in sequence); file plan, preview,
apply and undo instant. Nothing about any other assistant is measured here: show its own clock and read what it says.

## Say once, early
"Glide's fast path needs the addresses stated; a looser request still works, through a planner, and I will show both."
Give the other assistant the same sentence as Glide in every race. Do not speed up either side. Keep a clock in frame.

## Before you record
`docs/DEMO_LIVE.md` setup; local page up (pages: index, report, archive); loopback Chrome up; `glide doctor --live` all ok;
demo folder `~/glide-demo/messy` filled; old tabs closed. Run every command once off camera to warm connections.

## Timeline

| Time | Scene | Do and say | Expected |
|---|---|---|---|
| 0:00 to 0:30 | **Panel and onboarding** | `glide panel`. "No config files to hand-edit." Providers: show the gateway, key status (set/missing, never the value). Chains: one slot, the reasoning-effort dropdown. Features: computer control OFF by default, the engine toggle with its note. | one click per point, nothing typed |
| 0:30 to 0:45 | **Doctor** | `glide doctor --live`. "Every slot, measured, in one table." | rows ok: fast, smart, speech-to-text, text-to-speech, jev classifier; a bad row names its reason |
| 0:45 to 1:00 | **Fast answer and visible fallback** | `glide ask --timings "In two sentences, why is the sky blue?"`; again with one wrong key | about 1.8 s; a `fallback:` line, answer still arrives |
| 1:00 to 1:40 | **Race: act across pages** | Same sentence on both: `open http://127.0.0.1:8765/ and click the link Open the report; success means the address is http://127.0.0.1:8765/report.html; then click the link Open the archive; success means the address is http://127.0.0.1:8765/archive.html` | Glide about 2.2 s for three pages, three verified steps; read the other clock |
| 1:40 to 2:05 | **Evidence, not vibes** | `glide computer "Read http://127.0.0.1:8765/report.html and tell me how many widgets were sold, quoting the page" --engine structured --act` | 16 to 18 s; "42 widgets", the quote, a source line |
| 2:05 to 2:25 | **Honest about the slow path** | the loose request `open http://127.0.0.1:8765/, click the report link and tell me what it says` | 16 to 25 s; say why: a planner plus a reader plus a checker, several model calls in a row |
| 2:25 to 2:50 | **Safe by default, with undo** | `python -m glide.files plan ~/glide-demo/messy --intent type --out ~/glide-demo/plan.json`, `preview` (nothing moves), `apply --approve <hash>`, then `undo <manifest>` | 7 files into 5 folders; undo puts them back exactly |
| 2:50 to 3:00 | **Close** | show `run.json` and `failure.md` of a run | "Every run leaves its outcome; unknown outcomes are never retried." |

## Leave out
Hands-free voice and the menu-bar apps (not proven live); any claim of a multiple over another assistant that its own
on-screen clock does not show; Windows.

## Rehearse
Run the race command five times in a row; it finished 5 of 5. Use the exact wording, including the semicolons.
If a take fails, `~/glide-demo/runs/<take>/run.json` says why; cut to the next take.
