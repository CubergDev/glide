> **Historical document (4 October 2026).** This is the pre-decision sheet, kept for the options and tradeoffs behind each decision. It describes the workspace as it was then, not as it is now: statements such as "not started", "not yet" or "no integration has been done" are out of date, and the paths it names for the old layout may not exist. Current documents: [README](../../README.md) (what Glide is and how to run it), [ARCHITECTURE](../ARCHITECTURE.md) (how it is built), [DECISIONS](../DECISIONS.md) (decisions D0-D18 and their status), [LIVE_CHECKS](../LIVE_CHECKS.md) (what only a person can verify) and [CURRENT](../../CURRENT.md) (one-screen status). The rules in [AGENTS.md](../../AGENTS.md) always apply.

# Consolidation decisions: essential sheet

> **Superseded for decisions by `HANDOFF.md` section 2** (the user has answered). This sheet is kept for the options and tradeoffs behind each decision. Where the two differ, `HANDOFF.md` wins.

Status: **review complete for inputs supplied up to the local-archive batch; the final Mac and Windows batches are supplied but unreviewed; implementation not started; final coverage is PENDING.**
The final Mac and Windows source batches arrived after this sheet; they are not covered here (see `HANDOFF.md` section 12).
Everything below is evidence from the supplied files. Reported results (live runs, test counts) are **not** current verification.

## 1. What the evidence says

- **Variant-6 is the base candidate.** It is the latest complete tree and extends the variant-2 lineage. It keeps every "v2-only" module (browser providers, Obscura, Playwright CLI, query forms, diagnostics, click-controlled dictation) as live, modified code, and adds the routing, research and page-reading engine. My earlier "stale" label for those modules was wrong.
- **Variants 3 and 5 lose nothing against v6.** They are a reduced sibling. I compared every differing hunk in the execution engine, computer core, speech and pet code (74 hunk groups): 55 superseded by v6, 9 trivial, 9 alternative designs (none worth porting) and 1 unclear (a literal model id that sits in v6 itself, see D6). No lost work. Details are in `REVIEW_COVERAGE.json` and the scratchpad lineage files.
- **Variant-4 holds the only unique work that v6 lacks:** point-to-ask, webhooks, and an OS-keyring writer key. These are a side branch, not part of v6. `extension-local-archive-3` holds a different copy of point-ask and credentials, and I did not judge which is later.
- **The current workspace holds the architecture you want to keep:** the provider chains with visible `SwitchEvent`s, and the assistant router. Its `glide/computer` is the variant-1 loop plus about 50 changed lines in `runner.py` and `writer.py`, with `providers/` and `assistant/` added. Of the shared files, only `runner.py` and `writer.py` are expected to conflict textually (3 conflicts each); the rest merge cleanly as text but depend on new modules.
- **Two routers already exist.** v6 `execution/routing.py` is the code behind your recorded routing choices (browser = JEV, pure reasoning = one frontier call, research = frontier plans plus JEV browsing with cited evidence). It duplicates the current `assistant/router.py`. `extension-routing` is a task scheduler, not an intent router.

## 2. Decisions that block the merge

Each has a recommendation. Tell me where you disagree.

| # | Decision | Recommendation | Main tradeoff |
|---|---|---|---|
| D0 | **Merge frame** | Three-way merge: base v1, ours = current, theirs = v6, keeping `glide/computer`, `glide/providers`, `glide/assistant`. | Alternative: keep current as base and port v6 pieces by hand. That is safer per piece but loses v6 wiring and takes far longer. Wholesale v6 replacement would drop the provider chains and `SwitchEvent`s. |
| D1 | **Writer boundary** | One neutral `generate(GenerationRequest)` implemented over the provider chains. Delete the `messages.create` shim. | Touches `providers/writer_client.py`, `calls.MeteredWriter`, `world.FakeWriter` and several tests. Keeps failover visible. v6's writer has 812 lines against current's 434. |
| D2 | **Cancellation** | Adopt v6 `RunControl`; `ComputerTask.stop()` calls it. The variants' `check_abort` calls the checkpoint first, so it should compose with the current `abort_on`; this needs a test. | Small API change in `assistant/tasks.py`. The alternative keeps two cancel paths. |
| D3 | **Recording default** | Adopt v6's `record_content=False` default. It is the only form that matches AGENTS.md. Keep `failure` in `run.json` but scrub it. | Breaks `ComputerTask.would_do()` and tests that read `run.log` or answer files unless they opt in. Needs a test update. |
| D4 | **Key storage (ChatGPT sign-in, keyring)** | Core stays env-var names only, as AGENTS.md says. Offer ChatGPT sign-in only if you want it, as an explicit opt-in with a written exception. | v6 stores OAuth tokens and the ElevenLabs key in the OS keyring, plus an email-bearing settings file. A subscription silently falling back to a billed key is a spend decision, so no automatic fallback across that line. |
| D6 | **Hard-coded ids** | Remove every literal model id, endpoint and voice id from code (e.g. `gpt-6-astra` keyed on the OpenAI host, `chatgpt-writer`, OAuth URLs). Resolve them through `glide.toml`. | Required by AGENTS.md. Needs the config schema extended for planner and research roles. |
| D7 | **Execution engine and browser providers** | Port v6 `execution/` as `glide/computer/execution/`. Route its classifier through the chain, because it builds `TypeSafeClient()` directly today. Ship CDP and native; keep Obscura and Playwright CLI optional. | Without the chain change, Jev calls bypass failover and `SwitchEvent`. Only CDP was live-qualified (reported). |
| D9 | **One router** | **Option B:** hoist v6 `routing.decide` into `assistant/router.py`: stop fast-path, then the classifier, then the existing fast-LLM JSON call as fallback. The engine receives the route. `permit_smart_switch` stays out of the request path. | Alternative A is one LLM call but drops Jev-first routing. Option C (smart-switch as router) conflicts with v6's low-confidence policy. **Open question for you:** where is the line between a quick spoken answer and the frontier "reason" route? The sources do not say. |

## 3. Scope decisions with a default (retain-only unless you say otherwise)

| # | Item | Default | Why |
|---|---|---|---|
| D5 | v4 webhooks, point-to-ask, writer-key credentials | Leave out of the first consolidation; keep the files as reference. | The webhook queue stores excerpts and URLs and has a desktop-driving path. Point-ask needs v6's redesigned pet UI. The archive-3 copy must be reconciled first. |
| D8 | `extension-memory` and `extension-routing` | Retain as reference; port later as opt-in, default off. | The memory store persists user text. Both are async or MCP-oriented, untested in a live host, and each recheck found new defects. Their source and tests are byte-identical in the "current" folders. |
| D10 | OSWorld package, benchmarks, infra/GCP, sandbox | Retain only. | `osworld-gcp` copies `.env` to a cloud VM and creates billable resources. It is never run here. |
| D11 | Pet UI (PySide6), hands-free voice | Optional extras later; voice only through the provider adapters. | v6 changed these most; neither is core. One open item: v6 removed the pet UI's stdout buffer cap (v3 capped it at 4,096 characters). Intended, or a regression to restore? |
| D12 | Naming | Rename `clicker-*`, `permit`, `CLICKER_*` to `glide`/`GLIDE_*` at port time; one `glide` command tree. | Avoids parallel entry points. |
| D14 | Dependencies and `uv.lock` | Decide which of keyring, filelock, platformdirs, pyjwt, httpx≥0.28 become core versus extras. | None of these is installed here except httpx (version not checked), and `uv.lock` cannot be regenerated offline. **You would need to run `uv lock` online, or approve it.** |
| D16, D17 | Local prototype and eval corpus, site and deck, Raccoon materials | Retain in place, no integration. | The prototype is unfinished (its own notes say nothing is built or benchmarked). The deck hedges its own status. Raccoon stays a separate dev/eval tool. |
| D18 | Archive automation (CI, AI-review workflows, CODEOWNERS with a personal handle, `bootstrap-github.sh`) | Exclude; keep as reference. Never run the bootstrap script. | It changes remote repository settings and references secrets by name. The PR-template wording could be adopted later. |

## 4. Recorded failures and what can be reproduced offline

These are the failures the workstreams recorded. For each, the consolidation should add an offline regression test (no network, no desktop, no sockets, since loopback bind is denied in this sandbox and mock transports are enough):

| Failure | Offline repro |
|---|---|
| Readiness deadline too short (1.5 s, later raised) | Injected-clock fake page that becomes ready at 3 s: page read without reopening, block exactly at the configured deadline. |
| Connection dropped near 60 s, partial answers rejected | Fake stream that closes mid-way: `ProviderError` of the transport kind, no partial text used, no replay, visible switch. **The cause is still open**; a fake cannot prove a provider fix. |
| Credit exhausted (429) shown as a generic error | Mock transport returning 429: one request, safe message, no body text. |
| False "CDP disconnected" on a read error | A read raising an evaluation error must not map to connection loss. |
| Invalid first planner step, invalid control targets | Replay the recorded plan: one bounded correction, then stop; non-http(s) targets rejected. |
| YouTube playback reported failed although playing | Fake media state that turns "playing" after 6 s with a changed URL: one click only. |
| Dropped speech segments; internal counters spoken | Fake STT with several committed segments (none dropped); a TTS fake that never receives effect-count text. |
| Browser endpoint dead, tasks failing silently | Fake CDP endpoint refusing connections: actionable message, zero actions, goal kept. |

v6 already contains many of these tests. Mechanically, 124 of 197 workstream-3 file edits are fully reflected in v6. Four are not and need a check: `engine.py`, `speech/service.py`, `writer.py` and `tests/page_reader_world.cjs`. Workstreams 1, 4 and 9 mostly cover decks and docs with no supplied base file, so I cannot verify them.

## 5. Needs current verification; do not claim

- The ~60 s provider disconnect is fixed, or that the full research task passes. It is open.
- The 72.1 s museum result validates the current build. It predates the provider and readiness changes.
- Any reported live time or test count (1,490, 117, 124 and the rest) shows the merged tree passes. Nothing was rerun for the exports.
- Real microphone-to-speaker delivery, Obscura, Playwright CLI, desktop/OCR, or the restaurant-review workflow work live. None was live-qualified.
- Extension modules work in a live host or over MCP. Both reports say untested.
- Anything about publishing: a draft PR was created without permission in an earlier workstream, PR #2 is reported still open (closing it needs your explicit permission) and PR #3's status is unknown. Nothing here pushes, publishes or touches a PR.

## 6. Coverage and pending inputs

- `REVIEW_COVERAGE.json`: 2,694 manifest paths, 927 distinct contents, manifest hash `36755e11f723…`. Hash-identical copies share a review.
- 585 contents have a fresh review whose verbatim evidence was found in the exact bytes. 114 synthetic WAVs are verified at header and metadata level only (not listened to). 14 slide previews are image-viewed; 11 are cross-checked against `DECK_CONTENT.json` and 3 are not. 214 older cluster-reader reviews are reader-reported; 44 of 50 sampled agreed. The 6 disagreements were over-broad capability links, not wrong dispositions.
- The refresh changed 29 contents. Their previous reviews are void and absent from the ledger; the current bytes were reviewed fresh.
- **`coverage_complete` is false.** The final Mac and Windows batches are not in the ledger. New inputs inherit reviews by hash and appear as not-reviewed until reviewed.
- Limits: `uv` cannot write its cache here; the loopback port bind is denied (34 current-suite tests cannot run, one fewer after my guard fix); the v6 suite was not run because its core dependencies are missing; no live provider, desktop or microphone use.
- Changes already made in the workspace: `tests/conftest.py` now drops proxy settings so a loopback proxy cannot carry a request out (strengthens the guard), and `pyproject.toml` excludes `consolidation-inputs/` from ruff.

## 7. The short list I need from you

1. **D0:** confirm the merge frame (v6 into current, keeping the provider and assistant architecture).
2. **D4:** is ChatGPT sign-in wanted? If not, the keyring token store is dropped.
3. **D9:** confirm Option B, and tell me where quick answers end and the frontier "reason" route begins.
4. **D3:** accept `record_content=False` as the default, with the `would_do` and test updates.
5. **D7:** which browser providers ship in the core (my default: CDP and native).
6. **D5, D8, D10, D11, D18:** confirm these stay retained-only for now.
7. **D14:** may I list the new core dependencies for you to run `uv lock` online, or do you want a different route?
