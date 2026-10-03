# Consolidation result

Date: 4 Oct 2026. Status: **review and decisions complete; integration not started.** The final Mac and Windows source batches (944 files) were supplied after the review and are **not reviewed**. The plan for the next agent is in `HANDOFF.md`.

## 1. Changes made to the workspace

Only two edits. Everything else in the workspace is untouched.

- `tests/conftest.py`: the loopback-only network guard now also drops proxy settings (`*_PROXY` removed, `NO_PROXY=*`, `urllib.request.getproxies` returns nothing). A loopback proxy could otherwise carry a request out past the guard; this showed up as a `CONNECT` attempt to `example.com` during `test_no_real_machine`. This strengthens the guard.
- `pyproject.toml`: `[tool.ruff] extend-exclude = ["consolidation-inputs"]`, because ruff tried to write caches into the read-only inputs.

New documents: `HANDOFF.md`, `CONSOLIDATION_DECISIONS.md`, `REVIEW_COVERAGE.json`, this file.

## 2. Capabilities found (implemented versus proposed)

Implemented with tests in the supplied code, and mapped to a disposition in `REVIEW_COVERAGE.json`:

- **Current workspace (kept as the base architecture):** provider chains with failover, pinning, hedging and visible `SwitchEvent`s; LLM, speech-to-text, text-to-speech and classifier adapters; `glide doctor`; the assistant (router, epoch-based interruption, spoken output); the screen-driving loop.
- **Variant-6 (latest complete tree):** structured execution engine (contracts, effect journal, DOM and native backends, planning, policy, progress), task routing, research supervisor and page reading, browser providers (CDP, Obscura, Playwright CLI), query-form execution, diagnostics, run control and cancellation, content-gated recording, neutral writer boundary, desktop access preflight, ElevenLabs speech with VAD and full-duplex audio, raccoon pet UI, ChatGPT sign-in, speech qualification scripts.
- **Variant-4 only:** point-to-ask, webhook ingress and queue (GitHub, Gmail, Outlook), headless worker, OS-keyring writer key. Absent from variant-6.
- **Variants 1, 2, 3, 5:** older snapshots. Variants 3 and 5 are a reduced sibling of variant-2 and lose nothing relative to variant-6 (74 hunk groups compared: 55 superseded, 9 trivial, 9 alternative designs not worth porting, 1 unclear; no lost work).
- **Extensions:** `extension-memory` (SQLite memory, context planner, harness, MCP bridge, CLI) and `extension-routing` (task selection and scheduler). Source and tests of the `*-current` copies are byte-identical to the earlier ones.
- **Historical or admin material (retain only):** the local prototype and its evaluation framework with 114 synthetic WAV fixtures, five branch/PR backups (CI and AI-review workflows, CODEOWNERS, bootstrap script), Raccoon workflow materials, site and deck, OSWorld harness, GCP infra, research documents.

Proposals only, not implemented: MCP-events receiver, recipe cache and replay, local trilingual model stack, bounded chooser, the HacKU shopping-agent product, durable mandate stores, offline self-improvement loop.

## 3. What the review verified

`REVIEW_COVERAGE.json`: 2,694 manifest paths, 927 distinct contents (hash-identical copies share one review).

- 585 contents have a fresh review whose quoted evidence was found in the exact file bytes.
- 114 WAVs are verified at header and metadata level only (hash equals both manifests, RIFF header valid, duration and rate match); the audio was not decoded or played.
- 14 slide previews were viewed; 11 cross-check against `DECK_CONTENT.json`, 3 do not.
- 214 contents rest on the first-pass cluster readers' own reports; 50 were independently sampled and 44 agreed. The 6 disagreements were over-broad capability links, not wrong dispositions.
- 29 contents whose bytes changed in a later header cleanup were re-reviewed from the current bytes; their old reviews were voided.
- Mechanical checks: variant-3/5 and variant-4 hunks against variant-6; 396 workstream patch entries against the supplied files (124 of workstream-3's 197 file edits fully reflected, 4 not reflected, many partly); disk-hash drift check (none).

## 4. What was not verified

- No live provider, desktop, microphone, browser or network use. Every reported live result and test count (v6 note: 1,490 passed; memory 117; routing 124) is a report, not validation.
- Variant-6's own suite and the extensions' suites were not run (missing dependencies; extension tests were deliberately not run).
- The agent transcripts could not be inspected, so "no inventory agent ran anything risky" is unconfirmed. No run folders or browser artifacts were found.
- Two reviewers listed `~/.cache/uv` (outside the workspace) with `ls`; no contents were read.
- The final Mac (308 files) and Windows (636 files) source batches were supplied after the ledger was built and are not in `REVIEW_COVERAGE.json`. By hash lookup 803 of those files (713 distinct contents) have bytes the ledger has not seen; that is an upper bound on new work, not a review. Their scope and candidates are in `HANDOFF.md` section 12, which is based on the audit files and directory listings only.

## 5. Offline verification of the current workspace

Command (the sandbox could not use `uv`): `.venv/bin/python -m pytest -q -p no:cacheprovider` and `.venv/bin/ruff check --no-cache .`, `ruff format --check`.

- Baseline before edits: 1510 passed, 1 xfailed, 15 failed, 19 errors.
- After the guard fix: **1511 passed, 1 xfailed, 14 failed, 19 errors.** Ruff check and format are clean (90 files).
- All 33 failures and errors come from the sandbox denying loopback port binding (real-HTTP, real-websocket STT and writer-endpoint fixture tests, one loopback-reachability test). They are environmental. They should pass where binding is allowed; this was not confirmed.

## 6. Limits of the review environment

`uv` could not write its cache and there was no network, so `uv.lock` was not regenerated and nothing was installed. The review session's temporary files (assembler script, raw reviewer outputs, merge simulations, workflow outputs) were copied to `review-evidence/` in the workspace at the user's request (20 MB, 569 files; the bulky source copies `v2run`, `v3run`, `sim` and `stubs` were left out). The folder is gitignored and private.

## 7. Decisions and what remains open

Resolved by the user: see `HANDOFF.md` section 2. Open: D2 (user unsure; recommendation stands), the PySide6 pet fallback, the first branch name for pushing to `cubergdev/glide` (the remote the user named), the router's quick-answer versus frontier boundary (left to a smart classifier with an offline evaluation set), repo remote and branch names, what the unreviewed Mac and Windows batches add (notably the `assistant/{controller,platform,routes,audio}` family and the point-to-ask lineage to choose). Open technical failures: the ~60 s provider disconnect, the Brave/CDP process exit with unestablished cause, real microphone-to-speaker behavior, live status of Obscura, Playwright CLI, desktop/OCR and the restaurant-review workflow.
