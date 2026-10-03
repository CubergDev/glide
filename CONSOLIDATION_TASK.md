# Consolidation task

Merge the current workspace and every supplied local variant into one coherent, working Glide implementation. Preserve all useful capabilities, configurations, tests and documentation. Compare every variant, resolve overlapping implementations, and remove stale duplication only after retaining useful information.

Use dynamic workflows for each substantive phase: inventory, integration, verification and documentation. Consult the configured advisor for difficult design conflicts. Assign disjoint file ownership to parallel workers and reconcile their changes before verification.

## Inputs

- Current `glide/`, `tests/`, `pyproject.toml`, `uv.lock`, `glide.toml.example`, `AGENTS.md`, `CLAUDE.md` and `CURRENT.md`.
- Every file explicitly listed in `consolidation-inputs/INPUTS.json`.

The five input directories are independent snapshots. The manifest lists identical files so you can avoid repeated reads while still checking each variant's unique changes. The input directory is read-only: implement changes in the current workspace.

Use only these supplied local inputs. Do not follow external links, access remote repositories, inspect git history, read saved chats or memory, or search outside the workspace. Do not inspect environment values, credentials or personal settings. Additional material will be supplied explicitly if needed.

## Requirements

1. Produce a concise capability inventory covering every variant and document. Separate implemented behavior from earlier proposals.
2. Integrate compatible unique capabilities into the current architecture. Retain the provider abstractions, visible fallback events, configuration-driven model choices and verification of effects.
3. Resolve duplicate modules, commands and configuration shapes. Keep one clear path for each capability rather than parallel competing implementations.
4. Keep tests offline. Never invoke real screen capture, input, applications, local model servers or external APIs. Do not weaken `tests/conftest.py` guards.
5. Run appropriate offline tests and lint checks. Use cached dependencies and offline package operations. Diagnose failures instead of hiding or skipping them.
6. Consolidate documentation into a short current setup and architecture guide. Preserve useful detailed reference material locally and make stale alternatives clearly historical.
7. Summarize changes, capabilities retained, verification results and unresolved decisions in `CONSOLIDATION_RESULT.md`.

Do not push, publish, deploy, merge a remote change or modify account settings. Other work is active outside this workspace; do not interrupt it or overwrite its changes.
