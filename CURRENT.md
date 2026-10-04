# Glide: current status

Updated 4 October 2026. Glide is a Python voice and text assistant with a computer-control loop. Its language models,
speech services and classifier are configurable providers; nothing about a vendor is in the code.

## Where to read

- [README.md](README.md): what Glide is, install and extras, setup, the command tree and what each command takes control of,
  interruption and its limits, the safety model.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): package map, request flow, cancellation, provider chains, recording.
- [docs/DECISIONS.md](docs/DECISIONS.md): decisions D0-D18 and where each stands.
- [docs/LIVE_CHECKS.md](docs/LIVE_CHECKS.md): everything that cannot be proven offline, as checklists, grouped by risk.
- [AGENTS.md](AGENTS.md) and [CLAUDE.md](CLAUDE.md): rules for agents. [docs/PROPOSED_AGENTS_CHANGES.md](docs/PROPOSED_AGENTS_CHANGES.md) holds the edits
  `AGENTS.md` needs, for its owner to make.
- [docs/history/](docs/history/): the consolidation notes (historical) and an index of archive-only reference material.

## State

- **Offline suite:** passes apart from the work in progress below. Last measured on this branch before the docs change: 3428
  passed, 10 skipped, 1 xfailed, 20 failed. The 20 are known and owned by the engine-glue work: 17 in
  `tests/test_execution_cli.py` (the command tests still patch an old `cli.load_config`), two in `tests/test_cli_tree.py`
  (the old-name guard and the example-file URL guard) and one in `tests/test_diagnostics_files.py`. Run
  `uv run --python 3.13 pytest -q` for the current number.
- **Proven live:** nothing. Voice, browsers (CDP, native, Obscura, Playwright CLI), the research pipeline, provider
  cancellation over TLS, and the webhook and MCP servers are tested only against fakes.
- **Two open design items:** D9 (two routers still exist; the structured engine is reachable only through
  `glide computer --engine structured`) and D11 (the SwiftUI app and its socket server).
- **On branches, not merged:** the pet and point-to-ask, echo cancellation for speakers, the SwiftUI app, the rewritten CI
  setup. The table is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Run

Python 3.12 or later, managed with `uv`. Copy `glide.toml.example` to `glide.toml`, export the variables it names, run
`glide doctor`. Entry points: `glide` (the one command tree), and the older names `glide-computer`, `glide-inspect`,
`glide-webhooks`, `glide-webhook-worker`.

```sh
uv run --python 3.13 pytest -q
uv run --python 3.13 ruff check . && uv run --python 3.13 ruff format .
```

Detailed content recording is opt-in. Follow `AGENTS.md` for machine approval, keys and repository changes.
