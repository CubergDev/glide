# Glide: current project notes

Updated 4 October 2026.

Glide is a Python voice and text assistant with a computer-control loop. Its language models, speech services and classifier are configurable providers. Use this folder for the current implementation; older research describes alternatives and earlier plans.

## Configure and run

- Python 3.12 or later; dependencies are managed with `uv`.
- Copy `glide.toml.example` to `glide.toml` and set the environment variables it names.
- Configure the fast and smart language-model chains, speech recognition, speech output and classifier in `glide.toml`.
- Example model identifiers require live validation. `glide doctor` reports provider reachability.
- Entry points: `glide`, `glide-computer`, `glide-inspect`.

## Implementation

- `glide/assistant/`: handles text or speech, returns an answer or computer task, and speaks the result.
- `glide/computer/`: observes, classifies, acts and verifies the resulting state.
- `glide/providers/`: provider adapters, shared errors and fallback chains. Provider failures use `ProviderError`; switches are visible as `SwitchEvent`.
- `tests/`: offline verification with guards against real machine interaction.

## Working rules

Follow `AGENTS.md` for machine approval, configuration, handling keys, recording and repository changes. Verify effects from fresh observations and reconcile uncertain writes before retrying. Detailed content recording is opt-in.

Offline checks:

```sh
uv run --python 3.13 pytest -q
uv run --python 3.13 ruff check .
```

## Retained material

Detailed older notes, research and conversation material are in [Retained Material](/Users/cuberg/Documents/Retained-Material-2026-10-04). They provide historical context; the implementation and configuration here determine current behavior.
