## What and why

<!-- One or two sentences. -->

## Checklist

- [ ] Tests are offline and pass: `uv run --python 3.13 pytest -q`; `ruff check .` and `ruff format --check .` are clean.
- [ ] No guard weakened: `tests/conftest.py` still refuses anything that reaches the machine; any new call that does is added to it in this PR.
- [ ] No model ids, endpoints, prices or keys in code or workflows: they are configuration, and keys come from env vars named in `glide.toml`.
- [ ] Every effect is verified from a fresh observation; an unknown-outcome write is never replayed.
- [ ] Stored run data omits utterances, typed text, captured content and raw URLs unless detailed recording is opted into (D3).
- [ ] Fallbacks are visible (`SwitchEvent`) and adapters raise only `ProviderError`.

## Not verified live

<!-- What you did NOT run on a real machine, screen, microphone or provider key. Write "nothing" only if true. -->
