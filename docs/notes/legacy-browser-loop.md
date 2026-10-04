# The legacy browser DOM loop: proof it is not production, and what goes with it

Nothing is deleted by this note. It gives the evidence that `glide/computer/browser/{runner,decide,perceive,report,act}.py`
(1,397 lines) is not reachable from any entry point, and the exact set to remove together. The proof is a test that keeps
being true until the removal: `tests/test_process_seams.py` (`test_the_legacy_browser_loop_is_imported_by_nothing_but_itself`,
`test_nothing_imports_by_a_computed_name_so_the_scan_above_sees_every_import`, `test_no_entry_point_group_loads_a_module_by_name`).

## What the loop is

A self-contained loop over Chrome DevTools: `runner.run_goal` perceives the DOM (`perceive`), asks the classifier
(`decide`), acts with raw CDP input (`act`) and writes a run folder (`report`). It predates `glide.computer.execution`
(the structured engine, `execution/dom.py` and `execution/playwright_cli.py` for browsers). It is the only code that
sends input with no `control.dispatch` (`act.py` calls `Input.dispatchMouseEvent` and `Input.dispatchKeyEvent` itself),
which is one more reason to remove it rather than polish it.

## Proof

1. **Entry points.** `pyproject.toml` has five `[project.scripts]` and no `gui-scripts` or `[project.entry-points]` group:
   `glide.cli:main`, `glide.cli:computer_main`, `glide.cli:inspect_main`, `glide.webhooks.cli:main`,
   `glide.webhooks.worker:main`. None of those modules, and nothing they import, names a legacy module (below).
2. **Static imports over `glide/`** (every `import` and `from ... import`, at module level and inside functions, relative
   imports resolved, each package above an imported module counted). The importers of each module:

   | module | imported by |
   | --- | --- |
   | `browser/runner.py` | nobody |
   | `browser/decide.py` | `runner.py` |
   | `browser/report.py` | `runner.py` |
   | `browser/act.py` | `runner.py` |
   | `browser/perceive.py` | `runner.py`, `decide.py`, `report.py`, `act.py` |

   Before this branch's change to `browser/__init__.py` the package `__init__` imported `runner`, `decide` and `perceive`,
   so `from ..browser.cdp import CDPError` in `execution/dom.py` and `execution/playwright_cli.py` loaded the whole loop
   at run time (it was loaded, never called). `__init__` now imports only `cdp`; the test fails on the old file.
3. **Dynamic imports.** The only computed-name imports in `glide/` are the two `importlib.import_module` calls in
   `platform_adapter.py` (`f"{__package__}.windows"` and `.macos`); `features.py` only calls `find_spec`. No `__import__`,
   `runpy`, `exec` or `eval` of a module name, and no `python -m glide.computer.browser...` (the package has no `__main__`).
   Grep over `docs/`, `README`-style files and `glide.toml.example` finds no mention of `run_goal` or these modules.
4. **Callers by name.** `run_goal`, `typing_target`, `resolve_text`, `perceive`, `act.*` of the browser package are called
   only inside the package and by the five test files below. (`perceive` in `glide/computer/runner.py` and `cli.py` is
   `glide.computer.perception.perceive`, a different function; likewise `report.render_payload` in `glide/computer/`.)

Reproduce: `uv run pytest -q tests/test_process_seams.py -k "legacy or computed or entry_point"`.

## Remove together

Source (1,397 lines):

- `glide/computer/browser/runner.py`, `decide.py`, `perceive.py`, `report.py`, `act.py`.

Keep: `glide/computer/browser/__init__.py` (now only `CDPError` and `Session`), and `cdp.py`, which
`execution/dom.py` and `execution/playwright_cli.py` import (`CDPError`, `Session`, `_get_json`, `local_debugger_url`).

Source left dead by the removal, to remove in the same change:

- `glide/computer/writer.py`: `compose_browser_text` (used only by `browser/runner.py`; its tests are in
  `tests/test_browser_report.py`, below). `looks_credential`, `compose_url` and `valid_url` stay: `execution/query.py`
  and `actions.py` use them.
- Two comments that name `browser/decide.py`: `glide/providers/classifier.py` lines 44 and 109.
- In this branch's tests: the `glide/computer/browser/act.py` entry in `CDP_WRITERS` in `tests/test_process_seams.py`.

Tests to delete whole (824 lines):

- `tests/test_browser_loop.py` (16 tests; also the fakes `FakeBrowser`, `FakeTypeSafe`, `login_page`, `listing_page`, `run`
  that `test_browser_control.py` and `test_classifier_scenarios.py` import).
- `tests/test_browser_report.py` lines 14-17 and 51-197 (`serialize_answers`, `render_answers`, `RunFolder`, `load_step`,
  `render_payload`, `resolve_text`). **Keep** the credential tests in it (about lines 198-262: `looks_credential`,
  `CREDENTIAL_HINTS`) by moving them to a writer test file, and delete the `compose_browser_text` tests with that function.
  So this file is split, not deleted.

Tests to split (keep the non-legacy part):

- `tests/test_browser_control.py`: delete the first part (`test_browser_releases_key_on_stop...` through
  `test_navigation_to_http_and_https_goes_through`, lines 18-79, which drive `run_goal` and `act`). **Keep** lines 82-142
  (`_Socket`, `_Clock`, `_session` and the three `cdp.Session` deadline tests) and move them to a cdp test file.
- `tests/test_browser_offline.py`: delete the tests of `perceive`, `act.fingerprint`, `available_actions` (all but one).
  **Keep** `test_the_session_only_connects_to_this_chromes_loopback_port` (`local_debugger_url`, `CDPError`) and move it to
  the cdp test file.
- `tests/test_classifier_scenarios.py`: delete `import test_browser_loop as browser_loop` (line 22), the imports at lines
  30-32, and the five `test_the_browser_...` tests (lines 578-656), and the `browser_server` helper (line 658, used only by the last of them); the rest of the file is the native classifier.

Not affected: `tests/test_browser_providers.py`, `tests/test_snapshot_retry.py`, `tests/test_cli_diagnostics.py`,
`tests/conftest.py` and `tests/test_no_real_machine.py` import only `cdp` (and the last two are changed by
`cdp-launcher-removal.patch`, a separate change).

## Not proven

- A person running `python -c "from glide.computer.browser.runner import run_goal"` from an uncommitted script or a
  benchmark harness outside this repository. Nothing in the repository does; ask whoever ran OSWorld or the CDP benchmarks
  (`cdp.py` mentions "a benchmark run") before deleting.
- The static scan sees imports spelled in source. It cannot see a module name built from data; the second test pins that
  there is no such import in `glide/`.
