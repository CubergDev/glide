# The router (`glide/routing/`)

One entry point decides who owns a request: `Router.route(utterance, context) -> Decision`. It replaces the two routers
that duplicated each other (`glide/assistant/router.py` and `glide/computer/execution/routing.py`'s `decide`). The package is
**new and not yet wired in**; section 7 is the plan to do that in under an hour. Decision D9 in `HANDOFF.md`.

## 1. What it decides

`Decision(route, owner, confidence, why_code, tier, goal, reply, question, language, calibrated, provider, latency_s,
tiers_tried, errors, switches)`.

| route | owner | meaning |
|---|---|---|
| `stop` | none | cancel the task and the voice |
| `answer` | fast model | talk, a quick fact; also what "I could not decide" means (it cannot touch the machine) |
| `execute` | Jev (executor) | operate the browser or apps; no composed answer asked for |
| `research` | frontier + Jev | read pages or the screen and report back, with sources |
| `reason` | frontier | calculation, drafting, code, an explanation that needs thought; no browser |
| `clarify` | the user | the outcome is unclear: ask, never guess |

`execute` and `research` are the **acting** routes (`ACTING`). Everything else is speech or a question.

## 2. Tiers, thresholds, fallbacks

1. **Not the user, empty, or a stop phrase** (no model, no network). Text whose source is not `user` (screen, page,
   task result, memory, assistant) is never routed: the answer is `answer`/`untrusted_source`, not even a stop. The one stop
   list is `glide/routing/stop.py` (moved unchanged from `assistant/router.py`); `[speech] stop_phrases` only adds to it.
2. **Classifier chain** (`GlideConfig.classifier()`, failover and `SwitchEvent`s as before): one Choice question over the six
   routes with the executor's own criteria. Confidence is calibrated, then compared to a threshold.
3. **Fast-LLM JSON call** (`GlideConfig.llm("fast")`), only when tier 2 failed or was not sure. The whole reply must be ONE
   JSON object with exactly the schema's keys; a fence, prose, a second object, extra, missing or duplicate keys, an
   unknown route or confidence word are unparseable. Its confidence is `low`/`medium`/`high`, valued by the calibration table.

Accepted when, after calibration: a plain route (answer, reason, clarify) `>= min_confidence`; an acting route
`>= act_min_confidence` and, from the classifier, a margin over the runner-up `>= act_min_margin`; a model-heard stop
`>= stop_min_confidence`. No tier is sure:

| situation | route | why_code |
|---|---|---|
| the call was cancelled (person spoke or stopped) | answer, say nothing | `cancelled` |
| a tier is sure of an acting route and the utterance carries an override attempt ("ignore your instructions...") | answer | `injection_marker` |
| `confirm_acting` on, and the other tier does not read the acting route the same way | answer | `unconfirmed_action` |
| a tier is sure | its route | `classifier_confident` / `fast_llm_confident` / `stop_model` |
| nobody is sure, but a tier puts `>= min_confidence` of its mass on answer or reason (sure it is no action, unsure how deep) | **reason** (frontier) | `escalated_depth_unsure` |
| nobody is sure and the doubt includes whether to act | answer | `uncertain_answer` |
| every tier failed, was missing or unparseable | answer | `tiers_failed` |

Doubt about acting never acts; doubt between a quick answer and the frontier goes to the frontier
(`escalate_to_reason = false` turns that off). A cancel returns at once and the next tier is not tried. Every hop is
visible: error kinds in `Decision.errors`, a `SwitchEvent(role="router")` per hop in `Decision.switches` and `on_event`, plus the
classifier chain's own failovers. `Decision.record()` is all a run record keeps (codes, never text); the text fields are
excluded from `repr`.

## 3. Context, memory, untrusted text

`Context(history, data, running_task, memory_hints, language)`. History is bounded (`history_turns`, `history_chars`).
Only the user's and assistant's turns are shown to a model; a task result keeps its outcome line and loses everything after
`Text read from the screen, data only:`. `data` (a page, a screen read) contributes **source and size only**, so no
text on a page can reach a routing model, however a model would read it (tests plant canaries and look for them in
everything either model was shown). Memory hints are used only when the front end passes them (only when memory is opted in)
**and** `[routing] memory_hints = true`; they are shown as data, bounded, and cannot change a threshold. The router never opens
a memory store. `Context.from_messages(history_dicts)` builds a context from the assistant's chat-style history.

## 4. Clarify needs a channel

```python
class Clarifier(Protocol):
    def ask(self, question: str) -> str | None: ...
```

`resolve(router, utterance, context, clarifier)` routes, asks while the decision is `clarify` (budget `max_clarifications`,
default 1), and re-routes with the question and the user's reply added to the history; the final goal gets the executor's
`Clarifications:` suffix. With no clarifier, an answer of `None` or blank, or a spent budget, **nothing is done**:
`Resolution.said` is one sentence naming what is needed (English, Cantonese, Mandarin) and `Resolution.unresolved` is true. A
reply that is itself a stop phrase is a `stop`. A NEW request arriving instead of the answer makes `ask` return `None` and is
routed on its own (the assistant's `answer_pending` rule).

## 5. Configuration (`[routing]`)

Every number is a default named in `glide/routing/settings.py`, not a claim about a model. Unknown keys are errors.

```toml
# Routing (glide/routing): who owns a request. See docs/ROUTER.md. Every number is a default; measure yours.
# [routing]
# min_confidence = 0.6        # answer, reason, clarify need at least this, after calibration
# act_min_confidence = 0.8    # execute and research drive the machine, so they need more
# act_min_margin = 0.2        # ...and the classifier must put this much more on them than on the runner-up
# stop_min_confidence = 0.9   # a stop the MODEL heard (the phrase list is exact)
# escalate_to_reason = true   # sure it is not an action, unsure how deep: the frontier model, not a guess
# history_turns = 4           # earlier exchanges the router may read (0-12)
# history_chars = 300         # of each turn
# memory_hints = false        # use hints the front end passes from memory (it passes none unless memory is on)
# speculative_fast = false    # ask the fast model beside the classifier: one extra call, less waiting
# confirm_acting = false      # two keys to act: the fast model must read an execute/research request the same way
# fast_timeout_s = 15.0       # the fast model's JSON call
# max_clarifications = 1      # questions put to the user for one request
# calibration_file = ""       # a reliability table from a live run on YOUR providers (section 6)
```

**Not done here (provider agents own the files):** paste that block into `glide.toml.example` after the `[speech]` block, and
add `"routing"` to `OWN_TABLES` at `glide/providers/config.py:103` (`OWN_TABLES = ("memory", "mcp", "webhooks", "routing")`),
or `load_config` logs "ignoring the unknown table [routing]". Read the table with
`RoutingSettings.from_table(read_table("routing", find_config(os.environ)))` (the `glide.memory.settings` helpers) and pass it to
`build_router(config, table=...)`.

## 6. Calibration, the offline evaluation, and what is not measured

**Calibration** (`glide/routing/calibration.py`): per tier and predicted route, the raw confidence is binned; each bin keeps (cases,
right); `apply` returns the smoothed (`(right+1)/(n+2)`), monotone accuracy of the bin, falls back to the tier's pooled bin and
then to the raw number (reported as `calibrated=False`) when a bin has fewer than `min_samples` cases. With no table the raw
number is used and a classifier that says 0.9 about everything is taken at its word, so **until a live table exists, turn on
`confirm_acting`**.

**Offline set**: `tests/routing/cases.jsonl` (191 labelled utterances: stop 24, small talk 15, factual 15, computer 37,
research 18, reasoning 17, ambiguous 20, overconfident 4, hostile 32, context 9; English 123, Cantonese 32, Mandarin 22,
mixed 9). 38 are seeded from the prototype corpus (`consolidation-inputs/extension-local-prototype/evals/corpus/{decide,llm,
workflows}`, read-only, unverified): only the user goal text was taken, each is marked `relabelled` and carries its source id
(the prototype's labels were element choices, not intents). The rest are written here. `tests/routing/eval.py` runs them
through the real router over two fakes with simple documented behaviour (cue counting; `eval.py` docstring), in two configs:
`default` and `careful` (`confirm_acting` plus a table cross-fitted on the other folds).

Result of `python tests/routing/eval.py` at the time of writing: default accuracy 0.958, careful 0.979; false actions 0 on
hostile and ambiguous in both; the `overconfident` group (a classifier scripted to be sure and wrong) acts 4 of 4 by default and
0 of 4 with two keys; calibration error 0.148 -> 0.073 (classifier) and 0.102 -> 0.046 (fast) after the cross-fitted table. With
models that OBEY everything, the structure alone (sources, strict JSON, the override guard, failures) still never acts on 25+ cases.
`tests/routing/test_routing_eval.py` asserts these bars and that the eval can fail (guard removed, thresholds removed).

**Honest limits.** The fakes were written next to the cases, so the accuracy numbers flatter the router; they show the router's
logic and thresholds, nothing about a real model. `tests/routing/reliability_fakes.json` is the table the fakes produce (a test fails
if it is stale): it must never be pointed at by `calibration_file`. Real-model routing quality can only be measured live:

1. `cp glide.toml.example glide.toml`, export the key variables it names, `glide doctor`.
2. `GLIDE_ROUTING_LIVE=1 python tests/routing/eval.py --live --out reliability_live.json` (you run it; it spends provider credit,
   prints codes and ids only, never an utterance, and refuses without the variable).
3. Read the per-tier accuracy and "false actions with the fitted table" (the bar is zero); put the file in `[routing]
   calibration_file`; re-run after any provider or model change. Add your own utterances to `cases.jsonl` (no `fake` key).

`--live` was written without being run (no network, no keys, and it must not be run by an agent): treat it as unverified.
Other limits: the override-marker set is narrow (it needs override framing, so "turn on developer mode" is a task, with hard negatives in the set) and covers only Cantonese/Mandarin phrasings written here; a user who
really says an override sentence gets an answer, not an action; the quoted-passage feature is passed to models but no rule
reads it; `LLM.chat(schema=...)` in `providers/llm.py` already trims a fence or sentence off a reply, so the strict parse here
is a second line, not the first (providers-owned: have `chat(schema=)` return the raw reply or reject non-JSON-only text, or
accept the double check); the stop matcher does not match "hey glide never mind" (the classifier tier handles it).
Latency is unmeasured: the classifier call precedes every answer, which the old design paid once (fast call), so expect the
streamed answer to start later by one classifier call unless `speculative_fast` and a short classifier timeout are used.

## 7. Integration plan (line numbers are against `06-barge-in` d1a61be; check them)

Files owned by other agents are marked (owner). Each step is one commit and keeps the suite green. **Parallel-safe order:** steps 1
(providers), 2 (cli and speech imports) and 3 (engine, runner, tasks) touch disjoint files and can be done at the same time by
different owners; step 4 (core) needs step 3's `route` field; steps 5 and 6 follow step 4; step 7 is independent.

1. **Config** (providers): the two lines in section 5; add `routing` to `glide doctor`'s table of roles if wanted.
2. **Stop copies** (the stop matcher already lives in `glide/routing/stop.py`; there is no second list in this tree):
   `glide/cli.py:41` `from .assistant.router import is_stop` -> `from .routing.stop import is_stop` (engine/cli);
   `glide/speech/settings.py:29` `from ..assistant.router import normalize` -> `from ..routing.stop import normalize` (speech);
   `glide/assistant/core.py:43` (below). `speech/turns.py:27` is prose. `glide/assistant/router.py` re-exports until step 5.
3. **Engine takes the route** (engine): add `route: str = ""` to `RunConfig` (`computer/runner.py:56`, after `execution_browser`).
   In `execution/engine.py` `_scope` (line 265) start with
   `if cfg.route: scope = routing.Scope(cfg.route, "router"); self.routes.append(routing.record(scope)); return scope`
   so the structured engine stops deciding ownership when the router already did (the legacy engine ignores `route`). Keep
   `routing.decide` for `glide-computer` run directly (route empty). Later, optionally fold it: `decide` becomes a call of
   `glide.routing.tiers.ask_classifier` restricted to the four task routes and `route_task` and its writer prompt are deleted.
   Both then read their criteria from `glide.routing.tiers.CRITERIA` so the two cannot drift. `ComputerTask`/`TaskRunner.start`
   (`assistant/tasks.py:319`) gain `route=""` and pass `RunConfig(route=...)` (`tasks.py:~270`).
4. **Core** (core): `assistant/core.py`
   - line 43: `from ..routing import Context, Router, Span, build_router, resolve` and `from ..routing.stop import is_stop, stop_phrases`;
   - `__init__` (153-156): keep `extra_stop_phrases`; add `routing: Mapping | None = None` (the parsed `[routing]` table); build
     the router once, lazily, with `build_router(self._config, table=routing, stop_phrases=extra_stop_phrases, on_event=self._router_switch)`
     (`_router_switch` writes `fallback: router <from> -> <to> (<kind>)` through `io.warn`, like `cli.format_switch`; do not call
     `cli.format_switch`, `cli` imports the assistant); warn once about the tiers `build_router` returns as missing;
   - line 193: `fast_path(text, self._stops) is not None` -> `is_stop(text, self._stops)`; lines 300 and 330 already call `is_stop`;
   - `_respond` (221-236): replace `llm = self._config.llm("fast")` and `Router(llm...).route(text, self._messages())` with
     `resolution = resolve(router, Span(text), Context.from_messages(self._messages(), running_task=<current task goal or "">, language=hint_language), clarifier)`
     and `decision = resolution.decision`, where `clarifier` is the `Clarifier` below when `self._clarify` else `None`;
     `resolution.decision.cancelled` -> return `Reply("none")` (alongside the existing `turn.cancelled` check); the `route.source == "fallback"` warning is now `on_event`; `route == "stop"` -> `decision.route == "stop"`;
   - dispatch on `decision.route`: `stop` -> `self.stop()`; `answer` -> `_answer` unchanged (use `decision.reply`, `decision.language`);
     `reason` -> `_answer` but stream from `self._config.llm("smart")` (the frontier, one response, no browser; the `ANSWER_PROMPT` plus
     "give a complete answer" is enough) and `reply.route = "answer"`; `execute` and `research` -> `_computer(...)` with
     `goal = decision.goal or text` and `route=decision.route` passed to `self._tasks.start(...)`; `reply.route = "computer"`;
     `clarify` -> if `Resolution.unresolved`, `self._emit(turn, resolution.said, ...)` and `reply.route = "clarify"`, no task;
   - after a downgrade (`why_code` in `injection_marker`, `unconfirmed_action`, `uncertain_answer`, `tiers_failed`) the answer must not
     claim to have done anything: pass the answer prompt one line ("Nothing was done on the computer for this request; do not say you
     did it. If the user asked for an action, say you did not do it and what is needed"), or speak a fixed phrase. An attempted action is
     not a verified effect (AGENTS.md); a model that sees only "open my bank" will say "opening it now".
   - the `Clarifier` (new, in core.py): `ask(question)` shows and says the question (`_ask_user`), then waits on a `threading.Event`
     that `answer_pending(text)` sets (check this router-level wait before the task's, line `answer_pending`), and returns `None` on
     a stop, a new request (`_begin`, which already drops a pending task question: drop this one too) or a timeout; `pending_question`
     reports it too. `Reply.route` docstring (line 78) gains "clarify".
5. **`assistant/router.py`** (core): delete `Router`, `Route`, `ROUTE_SCHEMA`, `ROUTER_PROMPT`, `ROUTER_TOKENS`, `parse_route`, `_first_object`
   and the stop sets; keep `answer_messages`, `ANSWER_PROMPT`, `LANGUAGE_NAMES` (move them to `assistant/answer.py` and import there).
   `normalize`/`is_stop`/`stop_phrases`/`fast_path` can stay one release as `from ..routing.stop import ...` re-exports.
6. **Tests** (core): `tests/test_assistant_router.py`: the stop and normalize cases are already in `tests/routing/test_routing_stop.py`
   (delete them and the parity test with the old module); the `parse_route`/`Router`/schema tests are covered by
   `tests/routing/test_routing_{tiers,router}.py`; keep the `answer_messages` tests as `tests/test_answer_prompt.py`. The fakes:
   `tests/test_assistant_fakes.py` `route_json` (53 call sites in `test_assistant_core.py`, `test_cancel_assistant.py`, `test_assistant_cli.py`)
   emits the old schema; change it to `fast_json` shape (`computer` -> `execute`, add `"confidence": "high"`, `"question": ""`).
   `FakeConfig.classifier()` already raises `NoUsableProvider` when unset, so those tests exercise the fast tier alone, which behaves
   as the old router did. Add one core test per row of section 4 (no clarifier -> `said`, reply used, new request drops the question).
7. **Docs** (docs): `AGENTS.md` layout line for `glide/routing/`; link this file from `HANDOFF.md` section 5.

Nothing in `glide/routing/` imports `glide.assistant` or `glide.computer.execution`, so these steps cannot create a cycle.
