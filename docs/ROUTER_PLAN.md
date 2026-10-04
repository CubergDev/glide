# Router plan (D9 prep)

Status: a plan only. Nothing here is built, and the two routers below are left as they are.

## Today

- `glide.assistant.router.Route`: what to do with a request from the user (`answer`, `computer`, `stop`), from a
  deterministic stop fast path and then one fast-LLM JSON call. A reply that is not exactly one JSON object is an
  answer (finding 7).
- `glide.computer.execution.routing.Scope`: who owns a task once it is a computer task (`execute`, `research`,
  `reason`, `clarify`), from one classifier question and, only when that is uncertain, the writer.

Two small classes with overlapping meaning (`reason` and `answer`, `clarify`), decided in two places by two prompts.

## One entry point

One function takes the user's words and the data-wrapped history and returns one decision record. Everything the
assistant or the engine needs to know about the request comes out of it; nothing downstream re-decides.

Tiers, cheapest first, each one able to end the decision:

1. **Stop fast path.** Whole-utterance match, no model, no network, runs before anything is built. Stays exactly as
   it is, with the configured extra phrases.
2. **Classifier.** The cheap structured question (`Scope`'s Jev question): `execute`, `research`, `reason`, `clarify`
   plus the not-a-task case. A classifier chain that is down raises its own `ProviderError` and is not a reason to guess.
3. **Fast-LLM JSON tier.** Only for what tiers 1 and 2 leave: speech-sized answers and the acknowledgement of a task,
   in the same call. Strict JSON, one object.

**Failure means answer, never act.** An unreadable reply, an unknown route, a provider that is down, a cancelled call:
the request is answered (or, on a cancel, dropped silently), and a fallback is a visible `SwitchEvent` or `io.warn`. A
route to the computer is only ever produced by a decision that parsed cleanly.

The decision record carries the route, the acknowledgement, the goal, the language, who decided (tier and provider),
the latency, and `reasons` (why, in fixed words) so the choice can be recorded in run.json and replayed offline.

## How `clarify` gets a channel

`Assistant(..., clarify=True)` is the opt-in: a task may put one question to the user, which is shown and spoken, and
the task waits for `Assistant.answer_pending(text)` or for a stop. No front end sets it today, so a `clarify` route can
only end the task. A front end that sets it must also route its next input: when `pending_question` is set, a typed
line or an utterance goes to `answer_pending`, anything else is a new request and drops the question. That is a
decision for the front end, not the assistant. Front ends:

- `glide chat` (`glide/cli.py`, `cmd_chat`): must set it and send the next line to `answer_pending` first.
- `glide listen` and `glide voice` (`cmd_listen`; `glide/speech/session.py`): must set it, and the voice loop must
  decide whether the next utterance answers the question (a listening turn right after the spoken question is the
  natural choice).
- `glide ask` (one shot): should not. It has no next input, so a question can only end the task; it keeps `clarify=False`.

Those files belong to the front-end owners; the assistant side needs no change.

## Offline evaluation set

Routing quality has to be measured before the boundary is moved. The idea: a labelled set of requests, each with the
expected route and workflow, run against the router with fake or recorded model replies, scored on three numbers:
wrong-act rate (must be zero: a request that should not touch the machine routed to it), missed-act rate, and
answer-instead-of-clarify rate. It runs offline in pytest (fakes only), and against real keys only by hand.

Seed, unverified: the prototype's `evals/corpus/` in `consolidation-inputs/extension-local-prototype`
(`decide/questions.jsonl` 120 lines, `llm/tasks.jsonl` 296, `workflows/scenarios.jsonl` 36, by line count; their
labels were not read or checked, and `decide` looks like element choice rather than task scope). Add: multilingual
(English, Cantonese, Mandarin) stop and not-stop phrases, requests that quote screen text with embedded instructions,
and near-miss phrasings of stop ("stop the music").

## Decisions that need the user

1. Where a quick spoken answer ends and the frontier `reason` route begins (never defined; the evaluation set is the
   way to tune it).
2. Whether `reason` and `answer` stay two routes or become one with a length limit.
3. Which front ends enable `clarify`, and whether a voice user may answer a question by voice.
4. The acceptable wrong-act and missed-act rates for the evaluation set to gate on.
5. What a down classifier does for a request that is plainly a task: answer ("I can't reach it") or refuse.
6. Whether the router may use memory (D8, off by default) for context.
