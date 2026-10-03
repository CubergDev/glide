# What the router (D9) can take from extension-routing

Source: `consolidation-inputs/extension-routing-current` (`permit_smart_switch`). It is a task scheduler with a
model selector, not an intent router, and its escalation rules contradict the recorded choices, so **none of it is in
the request path**. This note lists only what the single router could reuse. Nothing here was ported or run.

## Do not adopt

- Escalation on low confidence: it sends low-confidence classifications, ambiguity or two failures to a frontier
  model (default confidence threshold 0.85). The recorded choice is the opposite split (browser actions: JEV with a
  small planner; pure reasoning: one frontier reply; research: frontier plans, JEV operates). Confidence thresholds
  also are not calibrated probabilities (the extension says so itself).
- The graph scheduler (dependencies, read/write resource conflicts, run-wide cost budgets, one active run per
  switch). It adds concurrency rules the engine does not need and a second place that owns ordering.
- Retry to the next model after `RetryableError`/`Escalate`: a failed action must never be replayed
  automatically (AGENTS.md); only read-only model calls could ever qualify.

## Worth reusing as ideas

- **Eligibility before preference.** A model is a candidate only if it supports the stage and tools, fits input plus
  reserved output, and satisfies current grants; then rank by a configured preference (cost, latency, quality or
  balanced) with a deterministic tie-break. All of it comes from host descriptors; no model ids or prices in code (D6).
  `glide.memory.planning.plan` already does the fit-and-grant part for context packing.
- **Goal text is opaque data.** Keywords in a user message never grant a capability or permission; permissions come
  from host grants rechecked at dispatch (the harness here rechecks after any approval wait).
- **Risk tiers as host data.** Simple writing needs a lower quality tier than planning or high-risk advice; the tier
  is a field on the model descriptor, not a rule baked into the router.
- **Memory retrieval as a background dependency of inference**, not a step the user waits for: fetch context while
  the classifier runs. Memory is opt-in and off by default (D8); the router should treat it as a feature that may be
  absent.
- **Visible routing decisions.** Every plan carries a `reasons` list (why each model, tool or memory was or was not
  chosen). The router's decision record should do the same, and any fallback must be a visible `SwitchEvent`.
- **Bounded, finite JSON for anything that crosses a boundary** (64 KiB values, 4,096 items, 32 levels) and
  detached copies so a callback cannot mutate router state.

## Open for the router work

Where the boundary between a quick spoken answer and the frontier "reason" route sits was never defined by the
user (HANDOFF section 5); this extension does not answer it either. An offline routing evaluation set remains the
way to tune it.
