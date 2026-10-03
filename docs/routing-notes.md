# Notes from extension-routing for the router (D9)

`extension-routing` (`permit_smart_switch`) is a task scheduler. Its scheduler and `SmartSwitch` are not ported: its
escalation rules (low confidence goes to the frontier, two failures escalate) contradict the recorded routing
choices. What the single router may reuse as ideas, none of it as code:

- **Capability filter before ranking**: a model is eligible only if it supports the stage, fits input plus reserved
  output in its window, and the current grants allow it; then rank by a configured preference (cost, latency,
  quality) with a deterministic tie-break. Costs and tiers come from configuration, never from code.
- **Stale-plan check**: a prepared plan carries the revision of memory, catalog and tool inventory; any change
  expires it (`glide.memory.Harness._check_plan` does this).
- **Escalation as a visible handoff**: when a fast tier gives up, pass the previous attempt's story to the next
  tier, and record the switch as a `SwitchEvent`.
- **Opt-in memory as a router feature**: `Harness.prepare` returns memory ids and a context string without a
  model call, so the router can read them as features when memory is enabled.
