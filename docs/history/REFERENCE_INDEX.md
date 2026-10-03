> **Historical document (4 October 2026).** An index of reference material that is **not part of Glide**. It exists so
> nobody mistakes archive material for product code or runs it. Current documents: [README](../../README.md),
> [ARCHITECTURE](../ARCHITECTURE.md), [DECISIONS](../DECISIONS.md), [LIVE_CHECKS](../LIVE_CHECKS.md).

# Reference material kept in the input archive only

Decision D10 (and D16, D17): the OSWorld package, the benchmarks, the GCP infrastructure and the sandbox are not part of the
package and are **never run**. This repository holds none of them, and none is to be copied in. They stay where the
consolidation received them: the input archive, a read-only folder named `consolidation-inputs/` in the original
workspace, which is git-ignored (as are `review-evidence/` and `REVIEW_COVERAGE.json`) and is never committed, pushed or
published. The folder names below are inside that archive. This page does not copy a path, a remote or a handle from it.

Nothing in the archive is authoritative about the current product. Where an older copy differs from the code here, the code
here wins; the audits said older checkouts must not overwrite later fixes.

## D10: never run, not packaged

| Material | What it is | Why it stays out |
|---|---|---|
| OSWorld package and harness (in the Glide-lineage variants and one Windows checkout) | An evaluation harness for a desktop-agent benchmark | Not a product feature. Its GCP variant copies a `.env` to a cloud VM and creates billable resources: never run it |
| Benchmarks and evaluation scripts | Speech, routing and agent benchmarks; speech benchmark prototypes with synthetic audio | Figures exclude real microphone behavior and are not validation of this tree |
| GCP infrastructure (`infra`) | Cloud provisioning for the benchmark | Creates and deletes cloud resources and spends money: needs the owner's explicit yes, and nothing here needs it |
| Sandbox | A containerised run environment from the OSWorld lineage | Not part of the package |

`tests/fixtures/osworld/` in this repository is **not** that harness. It holds two small test fixtures (a screenshot and a
recorded step) used by offline tests.

## D16, D17: retained in place, not integrated

| Folder in the archive | What it is |
|---|---|
| `variant-1` to `variant-6` | Five older snapshots and the latest complete working tree before consolidation (variant-6) |
| `extension-memory`, `extension-routing` | The memory module (ported, opt-in: `glide/memory`) and a task scheduler with a model selector (not an intent router and not ported; what the router could reuse is in [../notes/routing-extension-notes.md](../notes/routing-extension-notes.md)) |
| `extension-local-prototype` | An earlier unfinished assistant prototype and its evaluation framework, with 114 synthetic WAV files. Its routing cases could seed an offline routing evaluation set (unverified) |
| `extension-local-review`, `extension-local-archive-1` to `-5` | An alternate branch snapshot and five branch and pull-request backups (CI and AI-review workflows, a code-owners file, a bootstrap script). Never run the bootstrap script; the live repository settings are not to be changed (D18) |
| `extension-raccoon-local`, `extension-project-site` | Raccoon workflow materials, and site and deck source: administrative, not product |
| `extension-mac-extra-*`, `extension-windows-*` | The final Mac and Windows source batches (944 files). Supplied after the review; **not reviewed**. The audits' own description is in [HANDOFF.md](HANDOFF.md) section 12: a separate assistant/platform/routes family and several point-to-ask lineages that were candidates to diff, not code that was merged |
| `documents/`, `context/` | Research, reviews, an earlier `DECISIONS.md` and nine workstream records: historical evidence |
| `REVIEW_COVERAGE.json` | The per-content review ledger (2.7 MB). If it is ever wanted in the repository, the owner says so and a trimmed copy goes under `docs/history/` |
| `review-evidence/` | A copy of the review session's temporary files (assembler, raw reviewer outputs). Private |

## What was not reviewed

Coverage was never complete: the ledger's manifest predates the final Mac and Windows batches. See
[CONSOLIDATION_RESULT.md](CONSOLIDATION_RESULT.md) section 4 and [HANDOFF.md](HANDOFF.md) section 9.
