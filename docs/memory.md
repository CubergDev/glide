# Memory (glide.memory)

Memory is **off by default**. Turning it on is the user's decision, because it keeps text they typed or said.

```toml
[memory]
enabled = true        # or GLIDE_MEMORY=1. Default false.
auto_capture = false  # or GLIDE_MEMORY_AUTO=1. Default false: also save sentences like "I prefer concise answers".
data_dir = "..."      # optional. Default: GLIDE_DATA_DIR, else the per-user data directory (macOS: ~/Library/Application Support/glide).
```

With `enabled` false, nothing is created, read or written. Environment values override the file. A mistyped key or
value is an error, not a guess.

## What is stored (one SQLite file, `<data dir>/memory/memory.sqlite`)

| Table | Holds | Source of the text |
|---|---|---|
| `memories` | key, text, kind, source, confidence, scope ids, timestamps, optional expiry | what the user saved (`glide memory remember`, `/remember`, or auto capture if enabled) |
| `outcomes` | run id (`run:` + sha256 of run.json, never a path or URL), success flag, short tactic summary | the host, after it verified the effect; summaries containing URLs or credentials are refused |
| `proposals`, `overlays`, `harness` | refinement text derived from three or more verified outcomes, evidence ids, revision numbers | derived from `outcomes` |
| `events` | allowlisted metadata only: ids, tool names, counts, durations, status. 30 days, 1,000 rows per user/project | never arguments, results, notification text, resource URIs or the user's utterance |

Not stored: utterances (unless saved as a memory), screenshots, page or tool content, typed text, raw URLs, keys.
`ingest_run` reads only the `outcome` label of a finished `run.json`. A writer's `achieved` flag is never success
evidence; the caller passes `verified_success` after a fresh observation.

Protection: the data directory and `memory/` are 0700; the database and its `-wal`/`-shm` files are 0600; a symlink
or foreign-owned directory is refused; `secure_delete` is on, so `forget` really deletes; a credential heuristic
rejects recognizable keys (it is not a complete scanner).

`glide memory status` shows whether memory is on and where it lives without creating anything.
`glide memory --help` lists remember, recall, forget, events, propose, apply, rollback and plan.

## Dropped in the port

`ContextualWriter` and `wrap_context_factory` decorated the Anthropic-style `messages.create` shim that D1 deletes.
Context for a model turn is `Harness.prepare(...).context`, appended by whoever builds the request.
`planning.plan` (the extension's `routing.py`) selects context, skills and tools that fit a model window; it is not
the intent router.
