# Webhooks (optional)

GitHub, Gmail and Outlook callbacks, a durable SQLite queue and a separately started worker. Ported from the
variant-4 branch to the `glide` namespace. Nothing here runs unless you start it.

## What is off by default (D5)

| Thing | Default | Switch |
| --- | --- | --- |
| The whole service | refuses to build | `"enabled": true` in the config file |
| Each source | answers 404 | `"enabled": true` on that source |
| Content in the queue | identifiers only | `"record_content": true` |
| Desktop execution by the worker | blocked | worker `--allow-desktop`, **and a yes for every run** |
| Clicking and typing in an approved run | looks only | worker `--act` **and** the source's `allow_actions`, both |

Provider sources (GitHub, Gmail, Outlook) can never allow actions. Only a signed generic `standard` task
request can, and only on a source you configured with `allow_actions`. Nothing in a webhook body (`act`, paths,
models, handler names) selects any of this.

### Per-run desktop approval

`worker.process_call` hands an `agent.task.requested` call to the gate (`_desktop`). The gate runs, in order:
the worker flag, then `Approver.approve(call, act=...)` for this run, then a lease check, then a check that the run
is still the one that was shown, then the executor. A missing approver, a refusal, an approver that raises, a lease
lost while waiting, or a run whose goal, context or source changed after it was shown means nothing touches the
machine. Approval is never cached. `glide webhooks work --allow-desktop` uses `TerminalApprover` (an explicit `y` on
the terminal that started the worker; no terminal means no). It shows the run id, the source, the whole goal,
whether input will be sent, the size of the sender's context with a short excerpt, and an approval digest; control,
escape and direction characters are removed from everything the sender wrote. An unanswered prompt waits for as long
as the terminal does (the lease keeps renewing); it is never a yes. Tests use a fake approver and a fake executor.

The desktop loop is given the authenticated goal and nothing else. The sender's request context is untrusted and the
loop has no separate channel for evidence, so the context is not passed to it (it is still stored, and shown to the
approver).

An interrupted run that had been dispatched is `uncertain`: it blocks further claims for that agent and is never
replayed. An operator reconciles it through `POST /v1/agents/{agent}/messages/{id}/resolve` (scope
`agent:resolve`). A run that only looked (`dry run`) is reported `blocked`, never `completed`. The worker retries
only the final completion request, with a short bounded backoff, on a transport failure or a busy server.

Each leased run has one `RunControl` (`glide/computer/control.py`), fenced by the lease deadline. It is the current
control while the run is processed, so a lost or expired lease cancels the model call made for it, and a
KeyboardInterrupt stops the desktop task and ends the lease before it propagates.

## What the queue stores (D3)

Default: for GitHub, repository, number, action, commit ids and comment id. For mail, history/message ids and
times. No titles, bodies, comments, senders, mailbox addresses, sender source URIs or URLs of any kind.
A finished task has its goal and context replaced by a placeholder; worker summaries are replaced by the worker's
content-free `note`; event text is dropped; `secure_delete` plus a WAL truncate remove the bytes from the files.
With `record_content` on, excerpts, summaries
and event text are kept (still never a URL).

**The one D3 exception: an `uncertain` run.** A run that ended `uncertain` (a write may or may not have happened)
keeps its goal and context so an operator can review what it was meant to do. This is bounded:

- **Retention limit.** `uncertain_retention_days` in the webhook JSON (default 14, range 1 to 90), or the environment
  variable `GLIDE_WEBHOOK_UNCERTAIN_RETENTION_DAYS` (same range, wins over the file). Once a row has been uncertain
  that long (the limit itself counts as expired), its goal and context are deleted. The row keeps its identifiers,
  stays `uncertain` (it still blocks its agent; nothing is replayed) and reads `Expired, not reconciled.` with an
  `expired_at` time. `secure_delete` plus a WAL truncate remove the bytes from the files.
- **When it runs.** At service start and then hourly while it runs. Only rows still `uncertain` are touched, never
  pending, leased or finished ones.
- **Reconcile.** `glide webhooks reconcile <message-id> done|not-done|unknown` records your verdict and deletes the
  content at once. `done` (the write happened) leaves `completed`, `not-done` leaves `cancelled`, `unknown` leaves
  `failed`; none of them requeues or reruns anything, and the agent is unblocked. It works on an expired row too.
  Check the real effect yourself first (an attempted action is not a verified effect). The HTTP resolve endpoint
  deletes the content in the same way.
- **Recording on.** With `record_content` on nothing is deleted at the limit; the content is kept, as that setting says. The consequence: GitHub reports run on metadata only unless you opt
in, and they say so.

Receipts keep the source, event id and a digest of the authenticated body, so duplicate deliveries are
recognised. Treat the database and its backups as private application data.

## Trust anchors (documented D6 exception)

`trust_anchors.py` is the only file allowed to contain a URL: the Google key-set location and the issuer names
for verifying Gmail push tokens. They are fixed public facts of that protocol and are never supplied by a
callback, a JWT header or the config. `tests/webhooks/test_webhooks_literals.py` fails if any other file in the
package contains a URL, a model id or an old product name.

## Install and check

```text
uv sync --extra webhooks            # fastapi, uvicorn, pyjwt[crypto]; pins are the lead's to set
glide webhooks serve --config webhooks.json --check-config
```

`--check-config` reads the file and the named environment variables, builds nothing and starts nothing.
Signing secrets come from the environment variables named in the config (at least 32 bytes). A keyring is an
optional adapter behind `SecretSource` (`secret_sources.KeyringSecrets`), never the default.

The launch commands below reach local services and, with `--allow-desktop`, this machine. They have not been run
as part of this port; AGENTS.md approval applies to each exact command.

```text
glide webhooks serve --config webhooks.json
glide webhooks work --server http://127.0.0.1:8000 --agent triage-agent --allow-model --once
```

`glide-webhooks` and `glide-webhook-worker` are the same two commands under their older names. Without `--config`,
`serve` reads `$GLIDE_WEBHOOK_CONFIG`, else `config = "..."` under `[webhooks]` in glide.toml, else `webhooks.json`;
`glide doctor` reports whether that file says `"enabled": true`. The worker's `--config` is glide.toml.

The server binds loopback by default. A public bind needs `--behind-proxy` and a TLS reverse proxy that keeps
raw bodies and authentication headers and does not log bodies, bearer tokens or query strings. One server
process and one worker per agent: a small-team queue, not a broker.

## Layout

`contracts.py` messages; `settings.py` configuration; `secret_sources.py` secrets; `trust_anchors.py` the
exception above; `auth.py` signatures and agent JWTs (PyJWT and `cryptography` imported lazily); `translation.py`
and `mail.py` event-to-call mapping; `store.py` the queue; `app.py` the ASGI service; `cli.py` the launcher;
`worker.py` the consumer, the lease, the approval gate and the executor seam.

Not done: mailbox retrieval and OAuth onboarding (mail stays `blocked` until a trusted handler is supplied),
MCP event subscription, subscription renewal.
