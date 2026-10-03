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
the worker flag, then `Approver.approve(call)` for this run, then a lease check, then the executor. A missing
approver, a refusal, an approver that raises, or a lease lost while waiting means nothing touches the machine.
Approval is never cached. `glide webhooks work --allow-desktop` uses `TerminalApprover` (an explicit `y` on the
terminal that started the worker; no terminal means no). Tests use a fake approver and a fake executor.

An interrupted run that had been dispatched is `uncertain`: it blocks further claims for that agent and is never
replayed. An operator reconciles it through `POST /v1/agents/{agent}/messages/{id}/resolve` (scope
`agent:resolve`).

## What the queue stores (D3)

Default: for GitHub, repository, number, action, commit ids and comment id. For mail, history/message ids and
times. No titles, bodies, comments, senders, mailbox addresses, sender source URIs or URLs of any kind.
A finished task has its goal and context replaced by a placeholder; worker summaries are replaced by the worker's
content-free `note`; event text is dropped; `secure_delete` plus a WAL truncate remove the bytes from the files.
An `uncertain` task keeps its call until an operator resolves it. With `record_content` on, excerpts, summaries
and event text are kept (still never a URL). The consequence: GitHub reports run on metadata only unless you opt
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
`worker.py` the consumer, the approval gate and the executor seam.

Not done: mailbox retrieval and OAuth onboarding (mail stays `blocked` until a trusted handler is supplied),
MCP event subscription, subscription renewal.
