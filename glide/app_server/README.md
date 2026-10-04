# glide.app_server

The core's side of the SwiftUI app's protocol (`app/PROTOCOL.md`, version 1): `glide app-server` serves it on a local Unix socket.
The Swift side lives in `app/` on the `consolidation/app-swiftui` branch; the wire format here was checked against its codec with
the golden lines in `tests/fixtures/app_protocol/` (see the README there).

```
glide app-server [--socket PATH] [--record-content]
```

## Modules

| Module | What it does |
|---|---|
| `wire.py` | The envelope, every message the core sends, the parser for what the app sends, and `clean_text` (the one door untrusted text goes through). No state. |
| `transport.py` | Socket path rules, the listener, the peer-user check. |
| `server.py` | Connections, framing, handshake, bounded queues, idle handling, approvals. Knows nothing about the assistant. |
| `bridge.py` | Commands to `Assistant` calls; assistant, provider and task activity to messages. |
| `settings.py` | The settings payload (from the loaded `glide.toml`) and `settings_set`. |
| `runtime.py` | The text assistant, and hands-free voice (`build_voice`) when the app turns it on. |
| `cli.py` | `glide app-server`. |

## Transport

- Unix domain socket only. No TCP, no port. Path: `--socket`, else `$GLIDE_SOCKET`, else `~/Library/Application Support/Glide/glide.sock`.
- The directory is made `0700` if missing; an existing one that is open to group or others, owned by someone else, or a symlink is
  refused (not repaired). The socket is `chmod 0600` straight after `bind`. A path over 103 bytes, a symlink at the socket path,
  a file that is not a socket, a socket owned by another user, or a socket a live core listens on are all refused. A leftover
  socket of ours that nothing answers on is removed. Closing unlinks only the node this process made.
- Each connection's peer uid is read from the kernel (`LOCAL_PEERCRED` on macOS, `SO_PEERCRED` on Linux) and must be ours; a peer that
  cannot be identified is refused.
- One client at a time. A second connection is closed without a message (the app retries with backoff).
- Bounds (`server.Limits`): lines 1 MiB; 256 queued messages per client; 64 queued before messages that may be lost are skipped; 5 s to
  hand one message to the socket; 5 s to say hello; ping after 15 s quiet; dropped after 45 s without a line from the app; 20 unusable
  lines. A typed request is at most 4,000 characters and is refused, never cut. At most 4 typed requests are answered at once.
- A client that cannot keep up is disconnected; the assistant never waits for it.

## What is sent, and what is not

Without `--record-content` (and until the app switches `privacy.record_content` on), the stream carries state and content-free events:

| Message | Carries | Never carries |
|---|---|---|
| `state`, `speech` | assistant state, hands-free, muted, a fixed `detail` code; speech phases | |
| `transcript` | `utterance_id`, `role`, `partial`, `redacted: true`, `chars` (a count) | any word, prefix or hash of what was said |
| `task` | phase, step count, `verified` as the run reported it, `summary` = the runner's own one-word outcome | the writer's answer, what the screen showed, `would_do` |
| `switch` | role, slots, kind, reason (scrubbed of keys, cleaned, at most 200 characters) | |
| `error` | a fixed code and a short scrubbed message; for a failed request, the exception type only | the request |
| `settings` | names from `glide.toml`, the NAME of each key variable and whether it is set | any key value |

With recording on: `transcript.text` (cleaned, at most 4,000 characters, `redacted: false`) and `task.summary` (the run's text, at most
200). Recording on is also set on the configuration (`config.record_content`), which is what the run folders already obey.
It lives for this process only and always starts off unless `--record-content` is given. Turning it on is logged (no content).

**The one exception, on purpose:** `approval_request.command` always contains the goal, because a request that does not say what it
asks you to allow is not an approval. It is cleaned and bounded (400 characters), shown, and never logged or stored by the core.

Every string that did not come from the core's own code goes through `wire.clean_text`: control characters, invisible formatting
characters (bidi overrides, zero-width), surrogates and private-use characters are removed, whitespace is collapsed, length is bounded.
A parse error never contains the line, and the log never contains content: an unknown message type is logged by its (cleaned, 40
character) type name only.

## Approvals

The assistant asks (`IO.approve`, the one hook added to `glide/assistant/core.py`) before ANY computer task starts, a dry run too,
because a dry run captures the screen. The question goes to the connected app only: kind `input` when the task will click and type
(`computer.act_enabled` is on), `screen` otherwise, `expires_in_s` 30, and `command` = a fixed sentence plus the goal.

It is granted only by `approval_response` with `decision: "approve"` from the same connection, for that id, before the deadline, while
the request that asked is still wanted. No answer, `deny`, a disconnect, a `stop`, a newer request that cancels it, no connected app,
a late or repeated or unknown-id answer, a broken approver: all are no, and no task is started (not even a dry run). An unknown
`decision` value is a malformed line, not an approve. There is no default yes, no "always" and no scope.

What an approval covers: **starting the task**. Once approved, the task's own clicks and typing are not asked about one by one.

## Task phases

Mapped from the events a run already reports (`ComputerTask.events`), read about ten times a second:

| Run event | Protocol phase |
|---|---|
| `accepted` | `started` |
| `progress` (an action was sent) | `attempted`, `step` = attempts so far |
| `action_checked` (a fresh observation followed) | `step` |
| `completed` (the final reading found the goal achieved) | `verified` (`verified: true`), then `completed` |
| `dry_run` | `completed`, summary "dry run: nothing was done" |
| `blocked` | `failed`; `reconcile_required` first if a write was still unconfirmed |
| `cancelled` | `stopped`; `reconcile_required` first if a write was still unconfirmed |

`unverified` is never sent: the run reports no negative verdict short of a failure. Nothing here decides that an action had its effect.

## Additions inside version 1

An older app ignores all of these (unknown message types and unknown fields are ignored by contract). They should be added to
`app/PROTOCOL.md` on the app branch:

- `approval_closed` `{approval_id, outcome}`, `outcome` one of `approved denied expired cancelled disconnected`: the core is no longer
  waiting, so the app can take the card down when a stop, rather than a click, ended the question.
- `transcript.chars`: the length of the withheld text (present only when `redacted` is true).
- `error.code` values the core uses: `malformed`, `line_too_long`, `text_too_long`, `too_many_errors`, `unsupported_version`,
  `unsupported_protocol`, `expected_hello`, `busy`, `unknown_task`, `request_failed`, `internal`, `warning`.
- `warning` (an `error` with `fatal: false` and no `reply_to`) carries what the assistant would print as a notice (a provider failed,
  a task was not approved).
- `hello.capabilities`: `text_input approvals settings switch_events task_events`, and `voice` when hands-free is available.
- `voice.language` is `""` when automatic.

## Not done, or not exact

- `level` (microphone and speaker meters) is not sent: the audio paths expose no level.
- `speech: interrupted` is sent when the app's `interrupt` or `stop` cut the speech. A barge-in by voice shows as `finished`. Speech
  shorter than the 0.1 s poll is not reported at all.
- Hands-free uses the assistant that `build_voice` makes, so it has its own history (and tasks cannot ask questions while it is on);
  typed requests go to it while it runs.
- Settings last for the process. Nothing writes `glide.toml`.
- Switches that happened while no app was connected are not replayed (the command line still prints them to stderr).
- None of the live behaviour has been run: see `docs/live-checks/app.md`.
