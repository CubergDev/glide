# Glide app protocol, version 1

How the SwiftUI app (`app/`) talks to the Python core. One local Unix domain socket, one JSON object per line. The Swift side is implemented in `Sources/GlideProtocol` and tested in `Tests/GlideProtocolTests`. **The core side does not exist yet**: whoever writes the Python socket server implements this file. The Swift models are the reference for field names.

## Transport

- A Unix domain stream socket. No TCP, no port number anywhere.
- Path: `--socket <path>` argument, else the `GLIDE_SOCKET` environment variable, else `~/Library/Application Support/Glide/glide.sock`. `~/` is expanded. The path must be at most 103 bytes (the `sockaddr_un` limit).
- The core creates the socket (mode 0600, in a directory that is 0700) and should check the peer uid on each accepted connection (`getpeereid`) and refuse any other user.
- The app refuses to connect unless the path is a socket owned by the current user (`SocketPath.validate`), so it never sends speech or typed text to a socket another account placed there.
- The core does not need to be started by the app. Starting the core from the app is not part of this version.

## Framing

- UTF-8 JSON, one object per line, terminated by `\n` (`\r\n` is tolerated, blank lines are ignored).
- A line is at most 1,048,576 bytes. An over-long line is dropped up to its newline and reported once; the connection stays up.
- A line that is not valid JSON or does not match its message's shape is dropped. Neither side echoes the line's content in an error or a log (it may be what the user said).

## Envelope

```json
{"v":1,"type":"state","id":"c12","reply_to":"a3","data":{...}}
```

| Field | Meaning |
|---|---|
| `v` | Major protocol version, an integer. A line with another value is refused. The core closes the connection; the app stops retrying and shows "unsupported protocol version". |
| `type` | The message type, below. |
| `id` | Optional. The app sets it on every command it sends (`a1`, `a2`, ...). The core may set it on events. |
| `reply_to` | Optional. On a core message that answers an app request, the request's `id`. |
| `data` | The payload object. Absent for `ping`, `pong`, `interrupt`, `settings_get`. |

Compatibility inside version 1: new optional fields and new message types may be added. Receivers ignore unknown fields and unknown message types. Enum values the receiver does not know decode to a neutral fallback (`unknown`, or `other` for approval kinds). Removing or retyping a field, or changing its meaning, needs `v: 2`.

## Handshake

1. The app connects and sends `hello` first.
2. The core answers `hello` within 5 seconds, with `protocol` set to the version it will speak. If the app's `protocol` is not one the core speaks, the core sends `error` with `fatal: true` and closes.
3. The app treats the connection as ready only after the core's `hello`. Nothing else is sent before that.
4. On a dropped connection the app reconnects with backoff (0.5, 1, 2, 5, then every 10 seconds) and repeats the handshake. After a reconnect, the app sends `settings_get`. It sends nothing that was pending before the drop (see "No replay").

## Core to app

| `type` | `data` | Notes |
|---|---|---|
| `hello` | `protocol` int, `core_version` string, `session_id` string, `capabilities` [string], `recording_content` bool | `recording_content` is true when the core runs with detailed recording opted in. The app shows a "Recording content" mark. |
| `state` | `assistant`, `hands_free` bool, `muted` bool, `detail` string? | `assistant` is one of `idle listening thinking acting speaking asking awaiting_approval error`. `detail` is a short machine reason such as `mic_unavailable`, never user content. Send on every change. |
| `transcript` | `utterance_id`, `role` (`user`/`assistant`), `text` string?, `partial` bool, `redacted` bool | A later message with the same `utterance_id` and `role` replaces the earlier one. `text` may be absent when the core withholds it (`redacted: true`). For live display only. |
| `speech` | `utterance_id`?, `phase` (`started`/`finished`/`interrupted`) | Assistant speech. `interrupted` means the user cut it off. |
| `level` | `source` (`mic`/`speaker`), `value` 0..1 | For the waveform. Send at about 10 to 20 per second and only while that source is active. The app clamps values. |
| `task` | `task_id`, `phase`, `step` int?, `summary` string?, `verified` bool? | See "Task phases". |
| `switch` | `role`, `from_slot`, `to_slot`?, `kind`, `reason` | One per core `SwitchEvent`. `to_slot` is null when nothing was left to try. Fields are the core's own strings. |
| `approval_request` | `approval_id`, `task_id`?, `kind`, `command`, `expires_in_s`? | See "Approvals". |
| `settings` | `revision` int, `settings` object | The reply to `settings_get` (with `reply_to`), or an unsolicited push when settings changed some other way. |
| `settings_result` | `ok` bool, `revision` int, `errors` [{`key`,`message`}] | The reply to `settings_set`. When `ok` is false nothing was applied. |
| `error` | `code`, `message`, `fatal` bool | With `reply_to`, it answers that request. `message` never contains keys, headers or request bodies. |
| `ping` | none | Optional keepalive. The app answers `pong`. |

### Task phases

`started`, `step`, `attempted`, `verified`, `unverified`, `reconcile_required`, `completed`, `failed`, `stopped`.

- `attempted` is an action that was sent. It is not an effect. The app shows it as "attempted, not yet checked".
- `verified` / `unverified` follow a fresh observation. `verified` carries `verified: true`, `unverified` carries `verified: false`.
- `reconcile_required` means a write has an unknown outcome. The app shows a warning and the pet asks a question. The core must not replay the write; it waits for the user.
- `summary` is a short label chosen by the core. It is untrusted data: the app shows it as plain text and never interprets it. The core leaves captured content out of it unless detailed recording is on.

### Approvals

`approval_request.kind` is one of `screen input app url file network spend other`. `command` is the exact command the core wants to run, as text.

- An approval covers that one command, once. There is no "always" and no scope field. The response vocabulary is `approve` or `deny`.
- `expires_in_s`, if set, is how long the core waits. When it passes, the core treats the request as denied and ignores a late response. The app also hides the card at the deadline.
- If the connection drops while an approval is open, the core treats it as denied. The app clears its cards on disconnect.
- The core ignores a response whose `approval_id` is unknown, already answered or expired, and ignores a second response to one id.
- With more than one connected client the core decides who may answer; the first valid response wins.
- The app never answers on its own. Only a click on the card sends `approval_response`.

## App to core

| `type` | `data` | Notes |
|---|---|---|
| `hello` | `protocol` int, `client` string, `client_version` string | First line of every connection. |
| `text_input` | `text` string | A typed request. Trimmed, never empty. Goes to the core only. |
| `interrupt` | none | Cut off the assistant's speech (barge-in). Does not stop a task. |
| `stop` | `task_id`?  | Stop the named task, or the current one, and any speech. |
| `approval_response` | `approval_id`, `decision` (`approve`/`deny`) | Answers one `approval_request`. |
| `voice_control` | `action` (`mute`/`unmute`) | Microphone mute. Hands-free on and off is a setting. |
| `settings_get` | none | Answered by `settings` with `reply_to`. |
| `settings_set` | `base_revision` int, `changes` [{`key`,`value`}] | Answered by `settings_result` with `reply_to`. |
| `pong` | none | Answer to `ping`. |

### Settings

The `settings` payload:

```json
{
  "voice":    {"hands_free":false,"headset":false,"silence_ms":600,"language":"en","silence_ms_range":[200,2000]},
  "privacy":  {"record_content":false},
  "computer": {"act_enabled":false},
  "roles": [
    {"role":"stt","pinned":null,"chain":[
      {"name":"<slot name>","provider":"<provider>","model":"<model>","key_env":"<VARIABLE_NAME>","key_present":true,"status":"ready"}
    ]}
  ]
}
```

- Every provider, model, role and slot name comes from the core, taken from `glide.toml`. The app has no list of its own and prints them as given. `status` is `ready`, `resting`, `skipped` or `unknown`.
- A slot reports the name of its key variable and whether it is set. **No credential value appears anywhere in this protocol, in either direction.** There is no message and no setting that could carry one.
- `silence_ms_range` is optional. When absent the app does not invent one.

`settings_set` changes are a closed set. The core may reject any of them, and must validate ranges itself:

| `key` | `value` |
|---|---|
| `voice.hands_free` | bool |
| `voice.headset` | bool |
| `voice.silence_ms` | int |
| `voice.language` | string |
| `privacy.record_content` | bool |
| `computer.act_enabled` | bool |
| `roles.pin` | `{"role":"<role>","slot":"<slot name>"}`, or `"slot": null` to clear |

`base_revision` is the `revision` the app last saw. The core refuses a change made against an older revision (`ok: false`, an error with the key `revision`), and the app re-reads. A successful change always bumps `revision`. After every `settings_set` the app re-reads with `settings_get` and shows only what the core reports.

## Rules both sides keep

- **No replay.** The app never queues a command while disconnected and never resends one after a reconnect. A request whose answer was lost (connection dropped, timeout) is reported as "outcome unknown", and the app asks for the current state instead of repeating it. The core must not replay a write whose outcome it does not know.
- **Visible fallbacks.** Every provider switch is a `switch` message and appears in the app. Nothing is switched silently.
- **Content.** `transcript`, `task.summary` and `approval_request.command` can contain what the user said or what was on screen. The app keeps them in memory only (bounded to 50 lines and 50 switches) and never writes them to disk or logs. `hello.recording_content` says whether the core stores content; changing that is the setting `privacy.record_content`, off by default.
- **Untrusted text.** Everything inside `data` that is text from a page, a task, a provider or a transcript is data. The app renders it verbatim and never follows instructions in it.
- **Errors** carry a short code and a message without secrets, and never echo the offending line.

## Example

```
app  {"v":1,"type":"hello","id":"a1","data":{"protocol":1,"client":"glide-app","client_version":"0.1.0"}}
core {"v":1,"type":"hello","data":{"protocol":1,"core_version":"x","session_id":"s1","capabilities":["voice"],"recording_content":false}}
app  {"v":1,"type":"settings_get","id":"a2"}
core {"v":1,"type":"settings","reply_to":"a2","data":{"revision":3,"settings":{...}}}
core {"v":1,"type":"state","data":{"assistant":"listening","hands_free":true,"muted":false}}
core {"v":1,"type":"transcript","data":{"utterance_id":"u1","role":"user","text":"open the","partial":true}}
core {"v":1,"type":"approval_request","data":{"approval_id":"p1","kind":"app","command":"open Notes","expires_in_s":30}}
app  {"v":1,"type":"approval_response","id":"a3","data":{"approval_id":"p1","decision":"approve"}}
core {"v":1,"type":"task","data":{"task_id":"t1","phase":"attempted","step":1}}
core {"v":1,"type":"task","data":{"task_id":"t1","phase":"verified","verified":true}}
```

The wire format is pinned by `Tests/GlideProtocolTests/CodecTests.swift` (round trips for every message, unknown types and enum values, version refusal, no secrets, framing).

## For the core's implementer

- Serve the socket from the core process that owns the microphone and the screen-driving loop. Microphone and screen permissions belong to that process, not to the app.
- Map `SwitchEvent` to `switch`, `ProviderError` kinds to `switch.kind` and `error.code`, and `ComputerTask` stop to `stop`.
- Emit `task` phases from the real observe/act/verify loop, so `verified` only follows a fresh observation.
- Build `settings` from the loaded config. Never put a key value, header or request body in any message.
- Add a test that the server refuses a second user's connection and a `v` it does not speak. Tests stay offline: use a socket in a temp directory (the Swift tests do), and add any new machine-reaching call to `tests/conftest.py` in the same change.
