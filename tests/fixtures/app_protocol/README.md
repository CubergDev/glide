# Golden lines of the app protocol

One JSON envelope per file, the same bytes for both sides.

- `core/*.json`: what the core sends. `tests/app_server/test_app_wire.py` builds each one with `glide.app_server.wire` and compares.
  The Swift codec must decode each one.
- `app/*.json`: what the app sends. The core's parser must accept each one, and the Swift codec must decode and re-encode it.
- `FixtureTests.swift`: the Swift test that decodes both directories. It belongs in `app/Tests/GlideProtocolTests/` (it finds this
  directory three levels above its own, which is the repository root). It was run against the app scaffold on
  `consolidation/app-swiftui` with `./scripts/test.sh --filter FixtureTests`: four tests, all passing.

Two lines are additions inside version 1 that an older app ignores: `core/approval_closed.json` is a message type the Swift codec
reads as unknown, and `chars` in `core/transcript_*redacted.json` is a field it does not declare.
