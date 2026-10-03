"""The optional PySide6 pet: a floating raccoon over the same in-process assistant the command line uses.

Needs the `ui` extra (`uv sync --extra ui`). Importing this package starts no window, process or audio; each Qt
module imports PySide6 itself, so the pure-Python parts (`core`) work without it. It stays an optional extra until
the SwiftUI pet exists (decision D11).
"""
