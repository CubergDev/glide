"""Browser backend: the CDP client the execution backends attach through.

Importing this package imports only `cdp`. The older DOM loop (`runner`, `decide`, `perceive`, `report`, `act`) is not
production code and is no longer pulled in by it: import those modules by name. See docs/notes/legacy-browser-loop.md.
"""

from .cdp import CDPError, Session

__all__ = ["CDPError", "Session"]
