"""Opt-in memory, context planning and a bounded tool harness for Glide.

Importing this package performs no I/O and opens nothing. Memory is OFF by default; see `settings.py` for how
the user turns it on and docs/memory.md for exactly what is stored. The core has no third-party dependencies.

Note on names: `planning.plan` selects what context, skills and tools fit a model's window for one turn. It is not
the intent router (stop / answer / act); that is a separate component.
"""

from .catalog import Catalog
from .contracts import Model, Plan, Policy, Scope, Tool
from .events import Event, EventBus, SQLiteEventSink
from .harness import Harness, Reply, Request, ToolCall
from .runs import ingest_run
from .service import MemoryService
from .settings import MemorySettings, SettingsError
from .store import Store

__all__ = [
    "Catalog",
    "Event",
    "EventBus",
    "Harness",
    "MemoryService",
    "MemorySettings",
    "Model",
    "Plan",
    "Policy",
    "Reply",
    "Request",
    "SQLiteEventSink",
    "Scope",
    "SettingsError",
    "Store",
    "Tool",
    "ToolCall",
    "ingest_run",
]
