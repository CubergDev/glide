"""Load the shared guard for the memory tests. The guard file's name has a hyphen, so it is loaded by path."""

import importlib.util
import sys
from pathlib import Path

_KEY = "guards_memory_mcp"
if _KEY not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_KEY, Path(__file__).resolve().parents[1] / "guards_memory-mcp.py")
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_KEY] = _module
    _spec.loader.exec_module(_module)

no_real_memory_or_mcp = sys.modules[_KEY].no_real_memory_or_mcp
