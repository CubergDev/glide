"""Make the shared guard (tests/guards_memory-mcp.py, whose hyphenated name cannot be imported) available to this directory's tests.

Not a conftest on purpose: several conftest.py files without packages share the module name "conftest" and would
shadow tests/conftest.py. Each test module here imports the fixture from this loader.
"""

import asyncio
import functools
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


def sync(test):
    """Run an `async def` test to completion (pytest-asyncio is not a dependency)."""

    @functools.wraps(test)
    def run(*args, **kwargs):
        return asyncio.run(test(*args, **kwargs))

    return run
