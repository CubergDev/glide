"""The writer boundary's guard works, and the code that owns the boundary holds no vendor or model literal (D6)."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("guards_writer_boundary", HERE / "guards_writer-boundary.py")
_guards = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_guards)
no_vendor_sdk_client = _guards.no_vendor_sdk_client  # a fixture defined here is applied to this module


def test_building_a_vendor_sdk_client_in_a_test_fails_loudly():
    import openai

    with pytest.raises(AssertionError, match="OpenAI SDK client"):
        openai.OpenAI(api_key="x", base_url="http://127.0.0.1:9")


OWNED = (
    "glide/computer/writer.py",
    "glide/computer/calls.py",
    "glide/providers/writer_client.py",
    "glide/providers/errors.py",
    "glide/providers/llm.py",
    "glide/providers/classifier.py",
    "glide/providers/config.py",
)
# A model id, a vendor host, an OAuth endpoint or a voice id is configuration. glide/providers/config.py keeps the
# presets (vendor hosts and key variable names) and the built-in default chains, which are documented as the team's
# unverified starting point, so those lines are the only ones allowed to name them.
LITERAL = re.compile(r"claude-|gpt-|gemini-|deepseek-v|chatgpt|opencode\.ai|oauth|eleven_v|silero|jbfqnc", re.IGNORECASE)


@pytest.mark.parametrize("path", OWNED)
def test_the_boundary_code_names_no_model_voice_or_oauth_literal(path):
    root = HERE.parent
    lines = (root / path).read_text().splitlines()
    inside_defaults = False
    found = []
    for number, line in enumerate(lines, 1):
        if path.endswith("providers/config.py"):
            if line.startswith('DEFAULT_TOML = """'):
                inside_defaults = True
            elif inside_defaults and line.startswith('"""'):
                inside_defaults = False
            if inside_defaults or line.lstrip().startswith(("#", '"openai"', '"gemini"', '"deepseek"', '"openrouter"')):
                continue
        if LITERAL.search(line):
            found.append(f"{path}:{number}: {line.strip()}")
    assert not found, "\n".join(found)


def test_no_module_of_glide_imports_a_vendor_sdk():
    """The vendor SDKs read credentials and hosts from the process environment by themselves, which keys-from-glide.toml forbids."""
    import ast

    banned = {"anthropic", "openai"}
    found = []
    for path in sorted((HERE.parent / "glide").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom) and not node.level
                else []
            )
            found += [f"{path.relative_to(HERE.parent)}:{node.lineno}: {n}" for n in names if n.split(".")[0] in banned]
    assert not found, "\n".join(found)
