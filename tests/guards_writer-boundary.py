"""Guard for the writer boundary: no vendor SDK client may be built in a test.

The writer reaches its models only through the provider chains (glide/providers), which speak httpx and are
already covered by conftest's loopback-only socket guard. What the chains replaced were the vendor SDK clients,
which pick up credentials and hosts from the process environment by themselves. Nothing in glide/ builds one any
more, so building one in a test is a mistake this fixture turns into a failure instead of a request.

The filename has a hyphen, so it cannot be imported by name: tests/conftest.py loads it with importlib and
re-exports `no_vendor_sdk_client`, or a test does so for itself (tests/test_writer_boundary_guard.py).
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def no_vendor_sdk_client(monkeypatch):
    def refuse(vendor: str):
        def build(*args, **kwargs):
            from conftest import refusal

            raise refusal(
                f"the {vendor} SDK client",
                kind=AssertionError,
                message=f"a test built the {vendor} SDK client; the writer goes through glide.providers",
            )

        return build

    for module_name, attribute, vendor in (("anthropic", "Anthropic", "Anthropic"), ("openai", "OpenAI", "OpenAI")):
        try:
            module = __import__(module_name)
        except ImportError:
            continue
        monkeypatch.setattr(getattr(module, attribute), "__init__", refuse(vendor))
