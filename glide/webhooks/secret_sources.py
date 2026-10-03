"""Where webhook signing secrets come from.

Keys come from environment variables named in the configuration (D4). A secret store is an optional adapter
behind this one interface: `EnvSecrets` is the default and `KeyringSecrets` imports `keyring` only when it is
constructed, so the core and the tests never need it. Tests use `DictSecrets`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Protocol


class SecretSource(Protocol):
    def get(self, name: str) -> str:
        """The secret called `name`, or "" when it is not set. Never raises with the value in the message."""


class EnvSecrets:
    def get(self, name: str) -> str:
        return os.environ.get(name, "")


class DictSecrets:
    """An in-memory source for tests and for callers that already hold the values."""

    def __init__(self, values: Mapping[str, str] | None = None):
        self._values = dict(values or {})

    def get(self, name: str) -> str:
        return self._values.get(name, "")


class KeyringSecrets:
    """The operating system's keyring, an explicit opt-in. `name` is the credential's account name."""

    def __init__(self, service: str, *, backend=None):
        if not service:
            raise ValueError("A keyring service name is required.")
        self.service = service
        if backend is None:
            try:
                import keyring as backend
            except ImportError:
                raise ValueError("The keyring package is not installed.") from None
        self._backend = backend

    def get(self, name: str) -> str:
        try:
            return self._backend.get_password(self.service, name) or ""
        except Exception:  # a backend failure must not put the account, service or value in a message
            raise ValueError("The keyring could not be read.") from None
