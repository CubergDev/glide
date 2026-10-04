"""`glide setup`: the first-run wizard. A loopback page (or plain prompts with --text) that writes glide.toml and
never writes a key."""

from .cli import main

__all__ = ["main"]
