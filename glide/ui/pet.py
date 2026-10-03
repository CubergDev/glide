"""`python -m glide.ui`: the floating raccoon. Importing this module never opens a window, audio or a process."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..speech.settings import MAX_SILENCE_MS, MIN_SILENCE_MS


def open_window(core, config) -> int:
    """The one door to a real window; lazy so that the pure parts of glide.ui import without PySide6."""
    from .pet_ui import show_pet

    return show_pet(core, config)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m glide.ui",
        description="Floating raccoon for Glide. Starts passive: no microphone, request or task begins until you ask.",
    )
    parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="where .env is read and the core's run folders go")
    parser.add_argument("--config", default=None, help="the glide.toml to use (default: found as `glide doctor` finds it)")
    parser.add_argument("--silence-ms", type=int, default=None, help="how long a pause ends a spoken turn (default: [speech])")
    parser.add_argument(
        "--record-content",
        action="store_true",
        help="let the core keep what is said, typed and read for debugging (off by default; the pet shows an indicator)",
    )
    args = parser.parse_args(argv)
    if args.silence_ms is not None and not MIN_SILENCE_MS <= args.silence_ms <= MAX_SILENCE_MS:
        parser.error(f"--silence-ms must be between {MIN_SILENCE_MS} and {MAX_SILENCE_MS}")
    workspace = args.workspace.expanduser().resolve()
    if not workspace.is_dir():
        parser.error("Workspace must be an existing directory.")

    from ..computer.config import load_dotenv
    from ..providers.config import ConfigError, load_config
    from .core import PetCore

    load_dotenv(workspace / ".env")
    try:
        config = load_config(args.config, cwd=workspace)
    except (ConfigError, OSError) as error:
        parser.error(f"could not load the configuration: {error}")
    core = PetCore(config, runs_dir=workspace / "runs", record_content=args.record_content)
    if args.silence_ms is not None:
        core.silence_ms = args.silence_ms
    try:
        return open_window(core, config)
    except ImportError:
        parser.error("The pet needs the optional ui dependencies: uv sync --extra ui")
