"""A file planner that runs safely: plan, preview, execute with an exact-hash approval, verify, undo.
Off by default: nothing else in Glide calls it."""

from . import engine
from .engine import execute, undo
from .model import ActionResult, Approval, Move, Plan, Report
from .planner import DEFAULT_CATEGORIES, NAMED_FOLDERS, ORGANIZE_BY_TYPE, plan, preview
from .safety import Refused

__all__ = [
    "DEFAULT_CATEGORIES", "NAMED_FOLDERS", "ORGANIZE_BY_TYPE", "ActionResult", "Approval", "Move", "Plan", "Refused",
    "Report", "engine", "execute", "plan", "preview", "undo",
]  # fmt: skip
