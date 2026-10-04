"""Clauses: split one utterance into ordered steps, and step through a short lesson. Pure logic, no model, no screen.

`split_clauses(text)` is the entry point for the splitter and `LessonStepper` for the stepper. Nothing here is wired
into the router yet: a caller passes the transcript in and gets a plan or a step to present back.
"""

from __future__ import annotations

from glide.clauses.lesson import LessonResult, LessonStepper, parse_control
from glide.clauses.split import Plan, Step, split_clauses

__all__ = ["LessonResult", "LessonStepper", "Plan", "Step", "parse_control", "split_clauses"]
