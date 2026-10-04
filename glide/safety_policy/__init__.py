"""A pure policy table for proposed UI actions: allow, ask, handoff or refuse. `decide` is the one entry point."""

from .lexicon import Lexicon, load_lexicon, normalize
from .policy import Action, Decision, Reason, Verdict, decide

__all__ = ["Action", "Decision", "Lexicon", "Reason", "Verdict", "decide", "load_lexicon", "normalize"]
