"""One smart router: `Router.route(utterance, context) -> Decision`. See docs/ROUTER.md.

Importing this package reaches no machine, no network and no provider: every client is handed in.
"""

from .build import build_router
from .calibration import Calibration, Sample, expected_calibration_error
from .clarify import Clarifier, Resolution, resolve
from .decision import ACTING, ROUTES, Context, Decision, Span
from .router import Router
from .settings import RoutingSettings
from .stop import is_stop, normalize, stop_phrases

__all__ = [
    "ACTING",
    "ROUTES",
    "Calibration",
    "Clarifier",
    "Context",
    "Decision",
    "Resolution",
    "Router",
    "RoutingSettings",
    "Sample",
    "Span",
    "build_router",
    "expected_calibration_error",
    "is_stop",
    "normalize",
    "resolve",
    "stop_phrases",
]
