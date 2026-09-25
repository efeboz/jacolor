"""jacolor - Jacobian sparsity detection and coloring for PyTorch.

Pre-alpha. jacobian(f, x) gives the sparse Jacobian in one AD pass per color,
and prepare(f, x) does the tracing and coloring once for a loop that evaluates
many times. sparsity(f, x) reads the pattern off a traced function for a small
set of ops and refuses anything it cannot vouch for (see README).
"""

from . import propagate
from .coloring import ORDERS, Coloring, color_cols, color_rows, refine
from .compress import decompress, seeds
from .evaluate import VerificationError, VerificationInconclusive, jacobian
from .interop import pattern, pattern_from_pairs, to_scipy
from .analysis import Prepared, prepare
from .trace import (CustomBackward, TraceMismatch, TraceUnchecked, UnsupportedOp,
                    sparsity, supported_ops)

__version__ = "0.1.0.dev0"
__all__ = [
    "prepare",
    "Prepared",
    "jacobian",
    "sparsity",
    "supported_ops",
    "pattern",
    "pattern_from_pairs",
    "to_scipy",
    "UnsupportedOp",
    "CustomBackward",
    "TraceMismatch",
    "TraceUnchecked",
    "VerificationError",
    "VerificationInconclusive",
    "Coloring",
    "color_cols",
    "color_rows",
    "refine",
    "decompress",
    "seeds",
    "propagate",
    "ORDERS",
    "__version__",
]
