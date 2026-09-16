"""jacolor - Jacobian sparsity detection and coloring for PyTorch.

Pre-alpha. jacobian(f, x) gives the sparse Jacobian in one AD pass per color.
sparsity(f, x) reads the pattern off a traced function for a small set of ops and
refuses anything it cannot vouch for (see README).
"""

from . import propagate
from .coloring import ORDERS, Coloring, color_cols, color_rows
from .compress import decompress, seeds
from .evaluate import VerificationError, jacobian
from .trace import CustomBackward, TraceMismatch, UnsupportedOp, sparsity

__version__ = "0.1.0.dev0"
__all__ = [
    "jacobian",
    "sparsity",
    "UnsupportedOp",
    "CustomBackward",
    "TraceMismatch",
    "VerificationError",
    "Coloring",
    "color_cols",
    "color_rows",
    "decompress",
    "propagate",
    "seeds",
    "ORDERS",
    "__version__",
]
