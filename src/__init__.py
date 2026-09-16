"""jacolor - Jacobian sparsity detection and coloring for PyTorch.

Pre-alpha. sparsity(f, x) reads a Jacobian pattern off a traced function for a
small set of ops and refuses anything it cannot vouch for. Coloring, seeding and
decompression then give the Jacobian in one AD pass per color (see README).
"""

from . import propagate
from .coloring import ORDERS, Coloring, color_cols, color_rows
from .compress import decompress, seeds
from .trace import CustomBackward, UnsupportedOp, sparsity

__version__ = "0.1.0.dev0"
__all__ = [
    "sparsity",
    "UnsupportedOp",
    "CustomBackward",
    "Coloring",
    "color_cols",
    "color_rows",
    "decompress",
    "propagate",
    "seeds",
    "ORDERS",
    "__version__",
]
