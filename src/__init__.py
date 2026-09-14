"""jacolor - Jacobian sparsity detection and coloring for PyTorch.

Pre-alpha. Pattern storage, propagation rules, coloring, seeding and
decompression work. What is missing is the part that walks a PyTorch function
and applies the rules for you (see README).
"""

from . import propagate
from .coloring import ORDERS, Coloring, color_cols, color_rows
from .compress import decompress, seeds

__version__ = "0.1.0.dev0"
__all__ = [
    "Coloring",
    "color_cols",
    "color_rows",
    "decompress",
    "propagate",
    "seeds",
    "ORDERS",
    "__version__",
]
