"""jacolor - Jacobian sparsity detection and coloring for PyTorch.

Pre-alpha. Pattern storage, coloring, seeding and decompression work. The
tracing frontend and the operator rule set do not exist yet (see README).
"""

from .coloring import ORDERS, Coloring, color_cols, color_rows
from .compress import decompress, seeds

__version__ = "0.1.0.dev0"
__all__ = [
    "Coloring",
    "color_cols",
    "color_rows",
    "decompress",
    "seeds",
    "ORDERS",
    "__version__",
]
