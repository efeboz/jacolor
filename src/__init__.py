"""jacolor - Jacobian sparsity detection and coloring for PyTorch.

Pre-alpha. Only the pattern-storage and coloring core is implemented; the
tracing frontend and the operator rule set are not (see README).
"""

from .coloring import ORDERS, Coloring, color_cols, color_rows

__version__ = "0.1.0.dev0"
__all__ = ["Coloring", "color_cols", "color_rows", "ORDERS", "__version__"]
