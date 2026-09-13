import pathlib
import re

import numpy as np

from src import _boolcsr as bc

PKG = pathlib.Path(__file__).resolve().parents[1] / "jacolor"

# _boolcsr.py is bool-only: pattern values are bool, index arrays are int64,
# nothing else. int8/uint8 values are the exact wrap hazard, and a float dtype
# has no business here at all.
_BANNED = re.compile(
    r"\b(?:np|numpy)\.(?!bool_?\b|int64\b)"
    r"(u?int\d*|float\d*|complex\d*|u?byte|short|longlong|double|single)\b"
)


def test_boolcsr_uses_no_dtype_but_bool_and_int64():
    for i, line in enumerate((PKG / "_boolcsr.py").read_text().splitlines(), 1):
        code = line.split("#", 1)[0]
        hit = _BANNED.search(code)
        assert hit is None, f"_boolcsr.py:{i} uses banned dtype {hit.group(0)!r}"


def test_every_boolcsr_entry_point_returns_bool():
    P = bc.from_dense([[1, 0], [1, 1]])
    Q = bc.from_pairs([0, 1], [1, 0], shape=(2, 2))
    for M in (P, Q, bc.transpose(P), bc.matmul(P, Q), bc.check(P)):
        assert M.dtype == np.bool_
