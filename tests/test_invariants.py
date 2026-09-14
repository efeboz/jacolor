import pathlib
import re

import numpy as np

from src import _boolcsr as bc

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "src"

# Files that must decode as UTF-8. Extension list rather than "everything",
# so a stray binary in the tree does not fail the suite.
_TEXT = {".py", ".md", ".toml", ".yml", ".yaml", ".cfg", ".txt", ".ini"}
_SKIP = {".git", "__pycache__", ".pytest_cache", ".venv", "build", "dist"}

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


def _text_files():
    for p in sorted(ROOT.rglob("*")):
        if p.is_dir() or _SKIP & set(p.parts):
            continue
        if p.suffix in _TEXT or p.name in {"LICENSE", ".gitignore", ".gitattributes"}:
            yield p


def test_every_text_file_is_utf8():
    # Encoding drift is invisible until an editor writes latin-1 and CI on
    # another machine cannot read the file. A BOM breaks shebangs and parsers.
    for p in _text_files():
        raw = p.read_bytes()
        rel = p.relative_to(ROOT)
        assert not raw.startswith(b"\xef\xbb\xbf"), f"{rel} starts with a UTF-8 BOM"
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError as e:
            raise AssertionError(f"{rel} is not valid UTF-8: {e}") from None


def test_source_stays_ascii():
    # Markdown may use math symbols. Source may not: a stray non-ascii character
    # in code is nearly always a paste accident.
    for p in _text_files():
        if p.suffix != ".py":
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            bad = [c for c in line if ord(c) > 127]
            assert not bad, f"{p.relative_to(ROOT)}:{i} has non-ascii {bad}"
