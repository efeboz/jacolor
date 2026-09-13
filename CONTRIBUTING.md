# Contributing

jacolor's value is that its sparsity patterns are *never* missing an entry. A
too-dense pattern costs time; a too-sparse one silently returns a wrong
Jacobian. Every rule below exists to keep that asymmetry visible.

## Operating rules

The project's operating rules, restated as things a reviewer checks.

1. **A task is done when its "done when" criterion passes, not when code exists.**
2. **A failing test is fixed in the code.** Loosening a test, a tolerance, or a
   tightness claim requires a `docs/decisions/` entry giving the reason.
3. **Every operator rule PR ships:** the registry entry with its tightness, a
   Hypothesis soundness test (random shapes and args, local Jacobian at three
   random points, nonzeros ⊆ pattern), an equality test if the tightness is
   `exact`, and a regenerated supported-operations table.
4. **Every error carries the op overload** (or the function's qualified name)
   **and a source location.** Test the message, not just the exception type.
5. **Invariants that are linted, not trusted:**
   - Concrete values are read only through `.structural()`, never `.tensor` —
     a rule that prunes on a weight zero is unsound the moment the weight changes.
   - `_boolcsr.py` is bool-only (`tests/test_invariants.py`).
   - The ignore list for unknown ops is an explicit allowlist; anything else
     with a tracked input goes to dense fallback, never to "ignore".
6. **`verify=True` is not disabled in the test suite**, except in tests of the
   verification path itself.
7. **README and docs quote measured numbers with machine details.** No
   extrapolation, no "should scale to".
8. **Roadmap items do not land in v0.1.** A reserved API shape may exist;
   the behaviour raises `NotImplementedError` pointing at the roadmap.
9. **Spike findings are written down before the spike branch is deleted:**
   `docs/decisions/0001-spike.md` and `0002-frontend.md`. Spike code is not
   merged; it is rewritten under tests.
10. **`asdex` is the reference for API expectations** (`argnums`, `chunk_size`,
    save/load, verification against vanilla AD). Diverge only where the
    correctness contract demands it, and document the divergence.

## Prerequisites for the tracing frontend

No frontend code lands before `docs/decisions/0002-frontend.md` records the
zoo's pass/fail table for both candidates (`torch.export` and a real-tensor
tracer). The rule engine is keyed on ATen op overloads and must stay
frontend-agnostic: one rule set, two adapters.

## Running the tests

```bash
pip install -e ".[dev,fast]"
pytest
```

## Decision records

Anything that changes what the library is allowed to do — a frontend choice, a
loosened tolerance, a rule downgraded to `conservative` — gets a numbered file
in `docs/decisions/`. Copy `0000-template.md`.
