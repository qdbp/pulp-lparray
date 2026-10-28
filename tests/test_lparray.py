import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import ArrayLike
from pulp import COIN_CMD, LpBinary, LpProblem, lpSum

from pulp_lparray import lparray


def solve(prob: LpProblem) -> None:
    assert prob.solve(COIN_CMD(msg=False)).has_solution


def pinned(prob: LpProblem, values: ArrayLike) -> lparray:
    """Variables fixed to `values`, so that any formulation over them has one solution."""
    x = lparray.create(prob, "x", np.shape(values))
    (x == values).constrain(prob, "pin")
    return x


def adversaries(prob: LpProblem, *outputs: lparray) -> Iterator[None]:
    """Solve under a random objective over `outputs` and then under its negation, so that an
    output its formulation leaves free strays from its oracle."""
    rng = np.random.default_rng(0)
    push = lpSum((rng.uniform(-1, 1, o.shape) * o).sum().item() for o in outputs)
    for sign in (1, -1):
        prob.setObjective(sign * push)
        solve(prob)
        yield


def test_linear_algebra() -> None:
    # lparray sums and multiplies in time linear in the contracted length, where NumPy's object
    # loops chain PuLP's +, which copies its left operand. The results must match NumPy's.
    prob = LpProblem()
    rng = np.random.default_rng(5)
    a = rng.integers(-9, 10, size=(2, 3, 4))
    m = rng.integers(-9, 10, size=(4, 5))
    b = rng.integers(-9, 10, size=(2, 4, 5))
    x = pinned(prob, a)
    solve(prob)
    for got, want in [
        (np.add.reduce(x), np.add.reduce(a)),
        (x.sum(), a.sum()),
        (x.sum(axis=(0, -1), keepdims=True), a.sum(axis=(0, -1), keepdims=True)),
        (x @ m, a @ m),
        (m.T @ x.swapaxes(1, 2), m.T @ a.swapaxes(1, 2)),
        (x @ m[:, 0], a @ m[:, 0]),
        (m[:, 0] @ x[0].T, m[:, 0] @ a[0].T),
        (np.dot(x[0], b), np.dot(a[0], b)),
        (x[0, 0].dot(m[:, 0]), a[0, 0].dot(m[:, 0])),
    ]:
        assert isinstance(got, lparray)
        assert_allclose(got.values, want)
    with pytest.raises(ValueError, match="contract"):
        x @ m[:3]
    with pytest.raises(TypeError):  # NumPy would take PuLP comparisons for truth values
        np.maximum(x, 0)


def test_logic() -> None:
    prob = LpProblem()
    bits = np.random.default_rng(2).integers(0, 2, size=(5, 4))
    b = pinned(prob, bits)
    any_ = b.lp_bin_max(prob, "any", axis=1)
    all_ = b.lp_bin_min(prob, "all", axis=0)
    both = lparray.create(prob, "both", 5, cat=LpBinary)
    both.lp_bin_and(prob, "and", b[:, 0], 1 - bits[:, 1])
    either = lparray.create(prob, "either", 5, cat=LpBinary)
    either.lp_bin_or(prob, "or", b[:, 2], bits[:, 3])
    # Inputs broadcast to the output's shape: broadcast only against each other, these two
    # scalars would form one input of length 2.
    never = lparray.create(prob, "never", 2, cat=LpBinary)
    never.lp_bin_and(prob, "never", 0, 1)
    for _ in adversaries(prob, any_, all_, both, either, never):
        assert_allclose(any_.values, bits.any(axis=1))
        assert_allclose(all_.values, bits.all(axis=0))
        assert_allclose(both.values, bits[:, 0] & (1 - bits[:, 1]))
        assert_allclose(either.values, bits[:, 2] | bits[:, 3])
        assert_allclose(never.values, [0, 0])


def test_extrema() -> None:
    prob = LpProblem()
    vals = np.random.default_rng(0).integers(-9, 10, size=(6, 5))
    x = pinned(prob, vals)
    rowmax = x.lp_max(prob, "rowmax", axis=1, bigM=18)
    colmin = x.lp_min(prob, "colmin", axis=0, bigM=18)
    top = x.lp_max(prob, "top", bigM=18)
    argmax = x.lp_argmax(prob, "argmax", axis=1, bigM=18)
    argmin = x.lp_argmin(prob, "argmin", axis=0, bigM=18)
    top2 = x.lp_topk(prob, "top2", k=2, axis=1, bigM=18)
    for _ in adversaries(prob, rowmax, colmin, top, argmax, argmin, top2):
        assert_allclose(rowmax.values, vals.max(axis=1))
        assert_allclose(colmin.values, vals.min(axis=0))
        assert_allclose(top.values, vals.max())
        w = argmax.values.round()
        assert (w.sum(axis=1) == 1).all()
        assert (vals[w == 1] == vals.max(axis=1)).all()
        w = argmin.values.round()
        assert (w.sum(axis=0) == 1).all()
        assert (vals.T[w.T == 1] == vals.min(axis=0)).all()
        w = top2.values.round()
        assert (w.sum(axis=1) == 2).all()
        chosen, rest = np.where(w, vals, np.inf), np.where(w, -np.inf, vals)
        assert (chosen.min(axis=1) >= rest.max(axis=1)).all()


def test_abs_and_logical_clip() -> None:
    prob = LpProblem()
    vals = np.random.default_rng(1).integers(-5, 6, size=(4, 6))
    x = pinned(prob, vals)
    p, m = x.abs_decompose(prob, "parts", bigM=5)
    magnitude = x.abs(prob, "abs", bigM=5)
    nonzero = magnitude.logical_clip(prob, "nonzero", bigM=5)
    for _ in adversaries(prob, p, m, magnitude, nonzero):
        assert_allclose(p.values, np.maximum(vals, 0), atol=1e-9)
        assert_allclose(m.values, np.maximum(-vals, 0), atol=1e-9)
        assert_allclose(magnitude.values, np.abs(vals), atol=1e-9)
        assert_allclose(nonzero.values, vals != 0)


def test_interp() -> None:
    prob = LpProblem()
    rng = np.random.default_rng(3)
    xs = rng.uniform(0, 10, size=7)
    x = pinned(prob, xs)
    xp, fp = [0, 2, 5, 10], [0, 4, 1, 3]  # nonconvex, so only adjacency pins the weights
    shared = x.lp_interp(prob, "shared", xp=xp, fp=fp)
    xps = np.sort(rng.uniform(0, 10, size=(7, 4)), axis=1)  # breakpoints for each element
    xps[:, [0, -1]] = 0, 10
    fps = rng.uniform(0, 5, size=(7, 4))
    each = x.lp_interp(prob, "each", xp=xps, fp=fps)
    for _ in adversaries(prob, shared, each):
        assert_allclose(shared.values, np.interp(xs, xp, fp), atol=1e-6)
        assert_allclose(each.values, list(map(np.interp, xs, xps, fps)), atol=1e-6)


def test_pulp_scalar_operands() -> None:
    # PuLP 4 scalars set __array_priority__ = 20, and their reflected arithmetic sums an array
    # operand into one expression; == between two variables tests identity; and expressions
    # define __len__ and __getitem__, so NumPy reads a bare one as the sequence of its
    # variables. Each result below must stay elementwise.
    prob = LpProblem()
    x, y = lparray.create(prob, "x", 3), lparray.create(prob, "y", 3)
    v = prob.add_variable("v")
    e = 2 * v + 1
    filled = np.zeros(3, dtype=object).view(lparray)
    filled[1:] = e
    values = np.random.default_rng(4).uniform(size=7)
    for var, val in zip([*x.flat, *y.flat, v], values.tolist(), strict=True):
        var.varValue = val
    X, Y, V = values[:3], values[3:6], values[6]
    E = 2 * V + 1
    mask = [True, False, True]
    for got, want in [
        (x + v, X + V),
        (x - e, X - E),
        (2 * x + e, 2 * X + E),
        (np.where(mask, x, e), np.where(mask, X, E)),  # ty: ignore[no-matching-overload]
        (np.choose([0, 1, 0], [e, x]), np.choose([0, 1, 0], [E, X])),
        (filled, [0, E, E]),
        (x <= v, X - V),
        (x >= e, X - E),
        (x == y, X - Y),
        (v <= x, X - V),
        (e <= x, X - E),
    ]:
        assert isinstance(got, lparray)
        assert_allclose(got.values, want)
    (x == y).constrain(prob, "eq")
    (x.sum() <= e).constrain(prob, "total")


def test_readme() -> None:
    readme = (Path(__file__).parents[1] / "README.md").read_text()
    namespace: dict[str, object] = {}
    for block in re.findall(r"^```python\n(.*?)^```", readme, re.DOTALL | re.MULTILINE):
        exec(block, namespace)  # noqa: S102 (the README's own examples)
