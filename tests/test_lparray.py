from collections.abc import Iterator

import numpy as np
import numpy.random as npr
import pulp as pp
import pytest
from numpy.testing import assert_allclose
from numpy.typing import ArrayLike
from pulp import LpBinary, LpProblem

from examples.logistics_facility_topk_pwl import build_demo_model
from pulp_lparray import lparray


def solve(prob: pp.LpProblem) -> None:
    assert prob.solve(pp.COIN_CMD(msg=False)) == pp.LpStatusOptimal


def pinned(prob: LpProblem, values: ArrayLike) -> lparray:
    """Variables fixed to `values`, so that any formulation over them has one solution."""
    x = lparray.create(prob, "x", np.shape(values))
    (x == values).constrain(prob, "pin")
    return x


def adversaries(prob: LpProblem, *outputs: lparray) -> Iterator[None]:
    """Solve under a random objective over `outputs` and then under its negation, so that an
    output its formulation leaves free strays from its oracle."""
    rng = np.random.default_rng(0)
    push = sum((rng.uniform(-1, 1, o.shape) * o).sum() for o in outputs).item()
    for sign in (1, -1):
        prob.setObjective(sign * push)
        solve(prob)
        yield


def test_super_sudoku() -> None:
    """
    A Supersudoku board is a Sudoku board with the additional requirement that
    no single digit can be found at the same coordinates in two boxes.
    """

    def check_super_sudoku(arr: np.ndarray) -> bool:
        all_digits = frozenset([1, 2, 3, 4, 5, 6, 7, 8, 9])
        flat = arr.argmax(axis=-1) + 1

        for col in range(9):
            assert set(flat[:, col // 3, :, col % 3].ravel()) == all_digits

        for row in range(9):
            assert set(flat[row // 3, :, row % 3, :].ravel()) == all_digits

        for box in range(9):
            assert set(flat[box // 3, box % 3, :, :].ravel()) == all_digits

        for xy in range(9):
            assert set(flat[:, :, xy // 3, xy % 3].ravel()) == all_digits

        return True

    prob = pp.LpProblem("SuperSudoku", pp.LpMinimize)
    X = lparray.create(prob, "Board", (3, 3, 3, 3, 9), cat=pp.LpBinary)
    (X.sum(axis=-1) == 1).constrain(prob, "OneDigitPerCell")
    (X.sum(axis=(1, 3)) == 1).constrain(prob, "MaxOnePerRow")
    (X.sum(axis=(0, 2)) == 1).constrain(prob, "MaxOnePerCol")
    (X.sum(axis=(2, 3)) == 1).constrain(prob, "MaxOnePerBox")
    (X.sum(axis=(0, 1)) == 1).constrain(prob, "MaxOnePerDust")
    solve(prob)

    assert check_super_sudoku(X.values)


# noinspection PyArgumentList
def test_logical_clip() -> None:

    prob = pp.LpProblem("logical_clip", pp.LpMinimize)
    X = lparray.create(prob, "arr", (5, 5), cat=pp.LpInteger, lowBound=0, upBound=5)
    (X.sum(axis=1) >= 6).constrain(prob, "colsum")
    (X.sum(axis=0) >= 6).constrain(prob, "rowsum")

    lclip = X.logical_clip(prob, "lclip")

    # dodging these
    bern = npr.default_rng().binomial(3, 0.5, size=(5, 5))

    prob += (X * bern).sum().item()
    solve(prob)

    assert X.values.max() > 1
    assert lclip.values.max() <= 1
    assert lclip.values.max() >= 0


def test_abs() -> None:

    N = 20

    prob = pp.LpProblem("wavechaser", pp.LpMaximize)
    X = lparray.create(prob, "arr", (N,), cat=pp.LpInteger, lowBound=-1, upBound=1)
    wave = 2 * npr.default_rng().binomial(1, 0.5, size=(N,)) - 1

    xp, xm = X.abs_decompose(prob, "abs")
    xabs = xp + xm

    prob += (wave * X).sum().item()
    solve(prob)

    assert prob.objective == N
    assert xabs.values.sum() == N

    assert np.all(xp.values * wave >= 0)
    assert np.all(-xm.values * wave >= 0)


def test_logical_clip_again_because_i_forgot_i_already_had_a_test() -> None:
    """
    Binary knapsack problem, implemented inefficiently.
    """

    boxes_values = np.array([20, 10, 5, 1, 0])
    box_sizes = np.array([15, 10, 10, 5, 1])

    capacity = 30

    prob = pp.LpProblem(sense=pp.LpMaximize)

    taken = lparray.create(prob, "is_taken_base", box_sizes.shape, cat=pp.LpInteger)
    clipped = taken.logical_clip(prob, "is_taken")

    (box_sizes @ clipped <= capacity).constrain(prob, "MaxCapacity")
    prob += (clipped * boxes_values).sum().item()
    solve(prob)

    assert np.allclose(clipped.values, [1, 1, 0, 1, 0])


def test_topk() -> None:
    prob = pp.LpProblem("topk", pp.LpMaximize)
    X = lparray.create(prob, "arr", (6,), cat=pp.LpInteger, lowBound=0, upBound=10)

    vals = np.array([1, 7, 10, 4, 9, 2])
    for i, v in enumerate(vals):
        c = X[i] == v
        c.name = f"fix{i}"
        prob += c

    w = X.lp_bin_topk(prob, "tk", axis=0, k=2, bigM=100)
    solve(prob)

    picked = set(np.where(w.values > 0.5)[0].tolist())
    assert picked == {2, 4}


def test_piecewise_linear_sos2() -> None:
    prob = pp.LpProblem("pwl", pp.LpMinimize)
    dummy = lparray.create(prob, "d", (), cat=pp.LpContinuous)
    x, y = dummy.piecewise_linear_sos2(prob, "p", x=[0, 1, 2], y=[0, 1, 4])

    # pick x=1 => y=1
    c = x.item() == 1
    c.name = "fixx"
    prob += c
    prob += y.item()
    solve(prob)
    assert abs(y.values.item() - 1) < 1e-6


def test_logistics_demo_model() -> None:
    prob, vars_ = build_demo_model()
    solve(prob)

    open_ = vars_["open"].values
    assert open_.sum() == 4
    assert open_[:3].sum() == 2
    assert open_[3:].sum() == 2

    flow = vars_["flow"].values
    demand = np.array(
        [
            [4, 2, 1],
            [0, 3, 2],
            [1, 1, 0],
            [3, 0, 1],
            [2, 2, 2],
        ],
        dtype=float,
    )
    assert np.allclose(flow.sum(axis=0), demand)


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
    for _ in adversaries(prob, rowmax, colmin, top, argmax, argmin):
        assert_allclose(rowmax.values, vals.max(axis=1))
        assert_allclose(colmin.values, vals.min(axis=0))
        assert_allclose(top.values, vals.max())
        w = argmax.values.round()
        assert (w.sum(axis=1) == 1).all()
        assert (vals[w == 1] == vals.max(axis=1)).all()
        w = argmin.values.round()
        assert (w.sum(axis=0) == 1).all()
        assert (vals.T[w.T == 1] == vals.min(axis=0)).all()
