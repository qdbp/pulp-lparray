from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
from numpy.lib.array_utils import normalize_axis_tuple
from pulp import (
    LpAffineExpression,
    LpBinary,
    LpConstraint,
    LpProblem,
    LpVariable,
    lpSum,
    value,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from numpy.typing import ArrayLike, NDArray

type Category = Literal["Continuous", "Integer", "Binary"]
type Axis = int | tuple[int, ...] | None


class lparray(np.ndarray[tuple[int, ...], np.dtype[np.object_]]):
    """
    A numpy array holding homogeneous LpVariables, LpAffineExpression or
    LpConstraints.

    All variables in the array will have the same:
        - Lp* type
        - (intrinsic) upper bound
        - (intrinsic) lower bound
    and all vectorized operations will preserve this invariant. External
    manipulations that break this invariant will lead to incorrect behavior.

    All variables in an array are named, and share the same base name, which is
    extended by indices into a collection of index sets whose product spans the
    elements of the array. These index sets can be named or anonymous --
    anonymous index sets are just int ranges.

    Implements vectorized versions of various LP operations, specifically:

    All of:
        lparray {<=, ==, >=} {lparray, LpVariable}
    Will return an lparray of constraints: LpAffineExpressions with a sense.

    All of:
        lparray {+, -, *, /} {Real, ndarray[Real]}
        lparray @ ndarray
        lparray {+, -} {lparray, LpVariable}
    Will return an lparray of LpAffineExpression with the expected semantics.

    Constraint lparrays support the `constrain` method, which adds the
    constraints to an LpProblem and returns the LpConstraints it stores.

    In addition, an number of more sophisticated mathematical operations are
    supported, many of which involve the creation of lparrays of auxiliary
    variables behind the scenes.
    """

    __hash__ = None

    # PuLP 4 scalars set __array_priority__ = 20, so a lower priority makes ndarray operators
    # defer to the scalar's reflected method, which sums the whole array into one expression.
    __array_priority__ = math.inf

    if TYPE_CHECKING:
        # NumPy's stubs type these as plain or boolean ndarrays; __array_ufunc__ returns lparrays.
        def __getitem__(self, key: object, /) -> Any: ...  # noqa: ANN401
        def __neg__(self) -> lparray: ...
        def __add__(self, other: object, /) -> lparray: ...
        def __radd__(self, other: ArrayLike, /) -> lparray: ...
        def __sub__(self, other: object, /) -> lparray: ...
        def __rsub__(self, other: ArrayLike, /) -> lparray: ...
        def __mul__(self, other: object, /) -> lparray: ...
        def __rmul__(self, other: ArrayLike, /) -> lparray: ...
        def __truediv__(self, other: object, /) -> lparray: ...
        def __matmul__(self, other: object, /) -> lparray: ...
        def __rmatmul__(self, other: ArrayLike, /) -> lparray: ...
        # Deliberately incompatible: comparisons build constraints, and the rest keep the subclass.
        def __le__(self, other: object, /) -> lparray: ...  # ty: ignore[invalid-method-override]
        def __ge__(self, other: object, /) -> lparray: ...  # ty: ignore[invalid-method-override]
        def __eq__(self, other: object, /) -> lparray: ...  # ty: ignore[invalid-method-override]
        def sum(self, axis: Axis = None, *, keepdims: bool = False) -> lparray: ...  # ty: ignore[invalid-method-override]
        def reshape(self, *shape: Any, **kwargs: Any) -> lparray: ...  # ty: ignore[invalid-method-override]
        def ravel(self, order: Literal["K", "A", "C", "F"] = "C") -> lparray: ...  # ty: ignore[invalid-method-override]
        def flatten(self, order: Literal["K", "A", "C", "F"] = "C") -> lparray: ...  # ty: ignore[invalid-method-override]
        def squeeze(self, axis: Axis = None) -> lparray: ...  # ty: ignore[invalid-method-override]

    @classmethod
    def create(
        cls,
        prob: LpProblem,
        name: str,
        shape: int | tuple[int, ...],
        *,
        lowBound: float | None = None,
        upBound: float | None = None,
        cat: Category = "Continuous",
    ) -> Self:
        """Variables of `prob` named `name(i,j,…)` by their indices."""
        arr = np.empty(shape, dtype=object)
        for ix in np.ndindex(arr.shape):
            arr[ix] = prob.add_variable(_subscript(name, ix), lowBound, upBound, cat)
        return arr.view(cls)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any) -> object:
        args = [x.view(np.ndarray) if isinstance(x, lparray) else _shield(x) for x in inputs]
        if out := kwargs.get("out"):
            kwargs["out"] = tuple(o.view(np.ndarray) for o in out)
        match ufunc, method:
            case np.add, "reduce":
                result = _lpsum(*args, **kwargs)
            case np.matmul, "__call__":
                result = _matmul(*args, **kwargs)
            case np.less_equal | np.greater_equal | np.equal, "__call__":
                # PuLP 4 compares two variables by identity, yielding a bool. A difference is
                # always an expression, and comparing an expression with 0 is a constraint.
                difference = _shield(np.subtract(*args, dtype=object))
                result = ufunc(difference, 0, **kwargs | {"dtype": object})
            case np.add | np.subtract | np.multiply | np.true_divide | np.negative | np.positive, _:
                result = getattr(ufunc, method)(*args, **kwargs)
            case _:
                raise TypeError(
                    f"lparray supports affine ufuncs, <=, >= and ==, not np.{ufunc.__name__}."
                    f"{method}; lp_max, lp_min and abs linearise extrema and magnitudes"
                )
        if out:
            return out[0] if len(out) == 1 else out
        return _wrap(result)

    def __array_function__(
        self,
        func: Callable[..., Any],
        types: Iterable[type],
        args: Iterable[Any],
        kwargs: Mapping[str, Any],
    ) -> object:
        args = tuple(map(_shield, args))
        kwargs = {key: _shield(arg) for key, arg in kwargs.items()}
        if func is np.dot:
            return _wrap(_dot(*args, **kwargs))
        return _wrap(super().__array_function__(func, types, args, kwargs))

    def dot(self, b: ArrayLike, /) -> lparray:  # ty: ignore[invalid-method-override]
        """`np.dot(self, b)`, which ndarray.dot would compute without dispatching to lparray."""
        return np.dot(self, b)

    if not TYPE_CHECKING:  # NumPy's stubs type assignment as it is

        def __setitem__(self, key: object, value: object, /) -> None:
            super().__setitem__(key, _shield(value))

    @property
    def values(self) -> NDArray[np.float64]:
        """The solved value of each element."""
        return np.fromiter(map(value, self.flat), np.float64, self.size).reshape(self.shape)

    def constrain(self, prob: LpProblem, name: str) -> lparray:
        """Add each constraint to `prob` as `name(i,j,…)`; return the stored constraints."""
        stored = np.empty(self.shape, dtype=object)
        for ix in np.ndindex(self.shape):
            if not isinstance(c := self[ix], LpAffineExpression):
                raise TypeError(f"{_subscript(name, ix)}: {c!r} is not a constraint")
            stored[ix] = prob.addConstraint(c, _subscript(name, ix))
        return stored.view(lparray)

    def abs_decompose(self, prob: LpProblem, name: str, *, bigM: float) -> tuple[lparray, lparray]:
        """The positive and negative parts `(p, m)` of self: `p - m == self`, `p, m >= 0`,
        and at most one of them nonzero. Requires `|self| <= bigM`."""
        neg = lparray.create(prob, f"{name}_neg", self.shape, cat=LpBinary)
        p = lparray.create(prob, f"{name}_p", self.shape, lowBound=0)
        m = lparray.create(prob, f"{name}_m", self.shape, lowBound=0)
        (p - m == self).constrain(prob, f"{name}_split")
        (p <= bigM * (1 - neg)).constrain(prob, f"{name}_p_off")
        (m <= bigM * neg).constrain(prob, f"{name}_m_off")
        return p, m

    def abs(self, prob: LpProblem, name: str, *, bigM: float) -> lparray:
        """`|self|`; requires `|self| <= bigM`."""
        p, m = self.abs_decompose(prob, name, bigM=bigM)
        return p + m

    def logical_clip(self, prob: LpProblem, name: str, *, bigM: float) -> lparray:
        """Binary `min(self, 1)` of integral `0 <= self <= bigM`."""
        z = lparray.create(prob, f"{name}_z", self.shape, cat=LpBinary)
        (self >= z).constrain(prob, f"{name}_lb")
        (self <= bigM * z).constrain(prob, f"{name}_ub")
        return z

    def lp_max(self, prob: LpProblem, name: str, *, axis: Axis = None, bigM: float) -> lparray:
        """Maximum along `axis`; requires the spread of self along it to be at most bigM."""
        return self._extremum(prob, name, axis, bigM, 1)[0]

    def lp_min(self, prob: LpProblem, name: str, *, axis: Axis = None, bigM: float) -> lparray:
        """Minimum along `axis`; requires the spread of self along it to be at most bigM."""
        return self._extremum(prob, name, axis, bigM, -1)[0]

    def lp_argmax(self, prob: LpProblem, name: str, *, axis: Axis = None, bigM: float) -> lparray:
        """Binary indicator, one-hot along `axis`, of an entry attaining the maximum."""
        return self._extremum(prob, name, axis, bigM, 1)[1]

    def lp_argmin(self, prob: LpProblem, name: str, *, axis: Axis = None, bigM: float) -> lparray:
        """Binary indicator, one-hot along `axis`, of an entry attaining the minimum."""
        return self._extremum(prob, name, axis, bigM, -1)[1]

    def lp_bin_max(self, prob: LpProblem, name: str, *, axis: Axis = None) -> lparray:
        """Logical OR of binary self along `axis`."""
        axes, kept = self._reduction(axis)
        z = lparray.create(prob, f"{name}_z", kept, lowBound=0, upBound=1)
        _all(prob, name, 1 - z, 1 - self, axes)
        return z

    def lp_bin_min(self, prob: LpProblem, name: str, *, axis: Axis = None) -> lparray:
        """Logical AND of binary self along `axis`."""
        axes, kept = self._reduction(axis)
        z = lparray.create(prob, f"{name}_z", kept, lowBound=0, upBound=1)
        _all(prob, name, z, self, axes)
        return z

    def lp_bin_and(self, prob: LpProblem, name: str, *ins: ArrayLike) -> Self:
        """Constrain binary self to the elementwise logical AND of `ins`; return self."""
        _all(prob, name, self, _stacked(ins, self.shape), (0,))
        return self

    def lp_bin_or(self, prob: LpProblem, name: str, *ins: ArrayLike) -> Self:
        """Constrain binary self to the elementwise logical OR of `ins`; return self."""
        _all(prob, name, 1 - self, 1 - _stacked(ins, self.shape), (0,))
        return self

    def lp_topk(
        self, prob: LpProblem, name: str, *, k: int, axis: Axis = None, bigM: float
    ) -> lparray:
        """Binary indicator of `k` entries along `axis`, none smaller than an unselected one."""
        axes, kept = self._reduction(axis)
        if not 0 <= k <= math.prod(self.shape[d] for d in axes):
            raise ValueError(f"{name}: cannot select {k} of {self.shape} along {axes}")
        threshold = lparray.create(prob, f"{name}_t", kept)
        w = lparray.create(prob, f"{name}_w", self.shape, cat=LpBinary)
        (w.sum(axis=axes) == k).constrain(prob, f"{name}_k")
        gap = self - _expand(threshold, axes)
        (gap >= -bigM * (1 - w)).constrain(prob, f"{name}_in")
        (gap <= bigM * w).constrain(prob, f"{name}_out")
        return w

    def lp_interp(self, prob: LpProblem, name: str, *, xp: ArrayLike, fp: ArrayLike) -> lparray:
        """`np.interp(self, xp, fp)`, constraining self to `[xp[..., 0], xp[..., -1]]`.

        The breakpoints `xp` and values `fp` lie along their last axis; their leading axes
        broadcast against self."""
        xs, fs = np.broadcast_arrays(np.asarray(xp, dtype=float), np.asarray(fp, dtype=float))
        if xs.ndim == 0 or xs.shape[-1] < 2 or np.any(np.diff(xs) <= 0):
            raise ValueError("xp must increase strictly along its last axis, of length >= 2")
        shape = np.broadcast_shapes(self.shape, xs.shape[:-1])
        weight = lparray.create(prob, f"{name}_weight", (*shape, xs.shape[-1]), lowBound=0)
        segment = lparray.create(prob, f"{name}_segment", (*shape, xs.shape[-1] - 1), cat=LpBinary)
        (weight.sum(axis=-1) == 1).constrain(prob, f"{name}_weight_sum")
        (segment.sum(axis=-1) == 1).constrain(prob, f"{name}_segment_sum")
        # A weight may be positive only beside the chosen segment: SOS2 adjacency.
        edge = np.zeros((*shape, 1), dtype=object)
        flanks = np.concatenate([edge, segment, edge], axis=-1)
        (weight <= flanks[..., :-1] + flanks[..., 1:]).constrain(prob, f"{name}_adjacent")
        (self == (weight * xs).sum(axis=-1)).constrain(prob, f"{name}_x")
        return (weight * fs).sum(axis=-1)

    def _extremum(
        self, prob: LpProblem, name: str, axis: Axis, bigM: float, sign: Literal[1, -1]
    ) -> tuple[lparray, lparray]:
        """`(t, w)`: `sign·t = max(sign·self)` along `axis`, `w` one-hot where attained."""
        axes, kept = self._reduction(axis)
        if not math.prod(self.shape[d] for d in axes):
            raise ValueError(f"no extremum of an empty slice of {self.shape} along {axes}")
        t = lparray.create(prob, f"{name}_t", kept)
        w = lparray.create(prob, f"{name}_w", self.shape, cat=LpBinary)
        (w.sum(axis=axes) == 1).constrain(prob, f"{name}_one")
        gap = sign * (_expand(t, axes) - self)
        (gap >= 0).constrain(prob, f"{name}_bound")
        (gap <= bigM * (1 - w)).constrain(prob, f"{name}_attain")
        return t, w

    def _reduction(self, axis: Axis) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """The axes reduced over and the shape left."""
        axes = normalize_axis_tuple(range(self.ndim) if axis is None else axis, self.ndim)
        return axes, tuple(n for d, n in enumerate(self.shape) if d not in axes)


def _subscript(name: str, ix: tuple[int, ...]) -> str:
    return f"{name}({','.join(map(str, ix))})" if ix else name


def _box(x: object) -> NDArray[np.object_]:
    box = np.empty((), dtype=object)
    box[()] = x
    return box


def _shield[T](x: T) -> T | NDArray[np.object_]:
    """Box a PuLP 4 expression or constraint in a 0-d array, and those within lists and tuples:
    both define __len__ and __getitem__, so NumPy would read one as the sequence of its
    variables."""
    match x:
        case LpAffineExpression() | LpConstraint():
            return _box(x)
        case list() | tuple() if type(x) in (list, tuple):
            return type(x)(map(_shield, x))
        case _:
            return x


def _wrap(x: object) -> object:
    """View the object ndarrays among a NumPy result as lparrays, and box the PuLP scalars that
    NumPy returns for 0-d results into 0-d lparrays."""
    match x:
        case np.ndarray(dtype=np.object_):
            return x.view(lparray)
        case LpAffineExpression() | LpVariable() | LpConstraint():
            return _box(x).view(lparray)
        case tuple():
            return tuple(map(_wrap, x))
        case list():
            return list(map(_wrap, x))
        case _:
            return x


def _lpsum(
    a: NDArray[np.object_],
    *,
    axis: Axis = 0,
    dtype: object = None,
    keepdims: bool = False,
    where: object = True,
) -> NDArray[np.object_]:
    """`np.add.reduce` through `lpSum`. NumPy's object reduction chains `+`, and each PuLP `+`
    copies its left operand, so it takes time quadratic in the length reduced."""
    if dtype not in (None, object) or where is not True:
        raise TypeError("lparray sums take only axis and keepdims")
    if a.ndim == 0 and axis in (0, -1):
        axis = ()  # NumPy reduces a 0-d array along axis 0 to itself
    axes = normalize_axis_tuple(range(a.ndim) if axis is None else axis, a.ndim)
    kept = [d for d in range(a.ndim) if d not in axes]
    rows = a.transpose(*kept, *axes).reshape(
        *(a.shape[d] for d in kept), math.prod(a.shape[d] for d in axes)
    )
    out = np.empty(rows.shape[:-1], dtype=object)
    for ix in np.ndindex(out.shape):
        out[ix] = lpSum(rows[ix])
    return np.expand_dims(out, axes) if keepdims else out


def _matmul(a: ArrayLike, b: ArrayLike) -> NDArray[np.object_]:
    """`a @ b` as a sum of broadcast products, in time linear in the contracted length."""
    lhs, rhs = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    if not lhs.ndim or not rhs.ndim or lhs.shape[-1] != rhs.shape[-2 if rhs.ndim > 1 else 0]:
        raise ValueError(f"matmul cannot contract {lhs.shape} with {rhs.shape}")
    products = (lhs[None] if lhs.ndim == 1 else lhs)[..., :, :, None] * (
        rhs[:, None] if rhs.ndim == 1 else rhs
    )[..., None, :, :]
    promoted = (-2,) * (lhs.ndim == 1) + (-1,) * (rhs.ndim == 1)
    return _lpsum(products, axis=-2).squeeze(axis=promoted)


def _dot(a: ArrayLike, b: ArrayLike) -> object:
    """`np.dot(a, b)` through `_matmul`, which agrees with it when b has at most two axes."""
    lhs, rhs = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    if not lhs.ndim or not rhs.ndim:
        return np.multiply(lhs, rhs)
    if rhs.ndim <= 2:
        return _matmul(lhs, rhs)
    # dot contracts the last axis of a with the second-to-last of b; b's other axes follow a's.
    flat = np.moveaxis(rhs, -2, 0).reshape(rhs.shape[-2], -1)
    return _matmul(lhs, flat).reshape(*lhs.shape[:-1], *rhs.shape[:-2], rhs.shape[-1])


def _expand(t: lparray, axes: tuple[int, ...]) -> lparray:
    """A reduction's result with unit axes restored, to broadcast against its operand."""
    return np.expand_dims(t, axes).view(lparray)


def _stacked(ins: tuple[ArrayLike, ...], shape: tuple[int, ...]) -> NDArray[Any]:
    """`ins` broadcast to `shape` and stacked along a new first axis."""
    return np.stack([np.broadcast_to(_shield(x), shape, subok=True) for x in ins])


def _all(prob: LpProblem, name: str, z: lparray, x: NDArray[Any], axes: tuple[int, ...]) -> None:
    """Constrain `0 <= z <= 1` to the logical AND of binary `x` along `axes`."""
    n = math.prod(x.shape[d] for d in axes)
    (_expand(z, axes) <= x).constrain(prob, f"{name}_each")
    (z >= x.sum(axis=axes) - (n - 1)).constrain(prob, f"{name}_sum")
