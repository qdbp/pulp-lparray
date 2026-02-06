from __future__ import annotations

from collections.abc import Collection, Iterable
from typing import Any, Generic, Literal, NoReturn, Protocol, TypeVar, Union

import numpy as np
from numpy import ndarray
from pulp import (
    LpAffineExpression,
    LpBinary,
    LpConstraint,
    LpContinuous,
    LpInteger,
    LpProblem,
    LpVariable,
    value,
)

LpComparable = Union["lparray", LpVariable, int, float]
LpVarType = Literal["Binary", "Integer", "Continuous"]

Number = Union[int, float]
LP = TypeVar("LP", LpVariable, LpAffineExpression, LpComparable, LpConstraint)
LPV = TypeVar("LPV", LpVariable, LpAffineExpression)


class HasShape(Protocol):
    shape: tuple[int, ...]


Kwargs = dict[str, Any]


def count_out(it: Iterable[Any]) -> list[int]:
    return [ix for ix, _ in enumerate(it)]


# noinspection PyPep8Naming
class lparray(
    ndarray,
    Generic[LP],
):
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
    Will return an lparray of LpConstraints, with expected semantics.

    All of:
        lparray {+, -, *, /} {Real, ndarray[Real]}
        lparray @ ndarray
        lparray {+, -} {lparray, LpVariable}
    Will return an lparray of LpAffineExpression with the expected semantics.

    LpConstraint-type lparrays support the `constrain` method, which will
    bind the constraints to an LpProblem.

    In addition, an number of more sophisticated mathematical operations are
    supported, many of which involve the creation of lparrays of auxiliary
    variables behind the scenes.
    """

    @classmethod
    def create(
        cls,
        name: str,
        index_sets: tuple[Collection[Any], ...],
        *,
        lowBound: Number | None = None,
        upBound: Number | None = None,
        cat: LpVarType = "Continuous",
    ) -> lparray[LpVariable]:
        """
        Creates an lparray with shape from a cartesian product of index sets.

        Each LpVariable in the array at index [i_0, ..., i_n] will be named
        as "{name}_(ix_sets[0][i_0], ..., ix_sets[n][i_n])"

        Args:
            name: base name for the underlying LpVariables
            index_sets: an iterable of iterables containing the dimension names
                for the array.
            lowBound: passed to LpVariable, uniform for array
            upBound: passed as to LpVariable, uniform for array
            cat: passed as to LpVariable, uniform for array

        Return:
            an lparray with the specified shape, with variables named after
            their integer coordinates in the array.
        """

        if len(index_sets) == 0:
            return (
                np.array([LpVariable(name, cat=cat, upBound=upBound, lowBound=lowBound)])
                .squeeze()
                .view(lparray)
            )

        if len(index_sets) == 1:
            name += "("

        def recursive_worker(
            r_name: str,
            plane: np.ndarray,
            r_index_sets: tuple[Iterable[Any], ...],
        ) -> None:

            if len(r_index_sets) == 1:
                close_paren = r_name and (")" if "(" in r_name else "")
                for ix in count_out(r_index_sets[0]):
                    plane[ix] = LpVariable(
                        f"{r_name}{ix}{close_paren}",
                        cat=cat,
                        upBound=upBound,
                        lowBound=lowBound,
                    )
            else:
                open_paren = r_name and ("(" if "(" not in r_name else "")
                for ix in count_out(r_index_sets[0]):
                    recursive_worker(
                        f"{r_name}{open_paren}{ix},",
                        plane[ix],
                        r_index_sets[1:],
                    )

        arr = np.zeros(tuple(len(ixset) for ixset in index_sets), dtype=object)
        recursive_worker(name, arr, index_sets)

        return arr.view(lparray)  # ty: ignore[invalid-return-type]

    @classmethod
    def create_like(cls, name: str, like: HasShape, **kwargs: Kwargs) -> lparray[LpVariable]:
        """
        Creates an anonymous lparray with the same shape as a passed array.

        Args:
            name: base name for the LpVariables in the array
            like: an object supporting `.shape` whose shape will be used

        Returns:
            a new lparray[LpVariable]
        """
        return cls.create_anon(name, like.shape, **kwargs)

    @classmethod
    def create_anon(
        cls, name: str, shape: tuple[int, ...], **kwargs: Kwargs
    ) -> lparray[LpVariable]:
        """
        Creates an lparray with a given shape and nameless index sets.

        Args:
            name: base name for the underlying LpVariables
            shape: array shape, same as for numpy arrays

        Return:
            an lparray with the specified shape, with variables named after
            their integer coordinates in the array.
        """
        index_sets = tuple(list(range(d)) for d in shape)
        return cls.create(name, index_sets, **kwargs)

    def __ge__(self, other: LpComparable) -> lparray[LpConstraint]:
        return np.greater_equal(self, other, dtype=object)

    def __le__(self, other: LpComparable) -> lparray[LpConstraint]:
        return np.less_equal(self, other, dtype=object)

    def __lt__(self, other: LpComparable) -> NoReturn:
        raise NotImplementedError("lparrays support only <=, >=, and ==")

    def __gt__(self, other: LpComparable) -> NoReturn:
        raise NotImplementedError("lparrays support only <=, >=, and ==")

    def __eq__(self, other: LpComparable) -> lparray[LpConstraint]:
        return np.equal(self, other, dtype=object)

    @property
    def values(self: lparray[LPV]) -> np.ndarray:
        """
        Returns the underlying values of the PuLP variables by calling
        `pulp.value` on each element of self.

        If the problem has not been solved, all entries will be None.

        Returns:
            ndarray of the same shape as self.
        """

        return np.vectorize(value)(self).view(np.ndarray)

    def constrain(self: lparray[LpConstraint], prob: LpProblem, name: str) -> None:
        """
        Applies the constraints contained in self to the problem.

        Preconditions:
            all entries of self are `LpConstraints`.

        Arguments:
            prob: `LpProblem` which to apply constraints to.
            name: base name to use for the applied constraints.
        """
        if not isinstance(prob, LpProblem):
            raise TypeError(f"Trying to constrain a {type(prob)}. Did you pass prob?")
        if self.ndim == 0:
            cons = self.item()
            cons.name = name
            prob += cons
            return

        if name and self.ndim == 1:
            name += "("

        def recursive_worker(r_prob: LpProblem, plane: np.ndarray, r_name: str) -> None:
            if plane.ndim == 1:
                close_paren = r_name and (")" if "(" in r_name else "")
                for cx, const in enumerate(plane):
                    if not isinstance(const, LpConstraint):
                        raise TypeError(
                            f"Attempting to constrain problem with non-constraint {const}"
                        )
                    const.name = r_name and f"{r_name}{cx}{close_paren}"
                    r_prob += const
            else:
                open_paren = r_name and ("(" if "(" not in r_name else "")
                for px, subplane in enumerate(plane):
                    subname = r_name and f"{r_name}{open_paren}{px},"
                    recursive_worker(r_prob, subplane, subname)

        recursive_worker(prob, self, name)

    def abs_decompose(
        self: lparray[LPV],
        prob: LpProblem,
        name: str,
        *,
        bigM: Number = 1000.0,
        **kwargs: Kwargs,
    ) -> tuple[lparray[LpVariable], lparray[LpVariable]]:
        """
        Generates two arrays, xp and xm, that sum to |self|, with the following
        properties:

            xp >= 0
            xm >= 0
            xp == 0 XOR xm == 0

        Uses the big M method.
        Generates 2 * self.size visible new variables.
        Generates 1 * self.size binary auxiliary variables.

        Arguments:
            prob: LpProblem to bind aux variables to
            name: base name for generated variables
            bigM: the -lower and upper bound on self to assume.
            kwargs: extra arguments to `create`
        """

        # w == 1 <=> self <= 0
        w = lparray.create_like(f"{name}_abs_aux", self, lowBound=0, upBound=1, cat=LpBinary)
        # binding if self >= 0
        (self <= bigM * (1 - w)).constrain(prob, f"{name}_lb")
        # binding if self <= 0
        (self >= -bigM * w).constrain(prob, f"{name}_ub")

        # xp is the positive half of X, xm is the negative half of X
        xp = lparray.create_like(f"{name}_absp", self, **kwargs)
        xm = lparray.create_like(f"{name}_absm", self, **kwargs)

        (xp >= 0).constrain(prob, f"{name}_abs_xplb")
        (xm >= 0).constrain(prob, f"{name}_abs_xmlb")
        (xp - xm == self).constrain(prob, f"{name}_absdecomp")

        # xp >= 0 <=> xm == 0 and vice versa
        (xp <= bigM * (1 - w)).constrain(prob, f"{name}_absxpexcl")
        (xm <= bigM * w).constrain(prob, f"{name}_absxmexcl")

        return xp, xm

    def abs(self, prob: LpProblem, name: str, **kwargs: Kwargs) -> lparray[LpAffineExpression]:
        """
        Returns variable equal to |self|.

        Thin wrapper around `abs_decompose`
        """
        xp, xm = self.abs_decompose(prob, name, **kwargs)
        return xp + xm

    def abs_clip(
        self,
        prob: LpProblem,
        name: str,
        *,
        bigM: Number,
        lowBound: Number | None = None,
        upBound: Number | None = None,
        cat: LpVarType = "Continuous",
    ) -> lparray[LpAffineExpression]:
        xp, xm = self.abs_decompose(
            prob,
            name,
            bigM=bigM,
            lowBound=lowBound,
            upBound=upBound,
            cat=cat,
        )
        return xp + xm

    def logical_clip(self, prob: LpProblem, name: str, bigM: Number = 1000) -> lparray[LpVariable]:
        """
        Assumes self is integer >= 0.

        Returns an array of the same shape as self containing
            z_... = max(self_..., 1)

        Generates self.size new variables.
        """

        z = self.__class__.create(name, tuple(range(x) for x in self.shape), cat=LpBinary)

        (self >= z).constrain(prob, f"{name}_lb")
        (self <= bigM * z).constrain(prob, f"{name}_ub")

        return z

    def sumit(self, *args: object, **kwargs: Kwargs) -> LpVariable:
        """
        Equivalent to `self.sum().item()`
        """
        out = self.sum(*args, **kwargs)
        return out.item()

    def _lp_minmax(
        self,
        prob: LpProblem,
        name: str,
        which: Literal["min", "max"],
        cat: LpVarType,
        *,
        lb: Number | None = None,
        ub: Number | None = None,
        bigM: Number = 1000,
        axis: int | tuple[int, ...] | None = None,
    ) -> lparray[LpVariable]:
        r"""
        Returns an lparray the min/max of the given lparray along an axis.

        Axis can be multi-dimensional.

        Args:
            prob: the problem instance to which to apply the constraints.
            name: base LpVariable name for the min/max output array.
            which: "min" or "max" -- determines the operation
            cat: LpCategory of the output lparray
            lb: lower bound on the output array
            ub: upper bound on the output array
            bigM: the big M value used for auxiliary variable inequalities.
                Should be larger than any value that can appear in self in
                a feasible solution.
            axis: the axes along which to take the maximum

        Returns:
            lparray, indexed by self.shape \ axis
        """

        if not np.prod(self.shape):
            raise ValueError("No variables given!")

        if axis is None:
            axis = tuple(range(self.ndim))
        elif isinstance(axis, int):
            axis = (axis,)
        elif (
            not isinstance(axis, tuple)
            or not axis
            or any(not isinstance(ax, int) or not self.ndim > ax >= 0 for ax in axis)
        ):
            raise TypeError("Axis must be a tuple of positive integers")

        if cat == LpBinary:
            lb = 0
            ub = 1
        elif lb is None or ub is None:
            raise ValueError("Need to supply bounds for non-binary variables")

        assert which in ("min", "max")

        aux_name = f"{name}_{which}_aux"

        # axes of self which the max is indexed by
        keep_axis = tuple(ix for ix in range(self.ndim) if ix not in axis)

        # array of maxes
        target_shape = sum((self.shape[ax : ax + 1] for ax in keep_axis), ())
        target: lparray[LpVariable] = lparray.create_anon(
            name, target_shape, lowBound=lb, upBound=ub, cat=cat
        )

        # broadcastable version for comparison with self
        br = tuple(
            (slice(None, None, None) if ax in keep_axis else None) for ax in range(self.ndim)
        )
        target_br: lparray[LpVariable] = target[br]

        # indicator variable array.
        # w[ixs ∈ span(axis), ~axis] == 1 <=> self[ixs, ~axis] is binding
        w = self.create_like(aux_name, self, lowBound=0, upBound=1, cat=LpBinary)
        (w.sum(axis=axis) == 1).constrain(prob, f"{name}_aux_sum")

        if which == "max":
            (target_br >= self).constrain(prob, f"{name}_lt_max")
            (target_br <= self + bigM * (1 - w)).constrain(prob, f"{name}_attains_max")
        elif which == "min":
            (target_br <= self).constrain(prob, f"{name}_gt_min")
            (target_br >= self - bigM * (1 - w)).constrain(prob, f"{name}_attains_min")
        else:
            raise ValueError("which must be 'min' or 'max'")

        return target

    def _lp_int_minmax(
        self,
        prob: LpProblem,
        name: str,
        which: Literal["min", "max"],
        lb: int,
        ub: int,
        **kwargs: Kwargs,
    ) -> lparray[LpVariable]:

        cat = LpBinary if lb == 0 and ub == 1 else LpInteger

        return self._lp_minmax(prob, name, which=which, cat=cat, lb=lb, ub=ub, **kwargs)

    def lp_int_max(
        self, prob: LpProblem, name: str, lb: int, ub: int, **kwargs: Kwargs
    ) -> lparray[LpVariable]:
        """
        Returns an array corresponding to the maximum value of self along axes.

        Integer variable type.
        """
        return self._lp_int_minmax(prob, name, which="max", lb=lb, ub=ub, **kwargs)

    def lp_int_min(
        self, prob: LpProblem, name: str, lb: int, ub: int, **kwargs: Kwargs
    ) -> lparray[LpVariable]:
        """
        Returns an array corresponding to the maximum value of self along axes.

        Integer variable type.

        See Also:
            `_lp_minmax` for arguments.
        """
        return self._lp_int_minmax(prob, name, which="min", lb=lb, ub=ub, **kwargs)

    def lp_bin_max(self, prob: LpProblem, name: str, **kwargs: Kwargs) -> lparray[LpVariable]:
        """
        Returns an array corresponding to the maximum value of self along axes.

        Binary variable type.

        See Also:
            `_lp_minmax` for arguments.
        """
        return self._lp_int_minmax(prob, name, lb=0, ub=1, which="max", **kwargs)

    def lp_bin_min(self, prob: LpProblem, name: str, **kwargs: Kwargs) -> lparray[LpVariable]:
        """
        Returns an array corresponding to the minimum value of self along axes.

        Binary variable type.

        See Also:
            `_lp_minmax` for arguments.
        """
        return self._lp_int_minmax(prob, name, lb=0, ub=1, which="min", **kwargs)

    def lp_real_max(self, prob: LpProblem, name: str, **kwargs: Kwargs) -> lparray[LpVariable]:
        """
        Returns an array corresponding to the maximum value of self along axes.

        Continuous variable type.

        See Also:
            `_lp_minmax` for arguments.
        """
        return self._lp_minmax(prob, name, "max", LpContinuous, **kwargs)

    def lp_real_min(self, prob: LpProblem, name: str, **kwargs: Kwargs) -> lparray[LpVariable]:
        """
        Returns an array corresponding to the minimum value of self along axes.

        Continuous variable type.

        See Also:
            `_lp_minmax` for arguments.
        """
        return self._lp_minmax(prob, name, "min", LpContinuous, **kwargs)

    def lp_bin_and(
        self: lparray[LPV],
        prob: LpProblem,
        name: str,
        *ins: lparray[LpVariable] | lparray[LpAffineExpression] | ndarray,
    ) -> lparray[LPV]:
        """
        Constrains the array to be the logical AND of a number of binary
        inputs.

        Returns:
            self
        """
        for ix, _in in enumerate(ins):
            (self <= _in).constrain(prob, f"{name}_and_ub{ix}")
        # empty and = 1
        (self >= sum(ins, 1 - len(ins))).constrain(prob, f"{name}_and_lb")
        return self

    def lp_bin_or(
        self: lparray[LPV],
        prob: LpProblem,
        name: str,
        *ins: lparray[LpVariable] | lparray[LpAffineExpression] | ndarray,
    ) -> lparray[LPV]:
        """
        Constrains the array to be the logical OR of a number of binary inputs.

        Returns:
            self
        """
        for ix, _in in enumerate(ins):
            (self >= _in).constrain(prob, f"{name}_or_lb{ix}")
        # empty or = 0
        (self <= sum(ins)).constrain(prob, f"{name}_and_ub")
        return self
