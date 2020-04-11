from __future__ import annotations

from typing import (
    Union,
    Iterable,
    List,
    NoReturn,
    Tuple,
    Any,
    Collection,
    Optional,
)

import numpy as np
from pulp import (
    LpProblem,
    LpVariable,
    LpInteger,
    LpBinary,
    LpContinuous,
    value,
    LpConstraint,
)
from typing_extensions import Literal

LpComparable = Union["lparray", LpVariable, int, float]
LpVarType = Literal["Binary", "Integer", "Continuous"]
Number = Union[int, float]


def number(it: Iterable[Any]) -> List[int]:
    return [ix for ix, _ in enumerate(it)]


# noinspection PyPep8Naming
class lparray(np.ndarray):
    @staticmethod
    def bin_and(prob: LpProblem, name: str, out: lparray, *ins: lparray) -> None:
        for ix, _in in enumerate(ins):
            (out <= _in).constrain(prob, f"{name}_and_ub{ix}")
        (out >= sum(ins, 1 - len(ins))).constrain(prob, f"{name}_and_lb")

    @staticmethod
    def bin_or(prob: LpProblem, name: str, out: lparray, *ins: lparray) -> None:
        for ix, _in in enumerate(ins):
            (out >= _in).constrain(prob, f"{name}_or_lb{ix}")
        (out <= sum(ins)).constrain(prob, f"{name}_and_ub")

    @classmethod
    def create(
        cls,
        name: str,
        index_sets: Tuple[Collection[Any], ...],
        *,
        lowBound: Optional[Number] = None,
        upBound: Optional[Number] = None,
        cat: LpVarType = "Continuous",
    ) -> lparray:

        """
        Numpy array equivalent of pulp.LpVariable.dicts
        """

        if len(index_sets) == 0:
            return (  # type: ignore
                np.array(
                    [LpVariable(name, cat=cat, upBound=upBound, lowBound=lowBound)]
                )
                .squeeze()
                .view(lparray)
            )

        if len(index_sets) == 1:
            name = name + "("

        def recursive_worker(
            r_name: str, plane: np.ndarray, r_index_sets: Tuple[Collection[Any], ...]
        ) -> None:

            if len(r_index_sets) == 1:
                close_paren = r_name and (")" if "(" in r_name else "")
                for ix in number(r_index_sets[0]):
                    plane[ix] = LpVariable(
                        f"{r_name}{ix}{close_paren}",
                        cat=cat,
                        upBound=upBound,
                        lowBound=lowBound,
                    )
            else:
                open_paren = r_name and ("(" if "(" not in r_name else "")
                for ix in number(r_index_sets[0]):
                    recursive_worker(
                        f"{r_name}{open_paren}{ix},", plane[ix], r_index_sets[1:]
                    )

        arr = np.zeros(tuple(len(ixset) for ixset in index_sets), dtype=np.object)
        recursive_worker(name, arr, index_sets)

        return arr.view(lparray)  # type: ignore

    @classmethod
    def create_like(cls, name: str, like: lparray, **kwargs: Any) -> lparray:
        return cls.create_anon(name, like.shape, **kwargs)

    @classmethod
    def create_anon(cls, name: str, shape: Tuple[int, ...], **kwargs: Any) -> lparray:
        index_sets = tuple(list(range(d)) for d in shape)
        return cls.create(name, index_sets, **kwargs)

    def __ge__(self, other: LpComparable) -> lparray:
        return np.greater_equal(self, other, dtype=object)  # type: ignore

    def __le__(self, other: LpComparable) -> lparray:
        return np.less_equal(self, other, dtype=object)  # type: ignore

    def __lt__(self, other: LpComparable) -> NoReturn:
        raise NotImplementedError("lparrays support only <=, >=, and ==")

    def __gt__(self, other: LpComparable) -> NoReturn:
        raise NotImplementedError("lparrays support only <=, >=, and ==")

    def __eq__(self, other: LpComparable) -> lparray:
        return np.equal(self, other, dtype=object)  # type: ignore

    @property
    def values(self) -> np.ndarray:
        """
        Returns the underlying values of the PuLP variables by calling
        `pulp.value` on each element of self.

        If the problem has not been solved, all entries will be None.

        Returns:
            ndarray of the same shape as self.
        """

        return np.vectorize(lambda x: value(x))(self).view(np.ndarray)

    def constrain(self, prob: LpProblem, name: str) -> None:
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
            name = name + "("

        def recursive_worker(r_prob: LpProblem, plane: np.ndarray, r_name: str) -> None:
            if plane.ndim == 1:
                close_paren = r_name and (")" if "(" in r_name else "")
                for cx, const in enumerate(plane):
                    if not isinstance(const, LpConstraint):
                        raise TypeError(
                            "Attempting to constrain problem with "
                            f"non-constraint {const}"
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
        self, prob: LpProblem, name: str, bigM: Number = 1000.0, **kwargs: Any
    ) -> Tuple[lparray, lparray]:
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
            args: extra arguments to `create`
            bigM: the -lower and upper bound on self to assume.
        """

        # w == 1 <=> self <= 0
        w = lparray.create_like(
            f"{name}_abs_aux", self, lowBound=0, upBound=1, cat=LpBinary
        )
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

    def abs(self, **kwargs: Any) -> lparray:
        """
        Returns variable equal to |self|.

        Thin wrapper around `abs_decompose`
        """
        xp, xm = self.abs_decompose(**kwargs)
        return xp + xm  # type: ignore

    def logical_clip(self, prob: LpProblem, name: str, bigM: Number = 1000) -> lparray:
        """
        Assumes self is integer >= 0.

        Returns an array of the same shape as self containing
            z_... = max(self_..., 1)

        Generates self.size new variables.
        """

        z = self.__class__.create(
            name, tuple(range(x) for x in self.shape), cat=LpBinary
        )

        (self >= z).constrain(prob, f"{name}_lb")
        (self <= bigM * z).constrain(prob, f"{name}_ub")

        return z

    def sumit(self, *args: Any, **kwargs: Any) -> LpVariable:
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
        lb: Optional[Number] = None,
        ub: Optional[Number] = None,
        bigM: Number = 1000,
        axis: Union[None, int, Tuple[int, ...]] = None,
    ) -> lparray:

        if not np.product(self.shape):
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
            assert 0, "Need to supply constraints for non-binary variables!"

        assert which in ("min", "max")

        aux_name = f"{name}_{which}_aux"

        # axes of self which the max is indexed by
        keep_axis = tuple(ix for ix in range(self.ndim) if ix not in axis)

        # array of maxes
        target_shape = sum((self.shape[ax : ax + 1] for ax in keep_axis), ())
        target: lparray = lparray.create_anon(
            name, target_shape, lowBound=lb, upBound=ub, cat=cat
        )

        # broadcastable version for comparison with self
        br = tuple(
            (slice(None, None, None) if ax in keep_axis else None)
            for ax in range(self.ndim)
        )
        target_br: lparray = target[br]

        # indicator variable array. w[ixs ∈ span(axis), ~axis] == 1 <=> self[ixs, ~axis]
        # is binding in the subspace spanned by axis
        w = self.create_like(aux_name, self, lowBound=0, upBound=1, cat=LpBinary)
        (w.sum(axis=axis) == 1).constrain(prob, f"{name}_aux_sum")

        if which == "max":
            (target_br >= self).constrain(prob, f"{name}_lt_max")
            (target_br <= self + bigM * (1 - w)).constrain(prob, f"{name}_attains_max")
        elif which == "min":
            (target_br <= self).constrain(prob, f"{name}_gt_min")
            (target_br >= self - bigM * (1 - w)).constrain(prob, f"{name}_attains_min")
        else:
            assert 0

        return target

    def _lp_int_minmax(
        self,
        prob: LpProblem,
        name: str,
        which: Literal["min", "max"],
        lb: int,
        ub: int,
        **kwargs: Any,
    ) -> lparray:

        if lb == 0 and ub == 1:
            cat = LpBinary
        else:
            cat = LpInteger

        return self._lp_minmax(prob, name, which=which, cat=cat, lb=lb, ub=ub, **kwargs)

    def lp_int_max(
        self, prob: LpProblem, name: str, lb: int, ub: int, **kwargs: Any
    ) -> lparray:
        return self._lp_int_minmax(prob, name, which="max", lb=lb, ub=ub, **kwargs)

    def lp_int_min(
        self, prob: LpProblem, name: str, lb: int, ub: int, **kwargs: Any
    ) -> lparray:
        return self._lp_int_minmax(prob, name, which="min", lb=lb, ub=ub, **kwargs)

    def lp_bin_max(self, prob: LpProblem, name: str, **kwargs: Any) -> lparray:
        return self._lp_int_minmax(prob, name, lb=0, ub=1, which="max", **kwargs)

    def lp_bin_min(self, prob: LpProblem, name: str, **kwargs: Any) -> lparray:
        return self._lp_int_minmax(prob, name, lb=0, ub=1, which="min", **kwargs)

    def lp_real_max(self, prob: LpProblem, name: str, **kwargs: Any) -> lparray:
        return self._lp_minmax(prob, name, "max", LpContinuous, **kwargs)

    def lp_real_min(self, prob: LpProblem, name: str, **kwargs: Any) -> lparray:
        return self._lp_minmax(prob, name, "min", LpContinuous, **kwargs)
