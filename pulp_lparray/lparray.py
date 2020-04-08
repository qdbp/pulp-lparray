from __future__ import annotations

from typing import Union, Iterable, List, NoReturn, Tuple

import numpy as np
import pulp as pp
from pulp import LpProblem, LpVariable
from typing_extensions import Literal


def number(it: Iterable) -> List[int]:
    return list(ix for ix, _ in enumerate(it))


LpComparable = Union["lparray", LpVariable, int, float]


# noinspection PyPep8Naming
class lparray(np.ndarray):
    @staticmethod
    def bin_and(prob: pp.LpProblem, name: str, out: lparray, *ins: lparray):
        for ix, _in in enumerate(ins):
            (out <= _in).constrain(prob, f"{name}_and_ub{ix}")
        (out >= sum(ins, 1 - len(ins))).constrain(prob, f"{name}_and_lb")

    @staticmethod
    def bin_or(prob: pp.LpProblem, name: str, out: lparray, *ins: lparray):
        for ix, _in in enumerate(ins):
            (out >= _in).constrain(prob, f"{name}_or_lb{ix}")
        (out <= sum(ins)).constrain(prob, f"{name}_and_ub")

    @classmethod
    def create(cls, name: str, index_sets, *args, **kwargs) -> lparray:
        """
        Numpy array equivalent of pulp.LpVariable.dicts
        """

        if len(index_sets) == 0:
            return (  # type: ignore
                np.array([pp.LpVariable(name, *args, **kwargs)]).squeeze().view(lparray)
            )

        if len(index_sets) == 1:
            name = name + "("

        def _rworker(name: str, plane: np.ndarray, index_sets):
            if len(index_sets) == 1:
                close_paren = name and (")" if "(" in name else "")
                for ix in number(index_sets[0]):
                    plane[ix] = pp.LpVariable(
                        f"{name}{ix}{close_paren}", *args, **kwargs
                    )
            else:
                open_paren = name and ("(" if "(" not in name else "")
                for ix in number(index_sets[0]):
                    _rworker(f"{name}{open_paren}{ix},", plane[ix], index_sets[1:])

        arr = np.zeros(tuple(len(ixset) for ixset in index_sets), dtype=np.object)
        _rworker(name, arr, index_sets)

        return arr.view(lparray)  # type: ignore

    @classmethod
    def create_like(cls, name: str, like: lparray, *args, **kwargs) -> lparray:
        return cls.create_anon(name, like.shape, *args, **kwargs)

    @classmethod
    def create_anon(cls, name: str, shape: Tuple[int, ...], *args, **kwargs) -> lparray:
        ixsets = tuple(list(range(d)) for d in shape)
        return cls.create(name, ixsets, *args, **kwargs)

    def __ge__(self, other: LpComparable) -> lparray:
        return np.greater_equal(self, other, dtype=object)

    def __le__(self, other: LpComparable) -> lparray:
        return np.less_equal(self, other, dtype=object)

    def __lt__(self, other: LpComparable) -> NoReturn:
        raise NotImplementedError("lparrays support only <=, >=, and ==")

    def __gt__(self, other: LpComparable) -> NoReturn:
        raise NotImplementedError("lparrays support only <=, >=, and ==")

    def __eq__(self, other: LpComparable) -> lparray:
        return np.equal(self, other, dtype=object)

    @property
    def values(self) -> np.ndarray:
        """
        Returns the underlying values of the PuLP variables by calling
        `pulp.value` on each element of self.

        If the problem has not been solved, all entries will be None.

        Returns:
            ndarray of the same shape as self.
        """

        return np.vectorize(lambda x: pp.value(x))(self).view(np.ndarray)

    def constrain(self, prob: pp.LpProblem, name: str) -> None:
        """
        Applies the constraints contained in self to the problem.

        Preconditions:
            all entries of self are `LpConstraints`.

        Arguments:
            prob: `LpProblem` which to apply constraints to.
            name: base name to use for the applied constraints.
        """
        if not isinstance(prob, pp.LpProblem):
            raise TypeError(f"Trying to constrain a {type(prob)}. Did you pass prob?")
        if self.ndim == 0:
            cons = self.item()
            cons.name = name
            prob += cons
            return

        if name and self.ndim == 1:
            name = name + "("

        def _rworker(prob: LpProblem, plane, name: str) -> None:
            if plane.ndim == 1:
                close_paren = name and (")" if "(" in name else "")
                for cx, const in enumerate(plane):
                    if not isinstance(const, pp.LpConstraint):
                        raise TypeError(
                            "Attempting to constrain problem with "
                            f"non-constraint {const}"
                        )
                    const.name = name and f"{name}{cx}{close_paren}"
                    prob += const
            else:
                open_paren = name and ("(" if "(" not in name else "")
                for px, subplane in enumerate(plane):
                    subname = name and f"{name}{open_paren}{px},"
                    _rworker(prob, subplane, subname)

        _rworker(prob, self, name)

    def abs_decompose(self, prob: pp.LpProblem, name: str, *args, bigM=1000, **kwargs):
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
        w = lparray.create_like(f"{name}_abs_aux", self, 0, 1, cat=pp.LpBinary)
        # binding if self >= 0
        (self <= bigM * (1 - w)).constrain(prob, f"{name}_lb")
        # binding if self <= 0
        (self >= -bigM * w).constrain(prob, f"{name}_ub")

        # xp is the positive half of X, xm is the negative half of X
        xp = lparray.create_like(f"{name}_absp", self, *args, **kwargs)
        xm = lparray.create_like(f"{name}_absm", self, *args, **kwargs)

        (xp >= 0).constrain(prob, f"{name}_abs_xplb")
        (xm >= 0).constrain(prob, f"{name}_abs_xmlb")
        (xp - xm == self).constrain(prob, f"{name}_absdecomp")

        # xp >= 0 <=> xm == 0 and vice versa
        (xp <= bigM * (1 - w)).constrain(prob, f"{name}_absxpexcl")
        (xm <= bigM * w).constrain(prob, f"{name}_absxmexcl")

        return xp, xm

    def abs(self, *args, **kwargs) -> lparray:
        """
        Returns variable equal to |self|.

        Thin wrapper around `abs_decompose`
        """
        xp, xm = self.abs_decompose(*args, **kwargs)
        return xp + xm

    def logical_clip(self, prob: pp.LpProblem, name: str, bigM=1000) -> lparray:
        """
        Assumes self is integer >= 0.

        Returns an array of the same shape as self containing
            z_... = max(self_..., 1)

        Generates self.size new variables.
        """

        z = self.__class__.create(
            name, [range(x) for x in self.shape], 0, 1, pp.LpBinary
        )

        (self >= z).constrain(prob, f"{name}_lb")
        (self <= bigM * z).constrain(prob, f"{name}_ub")

        return z

    def sumit(self, *args, **kwargs) -> LpVariable:
        """
        Equivalent to `self.sum().item()`
        """
        out = self.sum(*args, **kwargs)
        return out.item()

    def _lp_minmax(
        self,
        prob: pp.LpProblem,
        name: str,
        which: Literal["min", "max"],
        categ: Literal["Continuous", "Integer", "Binary"],
        *,
        lb=None,
        ub=None,
        bigM=1000,
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
            or any(not isinstance(ax, int) or ax < 0 for ax in axis)
        ):
            raise TypeError("Axis must be a tuple of positive integers")

        if categ == pp.LpBinary:
            lb = 0
            ub = 1
        elif lb is None or ub is None:
            assert 0, "Need to supply constraints for non-binary variables!"

        assert which in ("min", "max")

        mmname = f"{name}_{which}"
        aux_name = f"{name}_{which}_aux"

        # axes of self which the max is indexed by
        keep_axis = tuple(sorted(set(range(self.ndim)) - set(axis)))

        # array of maxes
        minmax_shape = sum((self.shape[ax : ax + 1] for ax in keep_axis), ())
        z: lparray = lparray.create_anon(mmname, minmax_shape, lb, ub, categ)

        # broadcastable version for comparison with self
        minmax_br_index = tuple(
            (slice(None, None, None) if ax in keep_axis else None)
            for ax in range(self.ndim)
        )
        z_br: lparray = z[minmax_br_index]

        w = self.create_like(aux_name, self, lowBound=0, upBound=1, cat=pp.LpBinary)

        (w.sum(axis=axis) == 1).constrain(prob, f"{mmname}_auxsum")

        if which == "max":
            (z_br >= self).constrain(prob, f"{mmname}_lb")
            (z_br <= self + bigM * (1 - w)).constrain(prob, f"{mmname}_ub")
        elif which == "min":
            (z_br <= self).constrain(prob, f"{mmname}_ub")
            (z_br >= self - bigM * (1 - w)).constrain(prob, f"{mmname}_lb")
        else:
            assert 0

        return z

    def _lp_int_minmax(
        self,
        prob: pp.LpProblem,
        name: str,
        which: Literal["min", "max"],
        lb: int,
        ub: int,
        **kwargs,
    ) -> lparray:

        if lb == 0 and ub == 1:
            cat = pp.LpBinary
        else:
            cat = pp.LpInteger

        return self._lp_minmax(
            prob, name, which=which, categ=cat, lb=lb, ub=ub, **kwargs
        )

    def lp_int_max(
        self, prob: pp.LpProblem, name: str, lb: int, ub: int, **kwargs
    ) -> lparray:
        return self._lp_int_minmax(prob, name, which="max", lb=lb, ub=ub, **kwargs)

    def lp_int_min(
        self, prob: pp.LpProblem, name: str, lb: int, ub: int, *args, **kwargs
    ) -> lparray:
        return self._lp_int_minmax(prob, name, which="min", lb=lb, ub=ub, **kwargs)

    def lp_bin_max(self, prob: pp.LpProblem, name: str, *args, **kwargs):
        return self._lp_int_minmax(prob, name, lb=0, ub=1, which="max", **kwargs)

    def lp_bin_min(self, prob: pp.LpProblem, name: str, *args, **kwargs):
        return self._lp_int_minmax(prob, name, lb=0, ub=1, which="min", **kwargs)

    def lp_real_max(self, prob: pp.LpProblem, name: str, **kwargs):
        return self._lp_minmax(prob, name, "max", pp.LpContinuous, **kwargs)

    def lp_real_min(self, prob: pp.LpProblem, name: str, **kwargs):
        return self._lp_minmax(prob, name, "min", pp.LpContinuous, **kwargs)
