# pulp-lparray

*PuLP x NumPy, a match made in heaven.*

An `lparray` is a NumPy array of [PuLP](https://github.com/coin-or/pulp) variables, affine
expressions or constraints. Broadcasting, indexing, sums and matrix products build whole
families of expressions at once; `<=`, `>=` and `==` build families of constraints; and a
handful of methods linearise abs, max, min, argmax, top-k, logic and piecewise-linear functions.
Never see a `for` or an index variable again.

It needs Python 3.12+ and PuLP 4. PuLP 4 ships no solver: `pip install "pulp[cbc]"` adds CBC.

## Super Sudoku

A Super Sudoku is a Sudoku in which the cells at the same position within their boxes also hold
distinct digits.

```python
import numpy as np
from pulp import LpBinary, LpProblem

from pulp_lparray import lparray

clues = np.array(
    [
        [0, 4, 0, 0, 1, 8, 0, 0, 0],
        [0, 5, 0, 0, 0, 2, 0, 0, 0],
        [2, 0, 8, 0, 0, 0, 7, 1, 4],
        [1, 8, 2, 0, 0, 0, 5, 6, 0],
        [0, 6, 0, 0, 0, 0, 0, 0, 0],
        [4, 7, 0, 0, 0, 6, 0, 0, 2],
        [0, 0, 5, 0, 9, 7, 0, 0, 1],
        [0, 0, 4, 0, 0, 0, 0, 0, 0],
        [9, 0, 0, 0, 0, 0, 3, 0, 0],
    ]
)  # 0 marks an empty cell

prob = LpProblem("super_sudoku")
# board[R, C, r, c, n] == 1: the cell in row 3R + r and column 3C + c holds digit n + 1
board = lparray.create(prob, "board", (3, 3, 3, 3, 9), cat=LpBinary)
(board.sum(axis=4) == 1).constrain(prob, "cell")
(board.sum(axis=(1, 3)) == 1).constrain(prob, "row")
(board.sum(axis=(0, 2)) == 1).constrain(prob, "column")
(board.sum(axis=(2, 3)) == 1).constrain(prob, "box")
(board.sum(axis=(0, 1)) == 1).constrain(prob, "position")
cells = board.transpose(0, 2, 1, 3, 4).reshape(9, 9, 9)  # row, column, digit
row, col = clues.nonzero()
(cells[row, col, clues[row, col] - 1] == 1).constrain(prob, "clue")
prob.solve()

grid = cells.values.argmax(axis=2) + 1
assert (grid[row, col] == clues[row, col]).all()
assert all(sorted(line) == list(range(1, 10)) for line in [*grid, *grid.T])
```

## Portfolio allocation

Whole numbers of shares whose values best match target weights within a budget, measured by
relative error so that every position counts alike. Minimising `deviation` subject to
`deviation >= error` and `deviation >= -error` makes it `|error|` without binaries; `abs` is for
magnitudes that the objective does not push down.

```python
from pulp import LpInteger, LpMinimize

price = np.array([31.0, 77.0, 12.5, 220.0])
weight = np.array([0.4, 0.3, 0.2, 0.1])
funds = 10_000

prob = LpProblem("allocation", LpMinimize)
shares = lparray.create(prob, "shares", 4, lowBound=0, cat=LpInteger)
(shares @ price <= funds).constrain(prob, "budget")
target = funds * weight / price
error = (shares - target) / target
deviation = lparray.create(prob, "deviation", 4)
(deviation >= error).constrain(prob, "over")
(deviation >= -error).constrain(prob, "under")
prob += deviation.sum().item()
prob.solve()
```

## Facility logistics

Open exactly two facilities per region, ship every product to every customer, and pay an
operating cost with a volume discount on each facility's throughput. The discount makes the cost
concave, so `lp_interp` selects its segment with binaries.

```python
region = np.array([0, 0, 0, 1, 1, 1])
capacity = np.array([12, 6, 6, 12, 6, 6])
open_cost = np.array([10, 5, 7, 10, 5, 7])
ship_cost = np.array(
    [
        [2, 2, 3, 9, 9],
        [3, 1, 2, 9, 9],
        [2, 3, 1, 9, 9],
        [9, 9, 9, 2, 2],
        [9, 9, 9, 3, 1],
        [9, 9, 9, 2, 3],
    ]
)  # facility × customer
demand = np.array([[4, 2, 1], [0, 3, 2], [1, 1, 0], [3, 0, 1], [2, 2, 2]])  # customer × product

prob = LpProblem("logistics", LpMinimize)
is_open = lparray.create(prob, "open", 6, cat=LpBinary)
flow = lparray.create(prob, "flow", (6, 5, 3), lowBound=0)
(flow.sum(axis=0) == demand).constrain(prob, "demand")
throughput = flow.sum(axis=(1, 2))
(throughput <= capacity * is_open).constrain(prob, "capacity")
((region == np.arange(2)[:, None]) @ is_open == 2).constrain(prob, "two_per_region")
# The first 5 units cost 3 each, and the rest 2 each.
operating = throughput.lp_interp(prob, "operating", xp=[0, 5, 12], fp=[0, 15, 29])
prob += ((ship_cost[:, :, None] * flow).sum() + open_cost @ is_open + operating.sum()).item()
prob.solve()

assert (is_open.values.reshape(2, 3).sum(axis=1) == 2).all()
```

## Formulations

Methods taking `(prob, name)` add auxiliary variables and constraints named `{name}_{role}`.

| method | result |
| --- | --- |
| `x.abs(prob, name, bigM=…)` | `\|x\|` |
| `x.abs_decompose(prob, name, bigM=…)` | `(p, m)`: `p - m == x`, `p, m >= 0`, one of them zero |
| `x.logical_clip(prob, name, bigM=…)` | binary `min(x, 1)` of integral `x >= 0` |
| `x.lp_max(prob, name, axis=…, bigM=…)`, `lp_min` | maximum, minimum along `axis` |
| `x.lp_argmax(prob, name, axis=…, bigM=…)`, `lp_argmin` | one-hot indicator of an extremum along `axis` |
| `x.lp_topk(prob, name, k=…, axis=…, bigM=…)` | indicator of `k` largest entries along `axis` |
| `b.lp_bin_max(prob, name, axis=…)`, `lp_bin_min` | logical OR, AND of binary `b` along `axis` |
| `z.lp_bin_and(prob, name, *bs)`, `lp_bin_or` | constrains binary `z` to the AND, OR of `bs` |
| `x.lp_interp(prob, name, xp=…, fp=…)` | `np.interp(x, xp, fp)`, breakpoints broadcasting against `x` |

`bigM` must bound the spread of the values involved: `|x|` for `abs`, `x` for `logical_clip`,
and the range along `axis` for the rest. Too small a `bigM` silently cuts off solutions.
`lp_interp` constrains `x` to lie between the first and last breakpoints rather than clamping it.

## Semantics

`constrain(prob, name)` adds each constraint as `name(i,j,…)` and returns the array of
`LpConstraint`s that the problem stores.

Sums, `@` and `dot` take time linear in the contracted length. Other contractions, such as
`np.einsum` and `np.inner`, run NumPy's object loops, which chain PuLP's `+` and take quadratic
time. Reductions return 0-d lparrays, whose `item()` is the PuLP expression. Ufuncs that are not
affine, such as `np.maximum` and `np.abs`, raise `TypeError`; the methods above linearise them.

Keep the lparray on the left of `+` and `-` against a single PuLP variable or expression:
PuLP treats an array on its right as an iterable and sums it into one expression.

## Upgrading from 1.x

| 1.x | 2.0 |
| --- | --- |
| `lparray.create(name, index_sets)`, `create_anon(name, shape)`, `create_like(name, a)` | `lparray.create(prob, name, shape)` |
| `x.sumit()` | `x.sum().item()` |
| `lp_int_max`, `lp_real_max`, `lp_int_min`, `lp_real_min` | `lp_max`, `lp_min` |
| `lp_bin_argmax`, `lp_bin_argmin`, `lp_bin_topk` | `lp_argmax`, `lp_argmin`, `lp_topk` |
| `piecewise_linear_sos2` | `lp_interp` |
| `abs_clip` | `abs` |
| `one_hot`, `constrain_one_hot`, `all_different`, `indicator_le`, `indicator_ge` | the sum or comparison each wrapped |
| `soft_select_max`, `count_out` | removed |

Every `bigM` is required, and `constrain` returns the stored constraints.
