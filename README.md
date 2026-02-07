# PuLP_LPARRAY

*PuLP x NumPy, a match made in heaven.*

## Examples

I assume a familiarity with [PuLP](https://github.com/coin-or/PuLP).

### SuperSudoku

Let's solve an unconstrained SuperSudoku puzzle. A SuperSudoku is a Sudoku
with the additional requirement that all digits having box coordinates (x, y)
be distinct for all (x, y).

```import pulp as pp
from lparray import lparray

#                       name      R, C, r, c, n   lb ub type
X = lparray.create_anon("Board", (3, 3, 3, 3, 9), 0, 1, pp.LpBinary)
prob = pp.LpProblem("SuperSudoku", pp.LpMinimize)
(X.sum(axis=-1) == 1).constrain(prob, "OneDigitPerCell")
(X.sum(axis=(1, 3)) == 1).constrain(prob, "MaxOnePerRow")
(X.sum(axis=(0, 2)) == 1).constrain(prob, "MaxOnePerCol")
(X.sum(axis=(2, 3)) == 1).constrain(prob, "MaxOnePerBox")
(X.sum(axis=(0, 1)) == 1).constrain(prob, "MaxOnePerXY")
prob.solve()
board = X.values.argmax(axis=-1)
print(board)
```

Of course, in serious scientific use one would change all of the variable and
constraint names to be one character tokens; however, I hope the capacity for
terseness is clear.


### Excerpt From a Flow Problem

From the wild: we want to make sure that in a transport graph, the maximum flow
through nodes close to the sinks is bounded by a function of the distance from
the sinks:

```
def impose_flow_thinning(limit, dist, Dists: lparray):
    xp, _ = (-Dist + dist + 1).abs_decompose(prob, f"FlowThin{dist}Abs", 0, MAX_RANK, pp.LpInteger)
    le_dist_mask = xp.logical_clip(prob, f"FlowThin{dist}Lclip")
    (
        N * Fs[:, :, :] <= limit + (N - limit) * (1 - le_dist_mask)[:, None, None]
    ).constrain(prob, f"FlowThin{dist}")
```

### Finding an optimal stock allocation

From the wild: we want an integral portfolio allocation that mostly closely matches a fractional target.
We could use a rounding heuristic to approximate this, but PuLP-LPARRAY lets us do the correct thing easily.

```
alloc = lparray.create_anon("Alloc", shape=target.shape, cat=pulp.LpInteger)
(alloc >= 0).constrain(prob, "NonNegativePositions")

cost = (alloc @ price_arr).sum()
(cost <= funds).constrain(prob, "DoNotExceedFunds")

loss = (
    # rescale by inverse composition to punish relative deviations equally
    ((alloc - target_alloc) * (1 / target))
    .abs(prob, "Loss", bigM=1_000_000)
    .sumit()
)
prob += loss
```

### Logistics: facility activation + multi-commodity flows + piecewise costs

From the wild: we have customers with product demands, a set of candidate
facilities partitioned into regions, and we want to open *exactly K per region*.
We also want convex, piecewise-linear operating cost as a function of facility
throughput.

The whole thing is a few array ops and a small loop over facilities for the
piecewise costs:

```python
import numpy as np
import pulp as pp
from pulp import LpBinary, LpContinuous, LpMinimize

from pulp_lparray import lparray

R, F, C, P = 2, 6, 5, 3
facility_region = np.array([0, 0, 0, 1, 1, 1])

demand = np.array(
    [
        [4, 2, 1],
        [0, 3, 2],
        [1, 1, 0],
        [3, 0, 1],
        [2, 2, 2],
    ],
    dtype=float,
)  # (C, P)

cap = np.array([12, 6, 6, 12, 6, 6], dtype=float)  # (F,)
open_cost = np.array([10, 5, 7, 10, 5, 7], dtype=float)  # (F,)

ship_cost = np.array(
    [
        [2, 2, 3, 9, 9],
        [3, 1, 2, 9, 9],
        [2, 3, 1, 9, 9],
        [9, 9, 9, 2, 2],
        [9, 9, 9, 3, 1],
        [9, 9, 9, 2, 3],
    ],
    dtype=float,
)  # (F, C)

K = 2
prob = pp.LpProblem("facility_topk_pwl", LpMinimize)

open_ = lparray.create_anon("open", (F,), lowBound=0, upBound=1, cat=LpBinary)
flow = lparray.create_anon("flow", (F, C, P), lowBound=0, upBound=None, cat=LpContinuous)

(flow.sum(axis=0) == demand).constrain(prob, "demand")

throughput = flow.sum(axis=(1, 2))
(throughput <= cap * open_).constrain(prob, "cap")

for r in range(R):
    (open_[facility_region == r].sum() == K).constrain(prob, f"region{r}_k")

op_cost = lparray.create_anon("op_cost", (F,), lowBound=0, upBound=None, cat=LpContinuous)
for f in range(F):
    dummy = lparray.create_anon(f"pwl_d{f}", (), cat=LpContinuous)
    x, y = dummy.piecewise_linear_sos2(
        prob,
        f"pwl{f}",
        x=[0, 5, cap[f]],
        y=[0, 5, 5 + 3 * (cap[f] - 5)],
    )
    prob += x.item() == throughput[f]
    prob += op_cost[f] == y.item()

total_ship = (flow * ship_cost[:, :, None]).sumit()
total_open = (open_ * open_cost).sumit()
total_op = op_cost.sumit()
prob += total_ship + total_open + total_op

prob.solve()
```

## Features

It's just PuLP under the hood: `LpVariable`, `LpAffineExpression` and
`LpConstraint` do the heavy lifting.

All the power of numpy for your linear variable sets: broadcasting, reshaping
and indexing tricks galore. Never see a `for` or indexing variable ever again.

Special support functions that allow efficient linearization of
useful operations like min/max, abs, clip-to-binary, boolean operators, and
more. Wide support for the `axis` keyword.
