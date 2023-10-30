# PuLP_LPARRAY

*PuLP x NumPy, a match made in heaven.*

## Examples

I assume a familiarity with [PuLP](https://github.com/coin-or/PuLP).

### SuperSudoku

Let's solve an unconstrained SuperSudoku puzzle. A SuperSudoku is a Sudoku
with the additional requirement that all digits having box coordinates (x, y)
be distinct for all (x, y).

```
import pulp as pp
from pulp_lparray import lparray

#                       name      R, C, r, c, n   lb ub type
X = lparray.create_anon("Board", (3, 3, 3, 3, 9), cat=pp.LpBinary)
prob = pp.LpProblem("SuperSudoku", pp.LpMinimize)
(X.sum(axis=-1) == 1).constrain(prob, "OneDigitPerCell")
(X.sum(axis=(1, 3)) == 1).constrain(prob, "MaxOnePerRow")
(X.sum(axis=(0, 2)) == 1).constrain(prob, "MaxOnePerCol")
(X.sum(axis=(2, 3)) == 1).constrain(prob, "MaxOnePerBox")
(X.sum(axis=(0, 1)) == 1).constrain(prob, "MaxOnePerDust")
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

## Features

It's just PuLP under the hood: `LpVariable`, `LpAffineExpression` and
`LpConstraint` do the heavy lifting.

All the power of numpy for your linear variable sets: broadcasting, reshaping
and indexing tricks galore. Never see a `for` or indexing variable ever again.

Special support functions that allow efficient linearization of
useful operations like min/max, abs, clip-to-binary, boolean operators, and
more. Wide support for the `axis` keyword.
