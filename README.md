# pulp-lparray

*PuLP x NumPy, a match made in heaven.*

An `lparray` is a NumPy array of [PuLP](https://github.com/coin-or/pulp) variables, affine
expressions or constraints. Broadcasting, indexing, sums and matrix products build whole
families of expressions at once; `<=`, `>=` and `==` build families of constraints; and a
handful of methods linearise abs, max, min, argmax, top-k, logic and piecewise-linear functions.
Never see a `for` or an index variable again.

It needs Python 3.12+ and PuLP 4, which ships no solver; `pip install pulp-lparray "pulp[cbc]"`
installs it with CBC.

Each example uses the imports and names of the examples before it.

## Super Sudoku

A Super Sudoku is a Sudoku in which the cells at the same position within their boxes also hold
distinct digits. The board starts empty.

```python
from pulp import LpBinary, LpProblem

from pulp_lparray import lparray

prob = LpProblem("super_sudoku")
# board[R, C, r, c, n] == 1: the cell in row 3R + r and column 3C + c holds digit n + 1
board = lparray.create(prob, "board", (3, 3, 3, 3, 9), cat=LpBinary)
(board.sum(axis=4) == 1).constrain(prob, "cell")
(board.sum(axis=(1, 3)) == 1).constrain(prob, "row")
(board.sum(axis=(0, 2)) == 1).constrain(prob, "column")
(board.sum(axis=(2, 3)) == 1).constrain(prob, "box")
(board.sum(axis=(0, 1)) == 1).constrain(prob, "position")
prob.solve()
print(board.values.argmax(axis=4).transpose(0, 2, 1, 3).reshape(9, 9) + 1)
```

```text
[[2 3 1 5 6 8 7 4 9]
 [7 4 6 2 3 9 8 5 1]
 [9 8 5 7 4 1 3 6 2]
 [4 1 2 3 9 5 6 8 7]
 [3 6 7 4 8 2 9 1 5]
 [5 9 8 1 7 6 4 2 3]
 [8 2 3 9 5 4 1 7 6]
 [1 7 4 6 2 3 5 9 8]
 [6 5 9 8 1 7 2 3 4]]
```

## Image denoising

The denoised image minimises the number of pixels changed plus 0.6 for each pair of adjacent
pixels that differ. Minimising `edge` subject to `edge >= jump` and `edge >= -jump` makes it
`|jump|` without binaries; `abs` is for magnitudes that the objective does not push down.

```python
import numpy as np
from pulp import LpMinimize

rng = np.random.default_rng()
y, x = np.ogrid[-10:11, -20:21]
clean = np.hypot(x / 2, y) % 8 < 4  # a bullseye, stretched to print round
noisy = clean ^ (rng.random(clean.shape) < 0.05)

prob = LpProblem("denoise", LpMinimize)
image = lparray.create(prob, "image", noisy.shape, cat=LpBinary)
jump = np.concatenate([np.diff(image, axis=0).ravel(), np.diff(image, axis=1).ravel()])
edge = lparray.create(prob, "edge", jump.shape)
(edge >= jump).constrain(prob, "rise")
(edge >= -jump).constrain(prob, "fall")
prob += (image[~noisy].sum() + (1 - image[noisy]).sum() + 0.6 * edge.sum()).item()
prob.solve()

before, after = np.where([noisy, image.values > 0.5], "#", ".")
print("\n".join(map("".join, np.hstack([before, np.full((21, 1), " "), after]))))
```

```text
..##...####.#.######.#############....... .......###########################.......
.....###############################..... .....###############################.....
...#######.###############.##.#.######... ...###################################...
.############..........#...#############. .############..............#############.
##########.....................#.######## ##########.......................########
########.........................######## ########.........................########
#######................#..........####### #######...........................#######
######..........##########.........###### ######..........##########.........######
######......#.#############.........#.### ######........#############.........#####
#####......#.###############......#.##### #####........###############........#####
#####........###############........##### #####........###############........#####
#####.....#..###############........##### #####........###############........#####
#####.......#.#############....#....##### #####.........#############.........#####
######.........###########.........#.#### ######.........###########.........######
#######...................#........###### #######............................######
########.........#...............######## ########.........................########
#.########.......#.#...........########## ##########.....................##########
.###.########...............############. .############...............############.
...###.###############################.#. ...###################################...
..#..#############################.#..... .....###############################.....
.......######.##.#######.#####.###.#..... .......#############################.....
```

## Nurse rostering

Seven nurses cover a fortnight of early, late and night shifts. Nobody works twice in a day, an
early shift after a night, more than five days in any seven or more than two weekend days, and
the busiest nurse works as few shifts as possible.

```python
from numpy.lib.stride_tricks import sliding_window_view

nurses = ["Ana", "Ben", "Cai", "Dee", "Eli", "Fay", "Gus"]
weekend = np.arange(14) % 7 >= 5
need = np.where(weekend[:, None], [1, 1, 1], [2, 2, 1])  # day × early, late, night

prob = LpProblem("roster", LpMinimize)
shift = lparray.create(prob, "shift", (7, 14, 3), cat=LpBinary)  # nurse × day × shift
(shift.sum(axis=0) >= need).constrain(prob, "cover")
works = shift.sum(axis=2)
(works <= 1).constrain(prob, "once_a_day")
(shift[:, :-1, 2] + shift[:, 1:, 0] <= 1).constrain(prob, "rest")
week = sliding_window_view(works, 7, axis=1)  # nurse × window × day
(week.sum(axis=2) <= 5).constrain(prob, "five_in_seven")  # ty: ignore[unresolved-attribute]
(works[:, weekend].sum(axis=1) <= 2).constrain(prob, "weekends")
busiest = lparray.create(prob, "busiest", ())
(works.sum(axis=1) <= busiest).constrain(prob, "fair")
prob += busiest.item()
prob.solve()

codes = np.array([*".ELN"])[(shift.values @ [1, 2, 3]).round().astype(int)]
print("   ", *("MTWTFSS" * 2))
print("\n".join(map(" ".join, np.c_[nurses, codes])))
```

```text
    M T W T F S S M T W T F S S
Ana L . E . N L N . E . E L . .
Ben N L . E E . . N L N L E . .
Cai E E N L L . . E N . . . L L
Dee L E L . . . L L L E N . . N
Eli . N L N . E . E . L E N . E
Fay E . E E L . E . E L . L E .
Gus . L . L E N . L . E L E N .
```

## Best subset regression

The fit uses at most three of eight regressors and minimises the sum of absolute residuals. Each
coefficient is zero unless its regressor is in the support, and 10 bounds its magnitude.

```python
truth = np.array([0, 2, 0, -1.5, 0, 0, 3, 0])
X = rng.normal(size=(40, 8))
y = X @ truth + rng.normal(scale=0.3, size=40)

prob = LpProblem("best_subset", LpMinimize)
beta = lparray.create(prob, "beta", 8)
support = lparray.create(prob, "support", 8, cat=LpBinary)
(beta <= 10 * support).constrain(prob, "upper")
(beta >= -10 * support).constrain(prob, "lower")
(support.sum() <= 3).constrain(prob, "sparsity")
residual = y - X @ beta
error = lparray.create(prob, "error", 40)
(error >= residual).constrain(prob, "over")
(error >= -residual).constrain(prob, "under")
prob += error.sum().item()
prob.solve()
print(beta.values.round(2))
```

```text
[ 0.    2.01  0.   -1.6   0.    0.    3.06  0.  ]
```

## Facility location

Six candidate sites supply forty towns at the least total cost of building, shipping and
operating. A volume discount makes the operating cost concave, so `lp_interp` selects its segment
with binaries. The map shows each town as the letter of the site that supplies most of it, and
each unbuilt site as `+`.

```python
land = np.full((12, 36), ".")
town = rng.integers(0, land.shape, size=(40, 2))
site = rng.integers(0, land.shape, size=(6, 2))
demand = rng.integers(1, 10, size=40)
distance = np.linalg.norm(site[:, None] - town, axis=2)  # site × town

prob = LpProblem("facility_location", LpMinimize)
build = lparray.create(prob, "build", 6, cat=LpBinary)
flow = lparray.create(prob, "flow", (6, 40), lowBound=0)
(flow.sum(axis=0) == demand).constrain(prob, "demand")
throughput = flow.sum(axis=1)
(throughput <= 120 * build).constrain(prob, "capacity")
# Operating a site costs 5 a unit for its first 20 units and 1 a unit beyond.
operating = throughput.lp_interp(prob, "operating", xp=[0, 20, 120], fp=[0, 100, 200])
prob += ((distance * flow).sum() + 50 * build.sum() + operating.sum()).item()
prob.solve()

land[tuple(town.T)] = np.array([*"abcdef"])[flow.values.argmax(axis=0)]
land[tuple(site.T)] = np.where(build.values > 0.5, [*"ABCDEF"], "+")
print("\n".join(map("".join, land)))
```

```text
f..............e....a....A....a.....
.......................+...........a
..f.........................a......a
.f.....e.e..................a.......
...........e.E.e.........a..........
............e.........c..a........a.
.......f............................
....f..............c...c......a..a..
.f.......f.......c.C................
..f.........e......c......c.c.......
............e....c..................
f....fF........c....c........+......
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
NumPy functions such as `np.concatenate` and `sliding_window_view` return lparrays, although
NumPy's stubs type their results as plain ndarrays.

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
