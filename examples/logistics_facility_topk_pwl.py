import numpy as np
import pulp as pp
from pulp import LpBinary, LpContinuous, LpMinimize

from pulp_lparray import lparray


def build_demo_model() -> tuple[pp.LpProblem, dict[str, lparray]]:
    # Dimensions.
    R = 2  # regions
    F = 6  # facilities
    C = 5  # customers
    P = 3  # products

    # Data.
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

    k_per_region = 2

    prob = pp.LpProblem("facility_topk_pwl", LpMinimize)

    open_ = lparray.create(prob, "open", (F,), lowBound=0, upBound=1, cat=LpBinary)
    flow = lparray.create(prob, "flow", (F, C, P), lowBound=0, upBound=None, cat=LpContinuous)

    # Demand satisfaction.
    (flow.sum(axis=0) == demand).constrain(prob, "demand")

    throughput = flow.sum(axis=(1, 2))

    # Capacity only if open.
    (throughput <= cap * open_).constrain(prob, "cap")

    # Exactly k facilities open per region.
    for r in range(R):
        (open_[facility_region == r].sum() == k_per_region).constrain(prob, f"region{r}_k")

    # Piecewise operating cost per facility as function of throughput.
    # Convex cost: cheap up to 5 units, then steep.
    xp = np.stack([np.zeros(F), np.full(F, 5.0), cap], axis=-1)
    fp = np.stack([np.zeros(F), np.full(F, 5.0), 5 + 3 * (cap - 5)], axis=-1)
    op_cost = throughput.lp_interp(prob, "op_cost", xp=xp, fp=fp)

    # Shipping cost.
    ship_cost_3 = ship_cost[:, :, None]
    total_ship = (flow * ship_cost_3).sum().item()
    total_open = (open_ * open_cost).sum().item()
    total_op = op_cost.sum().item()
    prob += total_ship + total_open + total_op

    return prob, {
        "open": open_,
        "flow": flow,
        "throughput": throughput,
        "op_cost": op_cost,
    }


def main() -> None:
    prob, vars_ = build_demo_model()
    status = prob.solve().status
    objective = pp.value(prob.objective)
    open_ = vars_["open"].values
    throughput = vars_["throughput"].values

    _ = (status, objective, open_, throughput)


if __name__ == "__main__":
    main()
