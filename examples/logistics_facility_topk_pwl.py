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

    open_ = lparray.create_anon("open", (F,), lowBound=0, upBound=1, cat=LpBinary)
    flow = lparray.create_anon("flow", (F, C, P), lowBound=0, upBound=None, cat=LpContinuous)

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
    op_cost = lparray.create_anon("op_cost", (F,), lowBound=0, upBound=None, cat=LpContinuous)
    for f in range(F):
        dummy = lparray.create_anon(f"pwl_d{f}", (), cat=LpContinuous)
        x, y = dummy.piecewise_linear_sos2(
            prob,
            f"pwl{f}",
            x=[0, 5, cap[f]],
            y=[0, 5, 5 + 3 * (cap[f] - 5)],
        )
        c = x.item() == throughput[f]
        c.name = f"pwl_link_x{f}"
        prob += c
        c = op_cost[f] == y.item()
        c.name = f"pwl_link_y{f}"
        prob += c

    # Shipping cost.
    ship_cost_3 = ship_cost[:, :, None]
    total_ship = (flow * ship_cost_3).sumit()
    total_open = (open_ * open_cost).sumit()
    total_op = op_cost.sumit()
    prob += total_ship + total_open + total_op

    return prob, {
        "open": open_,
        "flow": flow,
        "throughput": throughput,
        "op_cost": op_cost,
    }


def main() -> None:
    prob, vars_ = build_demo_model()
    prob.solve()

    status = pp.LpStatus[prob.status]
    objective = prob.objective.value()
    open_ = vars_["open"].values
    throughput = vars_["throughput"].values

    _ = (status, objective, open_, throughput)


if __name__ == "__main__":
    main()
