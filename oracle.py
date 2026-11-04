"""Ground truth with an independent solver (OR-tools CP-SAT), used for validation only."""
import numpy as np
from ortools.sat.python import cp_model
from search import build_graph, k_paths
from csp import RWA
from topology import load_router_order

order = load_router_order()
H = np.load("data/processed/hourly.npz")["hourly"]
mask = ~np.eye(12, dtype=bool)
X = H[:, mask]
pairs = [(order[i], order[j]) for i in range(12) for j in range(12) if i != j]
adj, coords = build_graph()

ROUTES = []                                   # 3 routes for each of the 132 pairs
for s, d in pairs:
    found, _, _ = k_paths(adj, coords, s, d, k=3, method="astar")
    ROUTES.append([set(zip(p, p[1:])) for _, p in found])


def feasible(rl, W, seconds=20):
    m = cp_model.CpModel()
    x = {}
    for i, routes in enumerate(rl):
        for r in range(len(routes)):
            for w in range(W):
                x[i, r, w] = m.NewBoolVar(f"x{i}_{r}_{w}")
        m.AddExactlyOne([x[i, r, w] for r in range(len(routes)) for w in range(W)])
    users = {}
    for (i, r, w), v in x.items():
        for link in rl[i][r]:
            users.setdefault((link, w), []).append(v)
    for vs in users.values():
        m.AddAtMostOne(vs)                    # one request per (link, wavelength)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    st = solver.Solve(m)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return "feasible"
    return "infeasible" if st == cp_model.INFEASIBLE else "unknown"


def max_feasible_k(rank, W):
    lo, hi = 0, len(rank)                     # lo is always feasible
    while lo < hi:
        mid = (lo + hi + 1) // 2
        res = feasible([ROUTES[i] for i in rank[:mid]], W)
        if res == "feasible":
            lo = mid
        elif res == "infeasible":
            hi = mid - 1
        else:
            return None
    return lo


print("max K = largest number of top flows that can ALL be served (3 routes each)")
print(f"{'hour':>5} {'W':>2} {'max K':>6} | our CSP (BT+FC+AC3) at K*, at K*+1")
for hour in (680, 720, 760, 800, 830):
    rank = np.argsort(-X[hour])               # real traffic for now
    for W in (3, 4, 8):
        k = max_feasible_k(rank, W)
        if k is None:
            print(f"{hour:>5} {W:>2}   oracle timeout")
            continue
        out = []
        for kk in (k, k + 1):
            if kk == 0 or kk > 132:
                out.append("-")
                continue
            rl = [ROUTES[i] for i in rank[:kk]]
            _, status = RWA(rl, W).solve(use_fc=True, use_ac3=True)
            out.append(status)
        print(f"{hour:>5} {W:>2} {k:>6} | {out[0]:>10} {out[1]:>10}")
