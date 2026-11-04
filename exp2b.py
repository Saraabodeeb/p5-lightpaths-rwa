"""Experiment 2b: K sweep around the feasibility threshold, labelled by an independent solver."""
import numpy as np
from collections import Counter
from ortools.sat.python import cp_model
from integration import routes_for
from csp import RWA

H = np.load("data/processed/hourly.npz")["hourly"]
X = H[:, ~np.eye(12, dtype=bool)]
HOURS = [680, 720, 760, 800, 830]
CONFIGS = [("BT", False, False), ("BT+FC", True, False), ("BT+FC+AC3", True, True)]
CAP = 100_000


def feasible(rl, W, seconds=10):
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
        m.AddAtMostOne(vs)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = seconds
    st = s.Solve(m)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return "feasible"
    return "infeasible" if st == cp_model.INFEASIBLE else "unknown"


print(f"cap = {CAP} assignments. Per method: solved/proved-infeasible/limit (mean assignments)")
for W in (3, 4):
    for K in range(8, 26, 2):
        truth = Counter()
        stats = {n: {"st": [], "a": []} for n, _, _ in CONFIGS}
        for h in HOURS:
            rank = [int(i) for i in np.argsort(-X[h])[:K]]
            rl = [[set(zip(p, p[1:])) for _, p in routes_for(i, 3)] for i in rank]
            truth[feasible(rl, W)] += 1
            for name, fc, ac in CONFIGS:
                csp = RWA(rl, W, max_assignments=CAP)
                _, status = csp.solve(use_fc=fc, use_ac3=ac)
                stats[name]["st"].append(status)
                stats[name]["a"].append(csp.assignments)
        line = (f"W={W} K={K:>2} | truth: feasible {truth['feasible']} "
                f"infeasible {truth['infeasible']} unknown {truth['unknown']} |")
        for name, _, _ in CONFIGS:
            c = Counter(stats[name]["st"])
            line += f" {name} {c['solved']}/{c['no_solution']}/{c['limit']} ({np.mean(stats[name]['a']):.0f})"
        print(line, flush=True)
