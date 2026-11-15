"""Experiment 2 (final): BT vs BT+FC vs BT+FC+AC3. Raw CSV, mean+-std table, plot."""
import csv
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ortools.sat.python import cp_model
from integration import routes_for
from csp import RWA

H = np.load("data/processed/hourly.npz")["hourly"]
X = H[:, ~np.eye(12, dtype=bool)]
HOURS = [680, 720, 760, 800, 830]        # 5 runs = 5 test-week hours (the CSP is deterministic)
CONFIGS = [("BT", False, False), ("BT+FC", True, False), ("BT+FC+AC3", True, True)]
CAP = 100_000
SWEEP = list(range(8, 26, 2))
GRID = {3: SWEEP, 4: SWEEP + [40], 8: [10, 20, 40]}
os.makedirs("results", exist_ok=True)


def feasible(rl, W, seconds=10):
    """Independent ground truth (OR-tools CP-SAT), used for labelling only."""
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


rows = []
for W, Ks in GRID.items():
    for K in Ks:
        for h in HOURS:
            rank = [int(i) for i in np.argsort(-X[h])[:K]]
            rl = [[set(zip(p, p[1:])) for _, p in routes_for(i, 3)] for i in rank]
            truth = feasible(rl, W)
            for name, fc, ac in CONFIGS:
                csp = RWA(rl, W, max_assignments=CAP)
                t0 = time.time()
                _, status = csp.solve(use_fc=fc, use_ac3=ac)
                rows.append(dict(W=W, K=K, hour=h, method=name, truth=truth, status=status,
                                 assignments=csp.assignments, backtracks=csp.backtracks,
                                 seconds=round(time.time() - t0, 4)))
        print("done W", W, "K", K, flush=True)

with open("results/exp2_raw.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
print("saved results/exp2_raw.csv")


def sel(W, K, name):
    return [r for r in rows if r["W"] == W and r["K"] == K and r["method"] == name]


def ms(v):
    v = np.array(v, float)
    return f"{v.mean():.0f}+-{v.std():.0f}"


print(f"\ncap={CAP}; 5 runs = 5 hours; mean+-std; 'limit' runs are counted at the cap (lower bound)")
print(f"{'W':>2} {'K':>3} {'method':10s} {'feas/infeas':>11s} {'solved/proved/limit':>20s} "
      f"{'assignments':>14s} {'backtracks':>14s} {'sec':>6s}")
for W, Ks in GRID.items():
    for K in Ks:
        for name, _, _ in CONFIGS:
            s = sel(W, K, name)
            nf = sum(r["truth"] == "feasible" for r in s)
            ni = sum(r["truth"] == "infeasible" for r in s)
            c = [sum(r["status"] == k for r in s) for k in ("solved", "no_solution", "limit")]
            print(f"{W:>2} {K:>3} {name:10s} {nf:>5d}/{ni:<5d} {c[0]:>6d}/{c[1]}/{c[2]:<8d} "
                  f"{ms([r['assignments'] for r in s]):>14s} "
                  f"{ms([r['backtracks'] for r in s]):>14s} "
                  f"{np.mean([r['seconds'] for r in s]):>6.2f}")

fig, ax = plt.subplots(2, 2, figsize=(12, 7), sharex="col")
for col, W in enumerate((3, 4)):
    Ks = SWEEP
    for name, _, _ in CONFIGS:
        mu = [np.mean([r["assignments"] for r in sel(W, K, name)]) for K in Ks]
        sd = [np.std([r["assignments"] for r in sel(W, K, name)]) for K in Ks]
        ls = "--" if name == "BT+FC+AC3" else "-"
        ax[0][col].errorbar(Ks, mu, yerr=sd, marker="o", capsize=3, ls=ls, label=name)
        rate = []
        for K in Ks:
            inf = [r for r in sel(W, K, name) if r["truth"] == "infeasible"]
            rate.append(np.mean([r["status"] == "no_solution" for r in inf]) if inf else np.nan)
        ax[1][col].plot(Ks, rate, marker="o", ls=ls, label=name)
    ax[0][col].axhline(CAP, color="gray", ls=":", label="cap")
    ax[0][col].set_title(f"W={W}: assignments (BT+FC and BT+FC+AC3 overlap)")
    ax[0][col].set_ylabel("assignments (mean +- std, 5 hours)")
    ax[0][col].legend()
    ax[1][col].set_title(f"W={W}: truly infeasible cases proved infeasible")
    ax[1][col].set_ylabel("share proved")
    ax[1][col].set_xlabel("K (number of requests)")
    ax[1][col].set_ylim(-0.05, 1.05)
plt.tight_layout()
plt.savefig("results/exp2.png", dpi=120)
print("saved results/exp2.png")
