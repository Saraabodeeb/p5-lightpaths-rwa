import time
import numpy as np
from collections import Counter
from integration import routes_for
from csp import RWA

H = np.load("data/processed/hourly.npz")["hourly"]
X = H[:, ~np.eye(12, dtype=bool)]
HOURS = [680, 720, 760, 800, 830]          # 5 runs = 5 different test-week hours
CONFIGS = [("BT", False, False), ("BT+FC", True, False), ("BT+FC+AC3", True, True)]
CAP = 200_000

print(f"{'W':>2} {'K':>3} {'method':10s} {'solved/no_sol/limit':>20s} "
      f"{'assignments':>18s} {'backtracks':>18s} {'sec':>12s}")
for W in (4, 8):
    for K in (10, 20, 40):
        stats = {n: {"status": [], "a": [], "b": [], "t": []} for n, _, _ in CONFIGS}
        for h in HOURS:
            rank = [int(i) for i in np.argsort(-X[h])[:K]]
            rl = [[set(zip(p, p[1:])) for _, p in routes_for(i, 3)] for i in rank]
            for name, fc, ac in CONFIGS:
                csp = RWA(rl, W, max_assignments=CAP)
                t0 = time.time()
                _, status = csp.solve(use_fc=fc, use_ac3=ac)
                s = stats[name]
                s["status"].append(status)
                s["a"].append(csp.assignments)
                s["b"].append(csp.backtracks)
                s["t"].append(time.time() - t0)
        for name, _, _ in CONFIGS:
            s = stats[name]
            c = Counter(s["status"])
            ms = lambda v: f"{np.mean(v):.0f} +- {np.std(v):.0f}"
            print(f"{W:>2} {K:>3} {name:10s} "
                  f"{c['solved']}/{c['no_solution']}/{c['limit']:<14} "
                  f"{ms(s['a']):>18s} {ms(s['b']):>18s} "
                  f"{np.mean(s['t']):>6.2f} +- {np.std(s['t']):.2f}")
        print()
