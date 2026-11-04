import time
import numpy as np
from search import build_graph, k_paths
from csp import RWA
from topology import load_router_order

order = load_router_order()
H = np.load("data/processed/hourly.npz")["hourly"]
mask = ~np.eye(12, dtype=bool)
X = H[:, mask]
pairs = [(order[i], order[j]) for i in range(12) for j in range(12) if i != j]
adj, coords = build_graph()

HOUR = 700                                  # one hour inside the test week
rank = np.argsort(-X[HOUR])                 # biggest flows first

CONFIGS = [("BT", False, False), ("BT+FC", True, False),
           ("BT+AC3", False, True), ("BT+FC+AC3", True, True)]

print(f"{'W':>2} {'K':>3}  {'method':10s} {'status':12s} {'assign':>7} {'backtr':>7} {'ac3_rm':>6} {'sec':>6}")
for W in (4, 8):
    for K in (10, 20, 40):
        rl = []
        for idx in rank[:K]:
            s, d = pairs[idx]
            found, _, _ = k_paths(adj, coords, s, d, k=3, method="astar")
            rl.append([set(zip(p, p[1:])) for _, p in found])
        csp = RWA(rl, W)
        for name, fc, ac in CONFIGS:
            t0 = time.time()
            sol, status = csp.solve(use_fc=fc, use_ac3=ac)
            print(f"{W:>2} {K:>3}  {name:10s} {status:12s} {csp.assignments:>7} "
                  f"{csp.backtracks:>7} {csp.ac3_removed:>6} {time.time()-t0:>6.2f}")
        print()
