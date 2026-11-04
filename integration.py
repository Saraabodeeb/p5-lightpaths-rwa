"""Integration: scores -> Search (routes) -> CSP (wavelengths) -> feedback loop."""
import time
from collections import Counter
from functools import lru_cache
import numpy as np
from search import build_graph, k_paths
from csp import RWA
from topology import load_router_order

REACH_KM = 5000      # GENERATED parameter (document it in the data dictionary)
CAP = 20_000         # max CSP assignments per solve; 'limit' is treated as a failure

order = load_router_order()
PAIRS = [(order[i], order[j]) for i in range(12) for j in range(12) if i != j]
ADJ, COORDS = build_graph()


@lru_cache(maxsize=None)
def routes_for(pair_idx, k):
    """Up to k loop-free routes (A*), keeping only those within the reach limit."""
    s, d = PAIRS[pair_idx]
    found, _, _ = k_paths(ADJ, COORDS, s, d, k=k, method="astar")
    return tuple((c, tuple(p)) for c, p in found if c <= REACH_KM)


def plan(scores, K, W, cap=CAP, use_fc=True, use_ac3=True):
    """scores: 132 predicted demands. Returns served / blocked lists and counters."""
    ranking = [int(i) for i in np.argsort(-np.asarray(scores))[:K]]   # priority order
    blocked, active = [], []
    for i in ranking:                       # no route within reach -> blocked right away
        if routes_for(i, 3):
            active.append(i)
        else:
            blocked.append((i, "no route within reach limit"))

    loops, assignments, sol, routes = 0, 0, None, []
    while active:
        for k_routes in (3, 5):             # 2nd attempt = Search re-run with more routes
            routes = [routes_for(i, k_routes) for i in active]
            link_sets = [[set(zip(p, p[1:])) for _, p in r] for r in routes]
            csp = RWA(link_sets, W, max_assignments=cap)
            sol, status = csp.solve(use_fc=use_fc, use_ac3=use_ac3)
            loops += 1
            assignments += csp.assignments
            if sol is not None:
                break
        if sol is not None:
            break
        why = ("wavelengths used up (proved)" if status == "no_solution"
               else "wavelengths used up (search limit reached, not proved)")
        blocked.append((active.pop(), why))  # drop the lowest-priority request

    served = []
    if sol is not None:
        for n, i in enumerate(active):
            r, w = sol[n]
            served.append((i, routes[n][r][1], w))
    return {"served": served, "blocked": blocked, "loops": loops, "assignments": assignments}


if __name__ == "__main__":
    H = np.load("data/processed/hourly.npz")["hourly"]
    X = H[:, ~np.eye(12, dtype=bool)]
    ORACLE = {(680, 3): 15, (680, 4): 19, (680, 8): 57,
              (720, 3): 19, (720, 4): 23, (720, 8): 61,
              (760, 3): 11, (760, 4): 14, (760, 8): 53}

    print(f"{'hour':>5} {'W':>2} {'K':>3} | {'served':>6} {'blocked':>7} {'loops':>5} "
          f"{'assign':>7} {'sec':>6} | oracle max K")
    for hour in (680, 720, 760):
        for W, K in ((3, 20), (4, 40), (8, 40)):
            t0 = time.time()
            res = plan(X[hour], K, W)
            print(f"{hour:>5} {W:>2} {K:>3} | {len(res['served']):>6} {len(res['blocked']):>7} "
                  f"{res['loops']:>5} {res['assignments']:>7} {time.time()-t0:>6.1f} | "
                  f"{ORACLE[(hour, W)]}")
            for reason, n in Counter(r for _, r in res["blocked"]).items():
                print(f"        blocked {n}: {reason}")

    print("\nExample plan (hour 680, W=4, K=40):")
    res = plan(X[680], 40, 4)
    for i, path, w in res["served"][:5]:
        print(f"  {PAIRS[i][0]} -> {PAIRS[i][1]}  wavelength {w}  route {' > '.join(path)}")
    for i, why in res["blocked"][:3]:
        print(f"  BLOCKED {PAIRS[i][0]} -> {PAIRS[i][1]}: {why}")

    print("\nImpossible case (0 wavelengths):")
    res = plan(X[680], 5, 0)
    if not res["served"]:
        print(f"  NO SOLUTION: all {len(res['blocked'])} requests blocked "
              f"(no wavelengths available at this capacity).")
