"""Live demo: (1) search trace, (2) CSP with and without FC / AC-3, (3) system with NN on vs off."""
import heapq
from itertools import count
import numpy as np
from search import build_graph, k_paths, haversine
from csp import RWA
from integration import routes_for, plan
from forecast import nn_predictions, last_hour_predictions, true_values

ADJ, COORDS = build_graph()
short = lambda n: n.replace("ng", "")


def trace(src, dst, method, max_lines=40):
    """Prints the order in which partial paths are expanded (same logic as search.k_paths with k=1)."""
    tie = count()
    if method == "astar":
        h = lambda n: haversine(COORDS[n], COORDS[dst])
    else:
        h = lambda n: 0.0
    frontier = [(h(src), next(tie), 0.0, [src])]
    expanded = 0
    print(f"\n[{method.upper()}] {short(src)} -> {short(dst)}")
    print(f"{'step':>4} {'expanded path':46s} {'g km':>8} {'h km':>8} {'f km':>8}")
    while frontier:
        f, _, g, path = heapq.heappop(frontier)
        node = path[-1]
        if node == dst:
            print(f"GOAL: {' > '.join(short(p) for p in path)}  cost {g:.1f} km")
            return expanded, g
        expanded += 1
        if expanded <= max_lines:
            print(f"{expanded:>4} {' > '.join(short(p) for p in path):46s} "
                  f"{g:>8.1f} {h(node):>8.1f} {f:>8.1f}")
        for nbr, w in ADJ[node]:
            if nbr in path:
                continue
            g2 = g + w
            heapq.heappush(frontier, (g2 + h(nbr), next(tie), g2, path + [nbr]))
    return expanded, None


def section(title):
    print("\n" + "=" * 72 + "\n" + title + "\n" + "=" * 72)


if __name__ == "__main__":
    section("1. SEARCH TRACE: UCS vs A*")
    src, dst = "SNVAng", "NYCMng"
    for m in ("ucs", "astar"):
        e, g = trace(src, dst, m)
        found, e2, _ = k_paths(ADJ, COORDS, src, dst, k=1, method=m)
        assert e == e2 and abs(g - found[0][0]) < 1e-6, "trace differs from search.py"
        print(f"states expanded: {e}")

    section("2. CSP WITH AND WITHOUT FORWARD CHECKING / AC-3")
    H = np.load("data/processed/hourly.npz")["hourly"]
    X = H[:, ~np.eye(12, dtype=bool)]
    CONFIGS = [("BT", False, False), ("BT+FC", True, False), ("BT+FC+AC3", True, True)]

    def show(title, rl, W, cap):
        print(f"\n{title}")
        print(f"{'method':10s} {'result':12s} {'assignments':>12} {'backtracks':>11} {'AC-3 removed':>13}")
        for name, fc, ac in CONFIGS:
            c = RWA(rl, W, max_assignments=cap)
            _, st = c.solve(use_fc=fc, use_ac3=ac)
            print(f"{name:10s} {st:12s} {c.assignments:>12} {c.backtracks:>11} {c.ac3_removed:>13}")

    show("a) two requests with the same only route, 1 wavelength (impossible)",
         [[{"l"}], [{"l"}]], 1, 100_000)

    K, W, CAP = 16, 3, 100_000
    chosen = None
    for h in (680, 720, 760, 800, 830):
        rank = [int(i) for i in np.argsort(-X[h])[:K]]
        rl = [[set(zip(p, p[1:])) for _, p in routes_for(i, 3)] for i in rank]
        res = {}
        for name, fc, ac in CONFIGS:
            res[name] = RWA(rl, W, max_assignments=CAP).solve(use_fc=fc, use_ac3=ac)[1]
        if res["BT"] == "limit" and res["BT+FC"] == "no_solution":
            chosen = (h, rl)
            break
    if chosen is None:
        chosen = (h, rl)
    show(f"b) real instance: test hour {chosen[0]}, K={K} requests, W={W} wavelengths",
         chosen[1], W, CAP)
    print("'limit' = assignment cap reached: no answer, which is NOT a proof of infeasibility.")
    print("This instance was picked because it shows the contrast. Over 5 hours at K=16, W=3: BT proves infeasibility in 2 of 5, BT+FC in 4 of 5 (results/exp2.txt).")

    section("3. WHOLE SYSTEM WITH NN ON vs OFF (W=4, K=20)")
    true, nn0, last = true_values(), nn_predictions()[0], last_hour_predictions()
    K, W = 20, 4
    print(f"{'test hour':>9}   {'NN on':28s} {'NN off (last hour)':28s}")
    for h in (10, 40, 80, 120, 160):
        top = set(int(i) for i in np.argsort(-true[h])[:K])
        cells = []
        for p in (nn0[h], last[h]):
            r = plan(p, K, W)
            got = set(i for i, _, _ in r["served"]) & top
            cells.append(f"{len(got)}/{K} true top-K served, {len(r['blocked'])} blocked")
        print(f"{h:>9}   {cells[0]:28s} {cells[1]:28s}")
    print("Only 5 hours are shown. All 168 hours with confidence intervals: results/exp3.txt")
