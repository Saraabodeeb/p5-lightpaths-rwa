from collections import Counter
import numpy as np
import integration
from integration import plan, PAIRS, ADJ

H = np.load("data/processed/hourly.npz")["hourly"]
X = H[:, ~np.eye(12, dtype=bool)]
LINKS = {(u, v): w for u in ADJ for v, w in ADJ[u]}


def check_plan(res, scores, K, W):
    """Independent validity check (does not trust the solver)."""
    ranking = set(int(i) for i in np.argsort(-np.asarray(scores))[:K])
    served = [i for i, _, _ in res["served"]]
    blocked = [i for i, _ in res["blocked"]]
    assert len(set(served)) == len(served)
    assert not set(served) & set(blocked)
    assert set(served) | set(blocked) == ranking, "a request was lost"
    used = {}
    for i, path, w in res["served"]:
        s, d = PAIRS[i]
        assert path[0] == s and path[-1] == d and len(set(path)) == len(path)
        assert 0 <= w < W
        cost = 0.0
        for a, b in zip(path, path[1:]):
            assert (a, b) in LINKS, "not a real link"
            cost += LINKS[(a, b)]
            assert (a, b, w) not in used, "wavelength clash"
            used[(a, b, w)] = i
        assert cost <= integration.REACH_KM + 1e-6, "route beyond reach"


def case(name, scores, K, W, check):
    res = plan(scores, K, W)
    check_plan(res, scores, K, W)
    check(res)
    print(f"OK   {name}: served {len(res['served'])}, blocked {len(res['blocked'])}")
    return res


def no_solution(res):
    why = Counter(r for _, r in res["blocked"])
    print("     NO SOLUTION:", "; ".join(f"{n} request(s) - {r}" for r, n in why.items()))

# ---------------- normal ----------------
case("N1 hour 700, K=10, W=8", X[700], 10, 8, lambda r: (len(r["served"]) == 10) or 1 / 0)
case("N2 hour 700, K=20, W=8", X[700], 20, 8, lambda r: (len(r["served"]) == 20) or 1 / 0)
case("N3 hour 740, K=15, W=4", X[740], 15, 4, lambda r: (len(r["served"]) >= 8) or 1 / 0)
rnd = np.random.default_rng(7).random(132)
case("N4 random scores, K=10, W=8", rnd, 10, 8, lambda r: (len(r["served"]) >= 8) or 1 / 0)
case("N5 K=1, W=1", X[700], 1, 1, lambda r: (len(r["served"]) == 1) or 1 / 0)

# ---------------- edge ----------------
case("E1 K=0 (nothing requested)", X[700], 0, 4,
     lambda r: (not r["served"] and not r["blocked"]) or 1 / 0)
case("E2 K=60 > capacity, every request accounted for", X[700], 60, 8,
     lambda r: (len(r["served"]) + len(r["blocked"]) == 60) or 1 / 0)
same = np.ones(132)
a = plan(same, 10, 8)
b = plan(same, 10, 8)
check_plan(a, same, 10, 8)
assert a["served"] == b["served"], "ties are not deterministic"
print(f"OK   E3 all scores tied: deterministic, served {len(a['served'])}")

# ---------------- impossible (clear message) ----------------
r = case("I1 zero wavelengths", X[700], 5, 0,
         lambda r: (not r["served"] and len(r["blocked"]) == 5) or 1 / 0)
no_solution(r)

old = integration.REACH_KM
integration.REACH_KM = 100                       # no route can be that short
integration.routes_for.cache_clear()
far = np.zeros(132)
for s, d in (("LOSAng", "NYCMng"), ("SNVAng", "WASHng"), ("STTLng", "ATLAng")):
    far[PAIRS.index((s, d))] = 1.0
r = case("I2 reach limit 100 km", far, 3, 4,
         lambda r: (not r["served"] and all(w == "no route within reach limit"
                                            for _, w in r["blocked"])) or 1 / 0)
no_solution(r)
integration.REACH_KM = old
integration.routes_for.cache_clear()
print("all system tests passed")
