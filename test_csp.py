import itertools
from csp import RWA

CONFIGS = [(False, False), (True, False), (False, True), (True, True)]


def valid(sol, rl):
    for (i, (r1, w1)), (j, (r2, w2)) in itertools.combinations(sol.items(), 2):
        if w1 == w2 and rl[i][r1] & rl[j][r2]:
            return False
    return True


def run(name, rl, n_wl, expect):
    for fc, ac in CONFIGS:
        sol, status = RWA(rl, n_wl).solve(use_fc=fc, use_ac3=ac)
        assert status == expect, (name, fc, ac, status)
        if expect == "solved":
            assert len(sol) == len(rl) and valid(sol, rl), (name, fc, ac)
    print("OK  ", name)


# ---- normal ----
run("N1 two disjoint requests, 1 wavelength", [[{"a"}], [{"b"}]], 1, "solved")
run("N2 two sharing a link, 2 wavelengths", [[{"l"}], [{"l"}]], 2, "solved")
run("N3 three sharing a link, 3 wavelengths", [[{"l"}]] * 3, 3, "solved")
run("N4 second route avoids the clash, 1 wavelength",
    [[{"l1"}, {"l2"}], [{"l1"}]], 1, "solved")
run("N5 chain A-B, B-C, 2 wavelengths",
    [[{"l1"}], [{"l1", "l2"}], [{"l2"}]], 2, "solved")

# ---- edge ----
run("E1 zero requests", [], 3, "solved")
run("E2 one request, one route, one wavelength", [[{"l"}]], 1, "solved")
run("E3 a request with no route at all", [[]], 2, "no_solution")

# ---- impossible ----
run("I1 two requests, same only route, 1 wavelength", [[{"l"}], [{"l"}]], 1, "no_solution")
run("I2 four requests on one link, 3 wavelengths", [[{"l"}]] * 4, 3, "no_solution")

# ---- AC-3 should detect I1 before any assignment ----
c = RWA([[{"l"}], [{"l"}]], 1)
sol, status = c.solve(use_ac3=True)
assert status == "no_solution" and c.assignments == 0
print("OK   AC-3 detects I1 with 0 assignments")
print("all CSP tests passed")
