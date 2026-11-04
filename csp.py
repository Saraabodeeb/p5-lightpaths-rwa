"""Route and Wavelength Assignment as a CSP.
Backtracking, Forward Checking and AC-3 are separate switches, with counters."""
from collections import deque


class LimitReached(Exception):
    pass


class RWA:
    def __init__(self, route_links, n_wl, max_assignments=200_000):
        """route_links[i] = list of sets of directed links, one set per candidate route of request i.
        Requests must be sorted by priority (highest predicted demand first).
        A value is (route_index, wavelength)."""
        self.n = len(route_links)
        self.route_links = route_links
        self.n_wl = n_wl
        self.max_assignments = max_assignments
        self.domains0 = [[(r, w) for r in range(len(rs)) for w in range(n_wl)]
                         for rs in route_links]
        # share[i][j] = can requests i and j ever use the same directed link?
        self.share = [[False] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    self.share[i][j] = any(a & b for a in route_links[i]
                                           for b in route_links[j])
        self.assignments = 0
        self.backtracks = 0
        self.ac3_removed = 0

    def compat(self, i, vi, j, vj):
        """Binary constraint: same wavelength is not allowed on a shared link."""
        if vi[1] != vj[1]:
            return True
        return not (self.route_links[i][vi[0]] & self.route_links[j][vj[0]])

    def solve(self, use_fc=False, use_ac3=False):
        """Returns (solution or None, status). status: solved / no_solution / limit."""
        self.assignments = 0
        self.backtracks = 0
        self.ac3_removed = 0
        domains = [list(d) for d in self.domains0]
        if any(not d for d in domains):                 # a request with no value at all
            return None, "no_solution"
        try:
            if use_ac3 and not self._ac3(domains):
                return None, "no_solution"
            sol = self._bt({}, domains, use_fc)
        except LimitReached:
            return None, "limit"
        return (sol, "solved") if sol is not None else (None, "no_solution")

    def _bt(self, assign, domains, use_fc):
        if len(assign) == self.n:
            return dict(assign)
        var = len(assign)                               # static order: highest demand first
        for val in list(domains[var]):
            if not all(self.compat(var, val, j, assign[j])
                       for j in assign if self.share[var][j]):
                continue
            self.assignments += 1
            if self.assignments > self.max_assignments:
                raise LimitReached()
            assign[var] = val
            saved, ok = {}, True
            if use_fc:
                for j in range(var + 1, self.n):
                    if not self.share[var][j]:
                        continue
                    keep = [v for v in domains[j] if self.compat(var, val, j, v)]
                    if len(keep) != len(domains[j]):
                        saved[j] = domains[j]
                        domains[j] = keep
                    if not keep:                        # a future request has no value left
                        ok = False
                        break
            if ok:
                res = self._bt(assign, domains, use_fc)
                if res is not None:
                    return res
            for j, d in saved.items():                  # undo forward checking
                domains[j] = d
            del assign[var]
            self.backtracks += 1
        return None

    def _revise(self, domains, i, j):
        keep = [vi for vi in domains[i]
                if any(self.compat(i, vi, j, vj) for vj in domains[j])]
        removed = len(domains[i]) - len(keep)
        domains[i] = keep
        self.ac3_removed += removed
        return removed > 0

    def _ac3(self, domains):
        queue = deque((i, j) for i in range(self.n) for j in range(self.n)
                      if i != j and self.share[i][j])
        while queue:
            i, j = queue.popleft()
            if self._revise(domains, i, j):
                if not domains[i]:
                    return False
                for k in range(self.n):
                    if k != i and k != j and self.share[k][i]:
                        queue.append((k, i))
        return True
