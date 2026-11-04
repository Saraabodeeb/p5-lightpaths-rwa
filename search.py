"""BFS / UCS / A* returning up to k loop-free routes, with node counters."""
import heapq
import math
from itertools import count
from topology import load_topology


def haversine(a, b):
    """Great-circle distance in km between (lat, lon) points."""
    r = 6371.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp = p2 - p1
    dl = math.radians(b[1] - a[1])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def build_graph(path="data/topo-2003-04-10.txt"):
    routers, links = load_topology(path)
    coords = {n: (v[1], v[2]) for n, v in routers.items()}
    adj = {n: [] for n in routers}
    for x, y, _cap, _ospf in links:          # each direction is its own directed link
        adj[x].append((y, haversine(coords[x], coords[y])))
    return adj, coords


def k_paths(adj, coords, src, dst, k=3, method="ucs"):
    """Return (list of (cost_km, path), nodes_expanded, nodes_generated).

    method: "bfs" (fewest hops), "ucs" (shortest distance), "astar" (distance + straight-line h).
    Paths are loop-free: a node already on the path is never added again.
    """
    tie = count()

    def priority(g, hops, node):
        if method == "bfs":
            return hops
        if method == "ucs":
            return g
        return g + haversine(coords[node], coords[dst])      # astar

    frontier = [(priority(0.0, 0, src), next(tie), 0.0, [src])]
    found, expanded, generated = [], 0, 1
    while frontier and len(found) < k:
        _, _, g, path = heapq.heappop(frontier)
        node = path[-1]
        if node == dst:
            found.append((g, path))
            continue                                          # do not expand beyond the goal
        expanded += 1
        for nbr, w in adj[node]:
            if nbr in path:                                   # loop-free
                continue
            generated += 1
            g2 = g + w
            heapq.heappush(frontier, (priority(g2, len(path), nbr), next(tie), g2, path + [nbr]))
    return found, expanded, generated


if __name__ == "__main__":
    adj, coords = build_graph()
    for m in ("bfs", "ucs", "astar"):
        paths, e, g = k_paths(adj, coords, "SNVAng", "NYCMng", k=3, method=m)
        print(f"[{m}] expanded={e} generated={g}")
        for cost, p in paths:
            print(f"   {cost:8.1f} km  {' -> '.join(p)}")
