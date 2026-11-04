import itertools
import networkx as nx
from search import build_graph, k_paths

adj, coords = build_graph()
G = nx.DiGraph()
for u, nbrs in adj.items():
    for v, w in nbrs:
        G.add_edge(u, v, weight=w)
nodes = list(adj)

# 1) known answer: Sunnyvale and Los Angeles are directly connected
paths, _, _ = k_paths(adj, coords, "SNVAng", "LOSAng", k=1, method="ucs")
assert paths[0][1] == ["SNVAng", "LOSAng"], paths
print("known-answer test OK")

# 2) all 132 pairs vs networkx
bad = 0
tot = {m: [0, 0] for m in ("bfs", "ucs", "astar")}
for s, t in itertools.permutations(nodes, 2):
    ref = []
    for p in nx.shortest_simple_paths(G, s, t, weight="weight"):
        ref.append(sum(G[a][b]["weight"] for a, b in zip(p, p[1:])))
        if len(ref) == 3:
            break
    for m in ("ucs", "astar"):
        found, e, g = k_paths(adj, coords, s, t, k=3, method=m)
        costs = [c for c, _ in found]
        if len(costs) != len(ref) or any(abs(a - b) > 1e-6 for a, b in zip(costs, ref)):
            bad += 1
            print("MISMATCH", m, s, t, costs, ref)
        tot[m][0] += e
        tot[m][1] += g
    found, e, g = k_paths(adj, coords, s, t, k=3, method="bfs")
    hops = nx.shortest_path_length(G, s, t)
    assert len(found[0][1]) - 1 == hops, (s, t)
    tot["bfs"][0] += e
    tot["bfs"][1] += g

print("mismatches:", bad)
print("average nodes per request (132 requests, k=3):")
for m, (e, g) in tot.items():
    print(f"  {m:5s} expanded {e/132:6.1f}   generated {g/132:6.1f}")
