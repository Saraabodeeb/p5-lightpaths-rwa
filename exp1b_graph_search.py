"""Supplementary experiment 1b: tree/path search (our main implementation) vs graph search with an explored set (k=1)."""
import heapq
import itertools
import random
import time
from itertools import count
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from search import build_graph, k_paths, haversine
from exp1 import build_graph_gml, to_nx, METHODS, SEEDS


def graph_search(adj, coords, src, dst, method):
    """Graph search with an explored set. Returns (cost_km, path, expanded_unique_nodes, generated)."""
    tie = count()

    def prio(g, hops, node):
        if method == "bfs":
            return hops
        if method == "ucs":
            return g
        return g + haversine(coords[node], coords[dst])          # astar

    frontier = [(prio(0.0, 0, src), next(tie), 0.0, 0, src, (src,))]
    closed, expanded, generated = set(), 0, 1
    while frontier:
        _, _, g, hops, node, path = heapq.heappop(frontier)
        if node in closed:
            continue
        if node == dst:                                          # goal is not counted as an expansion (same as search.py)
            return g, list(path), expanded, generated
        closed.add(node)
        expanded += 1
        for nbr, w in adj[node]:
            if nbr in closed:
                continue
            generated += 1
            g2 = g + w
            heapq.heappush(frontier, (prio(g2, hops + 1, nbr), next(tie), g2, hops + 1, nbr, path + (nbr,)))
    return None, None, expanded, generated


def validate(name, adj, coords):
    G = to_nx(adj)
    bad, bfs_cost_diff, n = 0, 0, 0
    for s, t in itertools.permutations(list(adj), 2):
        n += 1
        best = nx.shortest_path_length(G, s, t, weight="weight")
        hops = nx.shortest_path_length(G, s, t)
        for m in METHODS:
            c, p, _, _ = graph_search(adj, coords, s, t, m)
            tree_c = k_paths(adj, coords, s, t, k=1, method=m)[0][0][0]
            if m == "bfs":
                bad += (len(p) - 1 != hops)
                bfs_cost_diff += (abs(c - tree_c) > 1e-6)
            else:
                bad += (abs(c - best) > 1e-6 or abs(c - tree_c) > 1e-6)
    print(f"[{name}] {n} pairs | graph search errors vs networkx / tree search: {bad} | "
          f"BFS route km differs from tree BFS in {bfs_cost_diff} pairs")


def timed(fn, reps=5):
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1000.0            # ms


def measure(adj, coords, seed, n_pairs=30):
    rng = random.Random(seed)                                    # same pairs as exp1.py
    pairs = rng.sample(list(itertools.permutations(list(adj), 2)), n_pairs)
    res = {}
    for m in METHODS:
        vals = []
        for s, t in pairs:
            _, te, tg = k_paths(adj, coords, s, t, k=1, method=m)
            _, _, ge, gg = graph_search(adj, coords, s, t, m)
            t_tree = timed(lambda: k_paths(adj, coords, s, t, k=1, method=m))
            t_graph = timed(lambda: graph_search(adj, coords, s, t, m))
            vals.append((te, ge, tg, gg, t_tree, t_graph))
        res[m] = np.mean(vals, axis=0)
    return res


if __name__ == "__main__":
    nets = {"Abilene (12 nodes)": build_graph(),
            "germany50 (50 nodes)": build_graph_gml("data/germany50.gml")}
    print("VALIDATION")
    for name, (adj, coords) in nets.items():
        validate(name, adj, coords)

    rows = {}
    for name, (adj, coords) in nets.items():
        per_seed = [measure(adj, coords, s) for s in SEEDS]
        for m in METHODS:
            arr = np.array([ps[m] for ps in per_seed])           # (5 seeds, 6 metrics)
            rows[(name, m)] = (arr.mean(axis=0), arr.std(axis=0))

    print("\nEXPERIMENT 1b (k=1, 30 pairs per seed, 5 seeds, mean +- std over seeds)")
    print("tree = partial paths (main implementation) | graph = unique nodes with explored set")
    print(f"{'network':22s} {'method':6s} {'expanded tree':>15s} {'expanded graph':>15s} "
          f"{'generated tree':>15s} {'generated graph':>16s} {'ms tree':>12s} {'ms graph':>12s}")
    for (name, m), (mu, sd) in rows.items():
        cells = " ".join(f"{mu[i]:>7.1f}+-{sd[i]:<5.1f}" if i < 4 else f"{mu[i]:>6.3f}+-{sd[i]:<4.3f}"
                         for i in range(6))
        print(f"{name:22s} {m:6s} {cells}")

    print("\nUCS / A* ratio of expanded states (means):")
    for name in nets:
        u, a = rows[(name, "ucs")][0], rows[(name, "astar")][0]
        print(f"  {name:22s} tree search {u[0] / a[0]:6.1f}x | graph search {u[1] / a[1]:6.1f}x")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, name in zip(axes, nets):
        for j, (lab, i) in enumerate((("tree (partial paths)", 0), ("graph (explored set)", 1))):
            mu = [rows[(name, m)][0][i] for m in METHODS]
            sd = [rows[(name, m)][1][i] for m in METHODS]
            ax.bar(np.arange(3) + 0.4 * j - 0.2, mu, 0.4, yerr=sd, capsize=3, label=lab)
        ax.set_xticks(range(3))
        ax.set_xticklabels([m.upper() for m in METHODS])
        ax.set_title(name + ", k=1")
        ax.set_ylabel("expanded (mean +- std, 5 seeds)")
        ax.legend()
    plt.tight_layout()
    plt.savefig("results/exp1b.png", dpi=120)
    print("\nsaved results/exp1b.png")
