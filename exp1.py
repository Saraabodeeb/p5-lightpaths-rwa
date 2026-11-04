"""Experiment 1: BFS vs UCS vs A* on Abilene (12 nodes) and germany50 (50 nodes)."""
import itertools
import random
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from search import build_graph, k_paths, haversine

METHODS = ("bfs", "ucs", "astar")
SEEDS = range(5)


def build_graph_gml(path):
    G = nx.read_gml(path, label="id")
    names = {n: d["label"] for n, d in G.nodes(data=True)}
    coords = {names[n]: (d["lat"], d["lon"]) for n, d in G.nodes(data=True)}
    assert len(coords) == G.number_of_nodes(), "duplicate city names"
    adj = {names[n]: [] for n in G.nodes}
    for u, v in G.edges():                        # undirected cable = 2 directed links
        w = haversine(coords[names[u]], coords[names[v]])
        adj[names[u]].append((names[v], w))
        adj[names[v]].append((names[u], w))
    return adj, coords


def to_nx(adj):
    G = nx.DiGraph()
    for u, nbrs in adj.items():
        for v, w in nbrs:
            G.add_edge(u, v, weight=w)
    return G


def validate(name, adj, coords, n_k3=150):
    G = to_nx(adj)
    pairs = list(itertools.permutations(list(adj), 2))
    bad_h = bad_c = 0
    for s, t in pairs:
        best = nx.shortest_path_length(G, s, t, weight="weight")
        if haversine(coords[s], coords[t]) > best + 1e-9:
            bad_h += 1
        for m in ("ucs", "astar"):
            found, _, _ = k_paths(adj, coords, s, t, k=1, method=m)
            if abs(found[0][0] - best) > 1e-6:
                bad_c += 1
    bad3 = 0
    sample = random.Random(0).sample(pairs, min(n_k3, len(pairs)))
    for s, t in sample:
        ref = []
        for p in nx.shortest_simple_paths(G, s, t, weight="weight"):
            ref.append(sum(G[a][b]["weight"] for a, b in zip(p, p[1:])))
            if len(ref) == 3:
                break
        for m in ("ucs", "astar"):
            costs = [c for c, _ in k_paths(adj, coords, s, t, k=3, method=m)[0]]
            if len(costs) != len(ref) or any(abs(a - b) > 1e-6 for a, b in zip(costs, ref)):
                bad3 += 1
    print(f"[{name}] heuristic overestimates: {bad_h}/{len(pairs)} pairs | "
          f"k=1 cost mismatches vs networkx: {bad_c} | "
          f"k=3 mismatches on {len(sample)} pairs: {bad3}")


def run(adj, coords, seed, n_pairs=30):
    rng = random.Random(seed)
    pairs = rng.sample(list(itertools.permutations(list(adj), 2)), n_pairs)
    res = {}
    for k in (1, 3):
        for m in METHODS:
            vals = []
            for s, t in pairs:
                found, e, g = k_paths(adj, coords, s, t, k=k, method=m)
                vals.append((e, g, found[0][0], len(found[0][1]) - 1))
            res[(k, m)] = np.mean(vals, axis=0)      # mean over the 30 pairs
    return res


if __name__ == "__main__":
    nets = {"Abilene (12 nodes)": build_graph(),
            "germany50 (50 nodes)": build_graph_gml("data/germany50.gml")}

    print("VALIDATION")
    for name, (adj, coords) in nets.items():
        validate(name, adj, coords)

    rows = {}
    for name, (adj, coords) in nets.items():
        per_seed = [run(adj, coords, s) for s in SEEDS]
        for k in (1, 3):
            for m in METHODS:
                arr = np.array([ps[(k, m)] for ps in per_seed])    # (5 seeds, 4 metrics)
                rows[(name, k, m)] = (arr.mean(axis=0), arr.std(axis=0))

    print("\nEXPERIMENT 1 (30 random pairs per seed, 5 seeds, mean +- std over seeds)")
    print(f"{'network':22s} {'k':>1} {'method':6s} {'expanded':>16s} {'generated':>16s} "
          f"{'1st path km':>14s} {'1st path hops':>14s}")
    for (name, k, m), (mu, sd) in rows.items():
        print(f"{name:22s} {k:>1} {m:6s} {mu[0]:>8.1f}+-{sd[0]:<6.1f} {mu[1]:>8.1f}+-{sd[1]:<6.1f} "
              f"{mu[2]:>8.1f}+-{sd[2]:<4.1f} {mu[3]:>8.2f}+-{sd[3]:<4.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, name in zip(axes, nets):
        for j, k in enumerate((1, 3)):
            mu = [rows[(name, k, m)][0][0] for m in METHODS]
            sd = [rows[(name, k, m)][1][0] for m in METHODS]
            ax.bar(np.arange(3) + 0.4 * j - 0.2, mu, 0.4, yerr=sd, capsize=3, label=f"k={k}")
        ax.set_xticks(range(3))
        ax.set_xticklabels([m.upper() for m in METHODS])
        ax.set_title(name)
        ax.set_ylabel("nodes expanded (mean +- std, 5 seeds)")
        ax.legend()
    plt.tight_layout()
    plt.savefig("exp1.png", dpi=120)
    print("\nsaved exp1.png")
