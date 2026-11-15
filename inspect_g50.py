import sys
import networkx as nx
from search import haversine

G = nx.read_gml(sys.argv[1], label="id")
print("nodes:", G.number_of_nodes(), "| edges:", G.number_of_edges(),
      "| directed:", G.is_directed())
n0 = next(iter(G.nodes(data=True)))
print("first node:", n0)
print("first edge:", next(iter(G.edges(data=True))))

def pick(d, names):
    for k in d:
        if k.lower() in names:
            return k

lat = pick(n0[1], {"latitude", "lat"})
lon = pick(n0[1], {"longitude", "lon", "long"})
if lat and lon:
    ls = [haversine((G.nodes[u][lat], G.nodes[u][lon]),
                    (G.nodes[v][lat], G.nodes[v][lon])) for u, v in G.edges()]
    print(f"computed link length km: min {min(ls):.2f} avg {sum(ls)/len(ls):.2f} max {max(ls):.2f}")
    print("TopoHub page says:       min 25.94 avg 100.71 max 252.3")
else:
    print("no latitude/longitude keys found")
