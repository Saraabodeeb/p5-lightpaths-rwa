def load_topology(path="data/topo-2003-04-10.txt"):
    routers, links = {}, []
    section = None
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line in ("router", "link"):
            section = line
            continue
        parts = line.split()
        if section == "router":
            name, city, lat, lon = parts
            routers[name] = (city, float(lat), float(lon))
        else:
            x, y, cap, ospf = parts
            links.append((x, y, int(cap), int(ospf)))
    return routers, links


def load_router_order(path="data/demands.txt"):
    order = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        src = line.split()[0].split(",")[0]
        if src not in order:
            order.append(src)
    return order


if __name__ == "__main__":
    routers, links = load_topology()
    order = load_router_order()
    print("routers:", len(routers))
    print("directed links:", len(links))
    print("order:", order)
