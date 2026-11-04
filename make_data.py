"""Downloads the Abilene files (if missing) and builds the hourly traffic dataset (weeks X06..X10)."""
import gzip
import os
import urllib.request
import numpy as np
from topology import load_router_order

SEED = 42                      # this step has no randomness; seed kept for reproducibility of later steps
np.random.seed(SEED)
BASE = "https://www.cs.utexas.edu/~yzhang/research/AbileneTM"
WEEKS = ["X06", "X07", "X08", "X09", "X10"]     # 2004-05-01 .. 2004-05-29, consecutive weeks
REMOTE_TO_LOCAL = {f"{w}.gz": f"{w}.gz" for w in WEEKS}
REMOTE_TO_LOCAL.update({"A": "A.txt", "links": "links.txt", "demands": "demands.txt",
                        "topo-2003-04-10.txt": "topo-2003-04-10.txt"})


def download():
    os.makedirs("data/processed", exist_ok=True)
    for remote, local in REMOTE_TO_LOCAL.items():
        path = f"data/{local}"
        if os.path.exists(path) and os.path.getsize(path) > 0:
            continue
        print("downloading", remote)
        urllib.request.urlretrieve(f"{BASE}/{remote}", path)


download()
order = load_router_order()
weeks_hourly = []
for w in WEEKS:
    M = np.loadtxt(gzip.open(f"data/{w}.gz"))
    assert M.shape == (2016, 720), M.shape
    real = M.reshape(2016, 144, 5)[:, :, 0].reshape(2016, 12, 12)   # real measurements only
    hourly = real.reshape(168, 12, 12, 12).sum(axis=1)               # 12 x 5 min -> 1 hour
    weeks_hourly.append(hourly)
    print(w, "hourly shape:", hourly.shape,
          "| hours with zero total traffic:", int((hourly.sum(axis=(1, 2)) == 0).sum()))

hourly = np.concatenate(weeks_hourly, axis=0)
print("final shape:", hourly.shape)
np.savez_compressed("data/processed/hourly.npz", hourly=hourly,
                    routers=np.array(order), weeks=np.array(WEEKS))
print("saved data/processed/hourly.npz")
