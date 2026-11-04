import numpy as np
import gzip
from topology import load_router_order

order = load_router_order()

M = np.loadtxt(gzip.open("data/X06.gz"))
tm = M.reshape(2016, 144, 5)[:, :, 0].reshape(2016, 12, 12)

print("min:", tm.min(), " max:", tm.max())
print("NaN count:", int(np.isnan(tm).sum()))
print("share of zeros:", round(float((tm == 0).mean()), 3))

tot = tm.sum(axis=0)
print("diagonal total:", float(np.trace(tot)))
print()
print("idx router    sent         received")
for i, name in enumerate(order):
    print(f"{i:2d}  {name:8s}  {tot[i].sum():.3e}  {tot[:, i].sum():.3e}")
