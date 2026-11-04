import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load("data/processed/hourly.npz")
hourly = d["hourly"]                      # (840, 12, 12)

total = hourly.sum(axis=(1, 2))           # total traffic per hour
plt.figure(figsize=(14, 4))
plt.plot(total)
for k in range(1, 5):
    plt.axvline(168 * k, color="gray", linestyle="--")   # week boundaries
plt.xlabel("hour (0..839)")
plt.ylabel("total traffic (relative units)")
plt.title("Abilene total traffic, weeks X06..X10")
plt.tight_layout()
plt.savefig("total_traffic.png", dpi=120)
print("saved total_traffic.png")
