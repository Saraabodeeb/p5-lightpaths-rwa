"""Experiment 3 (final): NN vs last-hour vs ceiling INSIDE the full system."""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from forecast import nn_predictions, last_hour_predictions, true_values
from integration import plan

SETTINGS = [(4, 20), (3, 15)]            # (W, K)
os.makedirs("results", exist_ok=True)
true = true_values()
nn = nn_predictions()                    # (5 seeds, 168, 132)
last = last_hour_predictions()


def per_hour(pred, K, W):
    served, traffic, blocked = [], [], []
    for h in range(168):
        top = set(int(i) for i in np.argsort(-true[h])[:K])
        res = plan(pred[h], K, W)
        got = set(i for i, _, _ in res["served"]) & top
        tt = np.expm1(true[h])
        served.append(len(got) / K)
        traffic.append(tt[list(got)].sum() / tt[list(top)].sum())
        blocked.append(len(res["blocked"]) / K)
    return np.array(served), np.array(traffic), np.array(blocked)


rows, data = [], {}
for W, K in SETTINGS:
    runs = {"last hour": [last], "NN": list(nn), "ceiling (true traffic)": [true]}
    for name, preds in runs.items():
        out = []
        for r, p in enumerate(preds):
            s, t, b = per_hour(p, K, W)
            out.append((s, t, b))
            for h in range(168):
                rows.append(dict(W=W, K=K, method=name, run=r, hour=h,
                                 served=round(float(s[h]), 4),
                                 traffic=round(float(t[h]), 4),
                                 blocked=round(float(b[h]), 4)))
        data[(W, K, name)] = out
        print("done W", W, "K", K, name, flush=True)

with open("results/exp3_raw.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
print("saved results/exp3_raw.csv")


def agg(out, j):
    v = np.array([o[j].mean() for o in out])      # one mean per run (NN: one per seed)
    return v.mean(), v.std()


def ci(diff):
    rng = np.random.default_rng(42)
    blocks = diff.reshape(28, 6)                  # blocks of 6 hours
    boots = [blocks[rng.integers(0, 28, 28)].mean() for _ in range(5000)]
    return np.percentile(boots, [2.5, 97.5])


NAMES = ("last hour", "NN", "ceiling (true traffic)")
print("\nEXPERIMENT 3: all 168 test hours; NN = mean+-std over 5 seeds (others are deterministic)")
for W, K in SETTINGS:
    print(f"\n=== W={W}, K={K} ===")
    print(f"{'method':24s} {'true top-K served':>19s} {'traffic share':>16s} {'blocked share':>16s}")
    for name in NAMES:
        out = data[(W, K, name)]
        cells = [agg(out, j) for j in range(3)]
        print(f"{name:24s} " + " ".join(f"{m:>9.3f}+-{s:<6.3f}" for m, s in cells))
    for j, label in ((0, "served"), (1, "traffic share")):
        nn_h = np.mean([o[j] for o in data[(W, K, "NN")]], axis=0)
        d = nn_h - data[(W, K, "last hour")][0][j]
        lo, hi = ci(d)
        print(f"NN - last hour, {label}: {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")

fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for a, (W, K) in zip(ax, SETTINGS):
    for m, (j, lab) in enumerate(((0, "true top-K served"), (1, "traffic share"))):
        mu = [agg(data[(W, K, n)], j)[0] for n in NAMES]
        sd = [agg(data[(W, K, n)], j)[1] for n in NAMES]
        a.bar(np.arange(3) + 0.4 * m - 0.2, mu, 0.4, yerr=sd, capsize=3, label=lab)
    a.set_xticks(range(3))
    a.set_xticklabels(["last hour", "NN", "ceiling"])
    a.set_title(f"W={W}, K={K} (168 test hours)")
    a.set_ylim(0, 1.0)
ax[0].set_ylabel("share (NN: mean +- std over 5 seeds)")
ax[0].legend(loc="lower left")
plt.tight_layout()
plt.savefig("results/exp3.png", dpi=120)
print("saved results/exp3.png")
