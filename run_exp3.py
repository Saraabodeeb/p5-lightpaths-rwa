import time
import numpy as np
from forecast import nn_predictions, last_hour_predictions, true_values
from integration import plan

K, W = 20, 4
HOURS = list(range(0, 168, 6))            # every 6th hour of the test week (28 hours)

true = true_values()
runs = {
    "last hour":                  [last_hour_predictions()],
    "NN (5 seeds)":               list(nn_predictions()),
    "true traffic (upper bound)": [true],
}


def evaluate(pred):
    served_rate, block_rate, traffic_share = [], [], []
    for h in HOURS:
        top_true = set(int(i) for i in np.argsort(-true[h])[:K])
        res = plan(pred[h], K, W)
        served = set(i for i, _, _ in res["served"])
        hit = list(served & top_true)
        tt = np.expm1(true[h])
        served_rate.append(len(hit) / K)
        block_rate.append(len(res["blocked"]) / K)
        traffic_share.append(tt[hit].sum() / tt[list(top_true)].sum())
    return np.mean(served_rate), np.mean(block_rate), np.mean(traffic_share)


print(f"K={K}, W={W}, {len(HOURS)} test hours\n")
print(f"{'method':28s} {'true top-K served':>19s} {'blocking':>16s} {'traffic share':>16s}")
for name, preds in runs.items():
    t0 = time.time()
    res = np.array([evaluate(p) for p in preds])          # (n_runs, 3)
    m, s = res.mean(axis=0), res.std(axis=0)
    fmt = lambda i: f"{m[i]:.3f} +- {s[i]:.3f}"
    print(f"{name:28s} {fmt(0):>19s} {fmt(1):>16s} {fmt(2):>16s}   ({time.time()-t0:.0f}s)")
