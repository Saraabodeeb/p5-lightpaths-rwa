import numpy as np
from forecast import nn_predictions, last_hour_predictions, true_values
from integration import plan

SETTINGS = [(4, 20), (3, 15)]            # (W, K)
true = true_values()
nn = nn_predictions()                    # (5, 168, 132)
last = last_hour_predictions()


def per_hour(pred, K, W):
    served, traffic = [], []
    for h in range(168):
        top = set(int(i) for i in np.argsort(-true[h])[:K])
        res = plan(pred[h], K, W)
        got = set(i for i, _, _ in res["served"]) & top
        tt = np.expm1(true[h])
        served.append(len(got) / K)
        traffic.append(tt[list(got)].sum() / tt[list(top)].sum())
    return np.array(served), np.array(traffic)


def ci(diff):
    rng = np.random.default_rng(42)
    blocks = diff.reshape(28, 6)         # blocks of 6 hours
    boots = [blocks[rng.integers(0, 28, 28)].mean() for _ in range(5000)]
    return np.percentile(boots, [2.5, 97.5])


for W, K in SETTINGS:
    print(f"\n=== W={W}, K={K}, all 168 test hours ===")
    s_last, t_last = per_hour(last, K, W)
    s_true, t_true = per_hour(true, K, W)
    runs = [per_hour(p, K, W) for p in nn]
    s_nn = np.mean([r[0] for r in runs], axis=0)       # per hour, mean over seeds
    t_nn = np.mean([r[1] for r in runs], axis=0)
    print(f"true top-K served: last {s_last.mean():.3f} | NN {s_nn.mean():.3f} | ceiling {s_true.mean():.3f}")
    print(f"traffic share:     last {t_last.mean():.3f} | NN {t_nn.mean():.3f} | ceiling {t_true.mean():.3f}")
    for label, d in (("served NN - last", s_nn - s_last), ("traffic NN - last", t_nn - t_last)):
        lo, hi = ci(d)
        print(f"{label}: {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
