import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

d = np.load("data/processed/hourly.npz")
H = d["hourly"]
mask = ~np.eye(12, dtype=bool)
X = np.log1p(H[:, mask])                 # (840, 132)
T, P = X.shape

TRAIN, VAL, TEST = range(168, 624), range(624, 672), range(672, 840)
pair_mean = X[168:624].mean(axis=0)      # TRAIN only

def build(ts):
    F, Y, base = [], [], []
    for t in ts:
        hist = X[t - 24:t].T
        week = X[t - 168][:, None]
        hour, day = t % 24, (t // 24) % 7
        tf = np.array([np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
                       np.sin(2*np.pi*day/7),   np.cos(2*np.pi*day/7)])
        F.append(np.hstack([hist, week, np.tile(tf, (P, 1)), pair_mean[:, None]]))
        Y.append(X[t] - X[t - 1])
        base.append(X[t - 1])
    return np.vstack(F), np.concatenate(Y), np.array(base)

def hit_per_hour(pred, true, k):
    return np.array([len(set(np.argsort(-p)[:k]) & set(np.argsort(-t)[:k])) / k
                     for p, t in zip(pred, true)])

Ftr, Ytr, _ = build(TRAIN)
Fva, _, Bva = build(VAL)
Fte, _, Bte = build(TEST)
sc = StandardScaler().fit(Ftr)
Ftr, Fva, Fte = sc.transform(Ftr), sc.transform(Fva), sc.transform(Fte)
true_va, true_te = X[list(VAL)], X[list(TEST)]

print("last hour on VALIDATION: K=10 %.3f | MAE %.4f" %
      (hit_per_hour(Bva, true_va, 10).mean(), np.abs(Bva - true_va).mean()))
print()

results = {}
for iters in (60, 150, 300):
    val_mae, te_preds = [], []
    for seed in range(5):
        nn = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=iters, random_state=seed)
        nn.fit(Ftr, Ytr)
        pv = Bva + nn.predict(Fva).reshape(len(VAL), P)
        val_mae.append(np.abs(pv - true_va).mean())
        te_preds.append(Bte + nn.predict(Fte).reshape(len(TEST), P))
    results[iters] = (np.mean(val_mae), te_preds)
    print(f"max_iter={iters:3d}: validation MAE {np.mean(val_mae):.4f}")

best = min(results, key=lambda i: results[i][0])
print(f"\nchosen by validation: max_iter={best}")
te_preds = results[best][1]

print("\n--- TEST (week X10) ---")
for k in (10, 20, 40):
    nn_h = np.mean([hit_per_hour(p, true_te, k) for p in te_preds], axis=0)  # per hour, mean over seeds
    base_h = hit_per_hour(Bte, true_te, k)
    diff = nn_h - base_h
    rng = np.random.default_rng(42)
    blocks = diff[:168].reshape(28, 6)                # blocks of 6 hours
    boots = [blocks[rng.integers(0, 28, 28)].mean() for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"K={k:2d}: NN {nn_h.mean():.3f} | last hour {base_h.mean():.3f} | "
          f"diff {diff.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}] | "
          f"hours NN better/equal/worse: {(diff>1e-9).sum()}/{(abs(diff)<=1e-9).sum()}/{(diff<-1e-9).sum()}")
