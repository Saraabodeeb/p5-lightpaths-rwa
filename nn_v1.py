import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

d = np.load("data/processed/hourly.npz")
H = d["hourly"]                          # (840, 12, 12)
mask = ~np.eye(12, dtype=bool)
X = np.log1p(H[:, mask])                 # (840, 132), log scale
T, P = X.shape

# hours: 0-167 history only | 168-623 train | 624-671 validation | 672-839 test
TRAIN = range(168, 624)
VAL = range(624, 672)
TEST = range(672, 840)

pair_mean = X[168:624].mean(axis=0)      # from TRAIN hours only (no leakage)

def build(ts):
    F, Y, base = [], [], []
    for t in ts:
        hist = X[t - 24:t].T                       # (132, 24) last 24 hours
        week = X[t - 168][:, None]                 # (132, 1) same hour last week
        hour, day = t % 24, (t // 24) % 7
        tf = np.array([np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
                       np.sin(2*np.pi*day/7),   np.cos(2*np.pi*day/7)])
        feats = np.hstack([hist, week, np.tile(tf, (P, 1)), pair_mean[:, None]])
        F.append(feats)
        Y.append(X[t] - X[t - 1])                  # learn the CHANGE from last hour
        base.append(X[t - 1])
    return np.vstack(F), np.concatenate(Y), np.array(base)

def topk_hit(pred, true, k):
    hits = []
    for p, t in zip(pred, true):
        hits.append(len(set(np.argsort(-p)[:k]) & set(np.argsort(-t)[:k])) / k)
    return float(np.mean(hits))

Ftr, Ytr, _ = build(TRAIN)
Fva, Yva, Bva = build(VAL)
Fte, Yte, Bte = build(TEST)

sc = StandardScaler().fit(Ftr)           # statistics from TRAIN only
Ftr, Fva, Fte = sc.transform(Ftr), sc.transform(Fva), sc.transform(Fte)

true_te = X[list(TEST)]
true_va = X[list(VAL)]
Ks = [10, 20, 40]

print("last hour (test)   " + "  ".join(f"K={k}: {topk_hit(Bte, true_te, k):.3f}" for k in Ks))
print("last hour MAE (log): %.4f" % np.abs(Bte - true_te).mean())
print()

res = {k: [] for k in Ks}
maes = []
for seed in range(5):
    nn = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=60,
                      random_state=seed)
    nn.fit(Ftr, Ytr)
    pred_te = Bte + nn.predict(Fte).reshape(len(TEST), P)
    pred_va = Bva + nn.predict(Fva).reshape(len(VAL), P)
    for k in Ks:
        res[k].append(topk_hit(pred_te, true_te, k))
    maes.append(np.abs(pred_te - true_te).mean())
    print(f"seed {seed}: val K=10 {topk_hit(pred_va, true_va, 10):.3f} | "
          f"test K=10 {res[10][-1]:.3f}  MAE {maes[-1]:.4f}")

print()
for k in Ks:
    print(f"NN test K={k}: {np.mean(res[k]):.3f} +- {np.std(res[k]):.3f}")
print(f"NN test MAE (log): {np.mean(maes):.4f} +- {np.std(maes):.4f}")
