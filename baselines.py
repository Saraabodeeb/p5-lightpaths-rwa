import numpy as np

d = np.load("data/processed/hourly.npz")
H = d["hourly"]                      # (840, 12, 12)

# keep the 132 pairs (remove the diagonal)
mask = ~np.eye(12, dtype=bool)
X = H[:, mask]                       # (840, 132)

# split (hours): 0-167 history only | 168-671 train | 672-839 test (week X10)
TEST_START, TEST_END = 672, 840
true = X[TEST_START:TEST_END]                    # (168, 132)

def topk_hit(pred, true, k):
    """Average share of the true top-k pairs that appear in the predicted top-k."""
    hits = []
    for p, t in zip(pred, true):
        a = set(np.argsort(-p)[:k])
        b = set(np.argsort(-t)[:k])
        hits.append(len(a & b) / k)
    return float(np.mean(hits))

rng = np.random.default_rng(42)
preds = {
    "random":    rng.random(true.shape),
    "last hour": X[TEST_START - 1:TEST_END - 1],
    "last week": X[TEST_START - 168:TEST_END - 168],
    "static":    np.tile(X[168:TEST_START].mean(axis=0), (true.shape[0], 1)),
}

Ks = [10, 20, 40]
print(f"{'method':12s}" + "".join(f"  K={k:<3d}" for k in Ks))
for name, p in preds.items():
    print(f"{name:12s}" + "".join(f"  {topk_hit(p, true, k):.3f}" for k in Ks))
