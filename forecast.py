"""NN forecaster as a module: next-hour predictions (log scale) for the test week."""
import os
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

H = np.load("data/processed/hourly.npz")["hourly"]
X = np.log1p(H[:, ~np.eye(12, dtype=bool)])          # (840, 132)
T, P = X.shape
TRAIN, TEST = range(168, 624), range(672, 840)       # same split as before
pair_mean = X[168:624].mean(axis=0)                  # TRAIN only
CACHE = "data/processed/nn_test_preds.npy"


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


def nn_predictions(n_seeds=5, max_iter=60):
    """Array (n_seeds, 168, 132). max_iter=60 was chosen on validation. Cached on disk."""
    if os.path.exists(CACHE):
        return np.load(CACHE)
    Ftr, Ytr, _ = build(TRAIN)
    Fte, _, Bte = build(TEST)
    sc = StandardScaler().fit(Ftr)
    Ftr, Fte = sc.transform(Ftr), sc.transform(Fte)
    out = []
    for seed in range(n_seeds):
        nn = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=max_iter, random_state=seed)
        nn.fit(Ftr, Ytr)
        out.append(Bte + nn.predict(Fte).reshape(len(TEST), P))
        print("trained NN seed", seed)
    out = np.array(out)
    np.save(CACHE, out)
    return out


def last_hour_predictions():
    return X[[t - 1 for t in TEST]]                  # (168, 132)


def true_values():
    return X[list(TEST)]                             # (168, 132)
