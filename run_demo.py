import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from forecast import nn_predictions, last_hour_predictions, true_values
from integration import plan, PAIRS, ADJ

HOUR, K, W = 40, 20, 4          # hour index inside the test week (0..167)
true = true_values()
nn_pred = nn_predictions()[0]   # NN seed 0 (demo only; the experiments use 5 seeds)
last = last_hour_predictions()

short = lambda n: n.replace("ng", "")
links = sorted((u, v) for u in ADJ for v, _ in ADJ[u])
idx = {l: i for i, l in enumerate(links)}

res = plan(nn_pred[HOUR], K, W)
print(f"INPUT : test-week hour {HOUR}, top-{K} flows predicted by the NN, {W} wavelengths")
print(f"\nSOLUTION: {len(res['served'])} served, {len(res['blocked'])} blocked "
      f"(CSP rounds: {res['loops']}, assignments: {res['assignments']})")
for i, path, w in res["served"]:
    print(f"  {short(PAIRS[i][0]):>5} -> {short(PAIRS[i][1]):<5} wavelength {w}  "
          f"{' > '.join(short(p) for p in path)}")
for i, why in res["blocked"]:
    print(f"  BLOCKED {short(PAIRS[i][0])} -> {short(PAIRS[i][1])}: {why}")

# heatmap: link x wavelength
usage = np.zeros((len(links), W))
for i, path, w in res["served"]:
    for a, b in zip(path, path[1:]):
        assert usage[idx[(a, b)], w] == 0, "wavelength clash: the CSP is broken"
        usage[idx[(a, b)], w] = 1

fig, ax = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={"width_ratios": [1, 1.4]})
ax[0].imshow(usage, aspect="auto", cmap="Blues")
ax[0].set_yticks(range(len(links)))
ax[0].set_yticklabels([f"{short(a)}>{short(b)}" for a, b in links], fontsize=7)
ax[0].set_xticks(range(W))
ax[0].set_xlabel("wavelength")
ax[0].set_title("Wavelength use per directed link")

order = [i for i, _, _ in res["served"]] + [i for i, _ in res["blocked"]]
tt = np.expm1(true[HOUR])
colors = ["tab:green"] * len(res["served"]) + ["tab:red"] * len(res["blocked"])
ax[1].bar(range(len(order)), tt[order], color=colors)
ax[1].set_xticks(range(len(order)))
ax[1].set_xticklabels([f"{short(PAIRS[i][0])}>{short(PAIRS[i][1])}" for i in order],
                      rotation=90, fontsize=7)
ax[1].set_ylabel("real traffic this hour (relative units)")
ax[1].set_title("Requests chosen by the NN: green = served, red = blocked")
plt.tight_layout()
plt.savefig("demo_output.png", dpi=120)
print("\nsaved demo_output.png")

# system with the NN vs system with last hour, same hour
top_true = set(int(i) for i in np.argsort(-true[HOUR])[:K])
print("\nNN ON vs OFF (OFF = last-hour forecast), same hour:")
for name, p in (("NN on ", nn_pred[HOUR]), ("NN off", last[HOUR])):
    r = plan(p, K, W)
    got = set(i for i, _, _ in r["served"]) & top_true
    print(f"  {name}: true top-{K} served = {len(got)}/{K}, blocked = {len(r['blocked'])}")
