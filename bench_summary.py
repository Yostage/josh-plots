import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from colors import CONTROL, TREATMENT

def parse_bench(path):
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("Length"):
                continue
            parts = [p.strip() for p in line.split(",")]
            length = int(parts[0].split()[-1])
            samples = np.array([int(x) for x in parts[1:]])
            data[length] = samples
    return data

d128 = parse_bench("128bit.csv")
d256 = parse_bench("256bit.csv")

lengths = sorted(d128.keys())

# --- Summary table ---
header = f"{'Length':>10} {'Treatment':>10} {'P25':>6} {'P50':>6} {'P75':>6} {'P90':>6} {'Mean':>7} {'Std':>6} {'IQR':>6}"
print(header)
print("-" * len(header))

for L in lengths:
    for label, samples in [("128-bit", d128[L]), ("256-bit", d256[L])]:
        p25, p50, p75, p90 = np.percentile(samples, [25, 50, 75, 90])
        print(f"{L:>10} {label:>10} {p25:>6.1f} {p50:>6.1f} {p75:>6.1f} {p90:>6.1f} {np.mean(samples):>7.1f} {np.std(samples, ddof=1):>6.1f} {p75-p25:>6.1f}")
    print()

# --- ECDF plot: one subplot per length, both treatments overlaid ---
ncols = 3
nrows = (len(lengths) + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 2.6))
axes = axes.flatten()

for i, L in enumerate(lengths):
    ax = axes[i]
    for label, samples, color in [("128-bit", d128[L], CONTROL), ("256-bit", d256[L], TREATMENT)]:
        xs = np.sort(samples)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.step(xs, ys, where='post', color=color, linewidth=1.5, label=label)

    ax.set_title(f"Length {L:,}", fontsize=10, fontweight='bold')
    ax.set_ylabel("CDF", fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, 1.05)
    if i == 0:
        ax.legend(fontsize=7, loc='lower right')

for ax in axes[-ncols:]:
    ax.set_xlabel("Time", fontsize=9)

for j in range(len(lengths), len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Empirical CDF: 128-bit vs 256-bit loads",
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("bench_ecdf.png", dpi=150, bbox_inches='tight')
print("Saved bench_ecdf.png")

# --- KDE plot: one subplot per length, both treatments overlaid ---
fig2, axes2 = plt.subplots(nrows, ncols, figsize=(14, nrows * 2.6))
axes2 = axes2.flatten()

for i, L in enumerate(lengths):
    ax = axes2[i]
    for label, samples, color in [("128-bit", d128[L], CONTROL), ("256-bit", d256[L], TREATMENT)]:
        kde = stats.gaussian_kde(samples)
        lo, hi = samples.min() - 5, samples.max() + 5
        xs = np.linspace(lo, hi, 200)
        ax.plot(xs, kde(xs), color=color, linewidth=1.5, label=label)
        ax.fill_between(xs, kde(xs), alpha=0.15, color=color)

    ax.set_title(f"Length {L:,}", fontsize=10, fontweight='bold')
    ax.set_ylabel("Density", fontsize=8)
    ax.grid(True, alpha=0.2)
    if i == 0:
        ax.legend(fontsize=7, loc='upper right')

for ax in axes2[-ncols:]:
    ax.set_xlabel("Time", fontsize=9)

for j in range(len(lengths), len(axes2)):
    axes2[j].set_visible(False)

fig2.suptitle("KDE Density: 128-bit vs 256-bit loads",
              fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("bench_kde.png", dpi=150, bbox_inches='tight')
print("Saved bench_kde.png")
