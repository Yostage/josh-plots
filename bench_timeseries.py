import numpy as np
import matplotlib.pyplot as plt
from colors import CONTROL, CONTROL_DARK, TREATMENT

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
ncols = 3
nrows = (len(lengths) + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 2.4), sharex=True)
axes = axes.flatten()

for i, L in enumerate(lengths):
    ax = axes[i]
    n128 = len(d128[L])
    n256 = len(d256[L])
    x128 = np.arange(n128)
    x256 = np.arange(n256)

    ax.scatter(x128, d128[L], s=12, alpha=0.5, color=CONTROL, label='128-bit', zorder=2)
    ax.scatter(x256, d256[L], s=12, alpha=0.5, color=TREATMENT, label='256-bit', zorder=2)

    # Rolling mean (window=10) to show trend
    w = 10
    if n128 >= w:
        rm128 = np.convolve(d128[L], np.ones(w)/w, mode='valid')
        ax.plot(np.arange(w-1, n128), rm128, color=CONTROL_DARK, linewidth=1.5, zorder=3)
    if n256 >= w:
        rm256 = np.convolve(d256[L], np.ones(w)/w, mode='valid')
        ax.plot(np.arange(w-1, n256), rm256, color=TREATMENT, linewidth=1.5, zorder=3)

    ax.set_title(f"Length {L:,}", fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.2)
    if i == 0:
        ax.legend(fontsize=7, loc='upper right')

# Label shared axes
for ax in axes[-(ncols):]:
    ax.set_xlabel("Sample #", fontsize=9)
for ax in axes[::ncols]:
    ax.set_ylabel("Time", fontsize=9)

# Hide unused subplots
for j in range(len(lengths), len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Sample-by-sample time series: 128-bit vs 256-bit loads\n"
             "(dots = raw samples, line = rolling mean window=10)",
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("bench_timeseries.png", dpi=150, bbox_inches='tight')
print("Saved bench_timeseries.png")
