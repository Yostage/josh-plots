import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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

d128 = parse_bench("128bit.csv")  # control
d256 = parse_bench("256bit.csv")  # treatment

lengths = sorted(d128.keys())
alpha = 0.05

# Compute % difference from control and CI for the difference
labels = []
pct_diffs = []
pct_ci_lo = []
pct_ci_hi = []
p_values = []

for L in lengths:
    ctrl, treat = d128[L], d256[L]
    mc, mt = np.mean(ctrl), np.mean(treat)
    pct = (mt - mc) / mc * 100

    # Welch CI on the raw difference, then convert to %
    se = np.sqrt(stats.sem(ctrl)**2 + stats.sem(treat)**2)
    df_w = (stats.sem(ctrl)**2 + stats.sem(treat)**2)**2 / (
        stats.sem(ctrl)**4 / (len(ctrl)-1) + stats.sem(treat)**4 / (len(treat)-1))
    t_crit = stats.t.ppf(1 - alpha/2, df_w)
    diff_lo = (mt - mc) - t_crit * se
    diff_hi = (mt - mc) + t_crit * se

    pct_diffs.append(pct)
    pct_ci_lo.append(diff_lo / mc * 100)
    pct_ci_hi.append(diff_hi / mc * 100)

    _, p = stats.ttest_ind(ctrl, treat, equal_var=False)
    p_values.append(p)
    labels.append(f"Length {L:,}")

pct_diffs = np.array(pct_diffs)
pct_ci_lo = np.array(pct_ci_lo)
pct_ci_hi = np.array(pct_ci_hi)
p_values = np.array(p_values)
significant = p_values < alpha

# --- Forest-style A/B plot ---
fig, ax = plt.subplots(figsize=(10, 8))

y = np.arange(len(lengths))[::-1]  # top-to-bottom, smallest length on top

for i in range(len(lengths)):
    color = '#e74c3c' if significant[i] and pct_diffs[i] > 0 else \
            '#27ae60' if significant[i] and pct_diffs[i] < 0 else \
            '#7f8c8d'
    ax.plot([pct_ci_lo[i], pct_ci_hi[i]], [y[i], y[i]], color=color, linewidth=2.5, solid_capstyle='round')
    ax.plot(pct_diffs[i], y[i], 'o', color=color, markersize=8, zorder=5)

    # Annotate with value and significance
    sig_str = " *" if significant[i] else ""
    ax.text(max(pct_ci_hi[i], 0) + 0.8, y[i],
            f"{pct_diffs[i]:+.1f}%{sig_str}",
            va='center', fontsize=9, color=color, fontweight='bold')

ax.axvline(0, color='black', linewidth=1, linestyle='-', zorder=0)
ax.axvspan(-0.5, 0.5, color='#ecf0f1', alpha=0.5, zorder=0)  # "noise band"

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel("% Change from Control (128-bit loads)", fontsize=11)
ax.set_title("A/B Comparison: 256-bit vs 128-bit loads (AVX/FMA f32)\n"
             "Control = 128-bit  |  Treatment = 256-bit  |  Lower is faster",
             fontsize=12, fontweight='bold')

ax.grid(axis='x', alpha=0.3)
ax.set_axisbelow(True)

# Legend
from matplotlib.lines import Line2D
legend_els = [
    Line2D([0], [0], color='#e74c3c', marker='o', linewidth=2.5, label='Significantly slower (p<0.05)'),
    Line2D([0], [0], color='#27ae60', marker='o', linewidth=2.5, label='Significantly faster (p<0.05)'),
    Line2D([0], [0], color='#7f8c8d', marker='o', linewidth=2.5, label='No significant difference'),
]
ax.legend(handles=legend_els, loc='lower right', fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig("bench_comparison.png", dpi=150)
print("Saved bench_comparison.png")

# Print summary table
print(f"\n{'Metric':<16} {'Control':>8} {'Treatment':>10} {'% Diff':>8} {'95% CI':>20} {'p':>8} {'Sig':>4}")
print("-" * 80)
for i, L in enumerate(lengths):
    mc = np.mean(d128[L])
    mt = np.mean(d256[L])
    print(f"Length {L:<8,} {mc:>8.1f} {mt:>10.1f} {pct_diffs[i]:>+7.1f}% [{pct_ci_lo[i]:>+6.1f}%, {pct_ci_hi[i]:>+5.1f}%] {p_values[i]:>8.4f} {'*' if significant[i] else ''}")
