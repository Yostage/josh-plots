import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from colors import DATA_COLORS, deemphasize_color, darken_color
from matplotlib.lines import Line2D
from scipy import stats

class Bench:
    def __init__(self, path):
        self.data = {}
        self.label = ""
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if not line.startswith("Length"):
                    assert(not self.label)
                    self.label = line
                    continue

                parts = [p.strip() for p in line.split(",")]
                length = int(parts[0].split()[-1])
                samples = np.array([int(x) for x in parts[1:]])
                self.data[length] = samples

parser = argparse.ArgumentParser(prog="bench", description="Benchmark results comparison")
parser.add_argument("control", type=str, help="Path of the control .csv file")
parser.add_argument("treatment", nargs="+", help="Path(s) to the treatment file(s)")
parser.add_argument("-o", "--output-dir", default=None, type=str, help="Directory to output images to")
parser.add_argument("-p", "--output-prefix", default="bench", type=str, help="Prefix for output image filenames")
args = parser.parse_args()

if args.output_dir:
    if args.output_dir.endswith("/") or args.output_dir.endswith("\\"):
        args.output_dir = args.output_dir[:-1]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.path.exists(args.output_dir):
        if not os.path.isdir(args.output_dir):
          print(f"Error: '{args.output_dir}' is not a directory.")
          sys.exit(1)

control = Bench(args.control)
treatments = [Bench(x) for x in args.treatment]

if len(treatments) > len(DATA_COLORS) - 1:
    print("Too many treatments! We're going to need more colors")
    sys.exit(1)

maxLabelLength = max([len(x.label) for x in [control, *treatments]])

lengths = sorted(control.data.keys())
alpha = 0.05

def welch_ci(a, b, alpha):
    """Compute mean difference and CI between two samples, handling zero-variance."""
    d = np.mean(b) - np.mean(a)
    se = np.sqrt(stats.sem(a)**2 + stats.sem(b)**2)
    if se == 0:
        return d, d, d
    denom = stats.sem(a)**4 / (len(a)-1) + stats.sem(b)**4 / (len(b)-1)
    if denom == 0:
        return d, d, d
    df_w = (stats.sem(a)**2 + stats.sem(b)**2)**2 / denom
    t_crit = stats.t.ppf(1 - alpha/2, df_w)
    return d, d - t_crit * se, d + t_crit * se
ncols = 3
nrows = (len(lengths) + ncols - 1) // ncols


def vs_label():
    other = f"'{treatments[0].label}'" if len(treatments) == 1 else "multiple"
    return f"'{control.label}' vs {other}"

def make_output_path(suffixWithExt):
    if args.output_dir:
        return f"{args.output_dir}/{args.output_prefix}_{suffixWithExt}"
    else:
        return f"{args.output_prefix}_{suffixWithExt}"

def save_fig(plt, name, dpi=300):
    path = make_output_path(name)
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"Saved {path}")

# ============================================================
# 1. Summary table
# ============================================================
header = f"{'Length':>10} { 'Treatment':>{maxLabelLength}} {'P25':>6} {'P50':>6} {'P75':>6} {'P90':>6} {'Mean':>7} {'Std':>6} {'IQR':>6}"
print(header)
print("-" * len(header))

for L in lengths:
    for label, samples in [
        (control.label, control.data[L]), 
        *((t.label, t.data[L]) for t in treatments)
    ]:
        p25, p50, p75, p90 = np.percentile(samples, [25, 50, 75, 90])
        print(f"{L:>10} {label:>{maxLabelLength}} {p25:>6.1f} {p50:>6.1f} {p75:>6.1f} {p90:>6.1f} {np.mean(samples):>7.1f} {np.std(samples, ddof=1):>6.1f} {p75-p25:>6.1f}")
    print()

# ============================================================
# 2. Simple overview (absolute times + difference)
# ============================================================
def make_simple(bench):
    means = []
    ci_lo = []
    ci_hi = []
    for L in lengths:
        data = bench.data[L]
        m = np.mean(data)
        ci = stats.t.interval(1 - alpha, len(data) - 1, loc=m, scale=stats.sem(data))
        means.append(m)
        ci_lo.append(ci[0])
        ci_hi.append(ci[1])
    return (np.array(means), np.array(ci_lo), np.array(ci_hi))

means_ctrl, ci_lo_ctrl, ci_hi_ctrl = make_simple(control)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
x = np.arange(len(lengths))
xlabels = [str(L) for L in lengths]

ax1.errorbar(x - 0.15, means_ctrl, yerr=[means_ctrl - ci_lo_ctrl, ci_hi_ctrl - means_ctrl],
             fmt='o-', capsize=4, label=control.label, color=DATA_COLORS[0])

for (treatment, color) in ((treatments[ti], DATA_COLORS[ti + 1]) for ti in range(len(treatments))):
  means_tmnt, ci_lo_tmnt, ci_hi_tmnt = make_simple(treatment)
  ax1.errorbar(x + 0.15, means_tmnt, yerr=[means_tmnt - ci_lo_tmnt, ci_hi_tmnt - means_tmnt],
              fmt='s-', capsize=4, label=treatment.label, color=color)
ax1.set_ylabel("Time (units from benchmark)")
ax1.set_title(f"Benchmark: {vs_label()}")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(x)
ax1.set_xticklabels(xlabels, rotation=45, ha='right')

for (treatment, color) in ((treatments[ti], DATA_COLORS[ti + 1]) for ti in range(len(treatments))):
    diffs, diff_ci_lo, diff_ci_hi = [], [], []
    for L in lengths:
        d, lo, hi = welch_ci(control.data[L], treatment.data[L], alpha)
        diffs.append(d)
        diff_ci_lo.append(lo)
        diff_ci_hi.append(hi)

    diffs = np.array(diffs)
    diff_ci_lo = np.array(diff_ci_lo)
    diff_ci_hi = np.array(diff_ci_hi)

    ax2.errorbar(x, diffs, yerr=[diffs - diff_ci_lo, diff_ci_hi - diffs],
                 fmt='D-', capsize=4, color=color)
ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
diff_label = "treatments" if len(treatments) > 1 else f"'{treatments[0].label}'"
ax2.set_ylabel(f"Difference ({diff_label} minus '{control.label}')")
ax2.set_xlabel("Array Length")
ax2.set_title(f"Mean Difference with 95% CI (positive = slower than control)")
ax2.grid(True, alpha=0.3)
ax2.set_xticks(x)
ax2.set_xticklabels(xlabels, rotation=45, ha='right')

plt.tight_layout()
save_fig(plt, "simple.png")

# ============================================================
# 3. A/B forest plot
# ============================================================

ab_labels, pct_diffs, pct_ci_lo, pct_ci_hi, p_values, base_colors = [], [], [], [], [], []
for L in lengths:
    for (treatmentIndex, treatment, base_color) in ((i, treatments[i], DATA_COLORS[i + 1]) for i in range(len(treatments))):
        ctrl, treat = control.data[L], treatment.data[L]
        mc, mt = np.mean(ctrl), np.mean(treat)
        d, diff_lo, diff_hi = welch_ci(ctrl, treat, alpha)

        pct_diffs.append((mt - mc) / mc * 100)
        pct_ci_lo.append(diff_lo / mc * 100)
        pct_ci_hi.append(diff_hi / mc * 100)
        _, p = stats.ttest_ind(ctrl, treat, equal_var=False)
        p_values.append(p)
        base_colors.append(base_color)
        ab_labels.append(f"Length {L}: '{treatment.label}'" if treatmentIndex == 0 else f"'{treatment.label}'")

pct_diffs = np.array(pct_diffs)
pct_ci_lo = np.array(pct_ci_lo)
pct_ci_hi = np.array(pct_ci_hi)
p_values = np.array(p_values)
significant = p_values < alpha

fig, ax = plt.subplots(figsize=(10, 8))
y = np.arange(len(ab_labels))[::-1]

for i in range(len(ab_labels)):
    color, marker = (darken_color(base_colors[i]), 'v') if significant[i] and pct_diffs[i] > 0 else \
            (base_colors[i], '^') if significant[i] and pct_diffs[i] < 0 else \
            ('#7f8c8d', 'o')
    ax.plot([pct_ci_lo[i], pct_ci_hi[i]], [y[i], y[i]], color=color, linewidth=2.5, solid_capstyle='round')
    ax.plot(pct_diffs[i], y[i], marker=marker, color=color, markersize=8, zorder=5)
    sig_str = " *" if significant[i] else ""
    ax.text(max(pct_ci_hi[i], 0) + 0.8, y[i],
            f"{pct_diffs[i]:+.1f}%{sig_str}",
            va='center', fontsize=9, color=color, fontweight='bold')

ax.axvline(0, color='black', linewidth=1, linestyle='-', zorder=0)
ax.axvspan(-0.5, 0.5, color='#ecf0f1', alpha=0.5, zorder=0)
ax.set_yticks(y)
ax.set_yticklabels(ab_labels, fontsize=10)
ax.set_ylabel("Name (Length)")
ax.set_xlabel(f"% Change from Control ({control.label})", fontsize=11)
ax.set_title(f"A/B Comparison\n"
             "Lower is faster",
             fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.set_axisbelow(True)
ax.legend(handles=[
    Line2D([0], [0], color=darken_color(DATA_COLORS[1]), marker='v', linewidth=2.5, label='Significantly slower (p<0.05)'),
    Line2D([0], [0], color=DATA_COLORS[1], marker='^', linewidth=2.5, label='Significantly faster (p<0.05)'),
    Line2D([0], [0], color=DATA_COLORS[0], marker='o', linewidth=2.5, label='No significant difference'),
], loc='lower right', fontsize=9, framealpha=0.9)

plt.tight_layout()
save_fig(plt, "comparison.png")

print(f"\n{'Metric':<16} {'Control':>8} {'Treatment':>10} {'% Diff':>8} {'95% CI':>20} {'p':>8} {'Sig':>4}")
print("-" * 80)
idx = 0
for L in lengths:
    mc = np.mean(control.data[L])
    for treatment in treatments:
        mt = np.mean(treatment.data[L])
        print(f"Length {L:<8,} {mc:>8.1f} {mt:>10.1f} {pct_diffs[idx]:>+7.1f}% [{pct_ci_lo[idx]:>+6.1f}%, {pct_ci_hi[idx]:>+5.1f}%] {p_values[idx]:>8.4f} {'*' if significant[idx] else ''}")
        idx += 1

# ============================================================
# 4. Time series scatter
# ============================================================
fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 2.4), sharex=True)
axes = axes.flatten()

for i, L in enumerate(lengths):
    ax = axes[i]
    n_ctrl= len(control.data[L])

    ax.scatter(np.arange(n_ctrl), control.data[L], s=12, alpha=0.5, color=DATA_COLORS[0], label=control.label, zorder=2)

    for (treatment, color) in ((treatments[ti], DATA_COLORS[ti + 1]) for ti in range(len(treatments))):
        nt = len(treatment.data[L])
        ax.scatter(np.arange(nt), treatment.data[L], s=12, alpha=0.5, color=color, label=treatment.label, zorder=2)

    w = 10
    if n_ctrl >= w:
        rmc = np.convolve(control.data[L], np.ones(w)/w, mode='valid')
        ax.plot(np.arange(w-1, n_ctrl), rmc, color=deemphasize_color(DATA_COLORS[0]), linewidth=1.5, zorder=3)

    for (treatment, color) in ((treatments[ti], DATA_COLORS[ti + 1]) for ti in range(len(treatments))):
        nt = len(treatment.data[L])
        if nt >= w:
            rmt = np.convolve(treatment.data[L], np.ones(w)/w, mode='valid')
            ax.plot(np.arange(w-1, nt), rmt, color=deemphasize_color(color), linewidth=1.5, zorder=3)

    ax.set_title(f"Length {L:,}", fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.2)
    if i == 0:
        ax.legend(fontsize=7, loc='upper right')

for ax in axes[-ncols:]:
    ax.set_xlabel("Sample #", fontsize=9)
for ax in axes[::ncols]:
    ax.set_ylabel("Time", fontsize=9)
for j in range(len(lengths), len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Sample-by-sample time series\n"
             "(dots = raw samples, line = rolling mean window=10)",
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
save_fig(plt, "timeseries.png")

# ============================================================
# 5. ECDF
# ============================================================
fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 2.6))
axes = axes.flatten()

for i, L in enumerate(lengths):
    ax = axes[i]
    for label, samples, color in [
        (control.label, control.data[L], DATA_COLORS[0]), 
        *((treatments[ti].label, treatments[ti].data[L], DATA_COLORS[ti + 1]) for ti in range(len(treatments)))
    ]:
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


fig.suptitle(f"Empirical CDF: {vs_label()}",
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
save_fig(plt, "ecdf.png")

# ============================================================
# 6. KDE density
# ============================================================
fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 2.6))
axes = axes.flatten()

for i, L in enumerate(lengths):
    ax = axes[i]
    for label, samples, color in [
        (control.label, control.data[L], DATA_COLORS[0]), 
        *((treatments[ti].label, treatments[ti].data[L], DATA_COLORS[ti + 1]) for ti in range(len(treatments)))
    ]:
        if np.std(samples) == 0:
            # Zero variance: draw a vertical line at the constant value
            ax.axvline(samples[0], color=color, linewidth=1.5, label=label)
        else:
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

for ax in axes[-ncols:]:
    ax.set_xlabel("Time", fontsize=9)
for j in range(len(lengths), len(axes)):
    axes[j].set_visible(False)

fig.suptitle(f"KDE Density: {vs_label()}",
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
save_fig(plt, "kde.png")

# ============================================================
# 7. Heatmap vs Control
# ============================================================
heatmap_data = np.zeros((len(lengths), len(treatments)))
for i, L in enumerate(lengths):
    mc = np.mean(control.data[L])
    for j, treatment in enumerate(treatments):
        mt = np.mean(treatment.data[L])
        heatmap_data[i, j] = (mt - mc) / mc * 100

fig, ax = plt.subplots(figsize=(max(8, len(treatments) * 1.5 + 2), max(4, len(lengths) * 0.6 + 2)))
vmax = max(abs(heatmap_data.min()), abs(heatmap_data.max()))
im = ax.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto', vmin=-vmax, vmax=vmax)
for i in range(len(lengths)):
    best_j = int(np.argmin(heatmap_data[i]))
    for j in range(len(treatments)):
        ax.text(j, i, f"{heatmap_data[i, j]:+.1f}%", ha='center', va='center', fontsize=9,
                color='white' if abs(heatmap_data[i, j]) > vmax * 0.6 else 'black')
        if j == best_j:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor='black', linewidth=5, zorder=2))
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor='white', linewidth=3, zorder=3))
ax.set_xticks(range(len(treatments)))
ax.set_xticklabels([t.label for t in treatments], rotation=45, ha='right')
ax.set_yticks(range(len(lengths)))
ax.set_yticklabels([str(L) for L in lengths])
ax.set_xlabel("Treatment")
ax.set_ylabel("Array Length")
ax.set_title(f"% Change vs Control ('{control.label}')\nGreen = faster, Red = slower", fontweight='bold')
fig.colorbar(im, ax=ax, label='% change', shrink=0.8)
plt.tight_layout()
save_fig(plt, "heatmap_vs_control.png")

# ============================================================
# 8. Heatmap vs Best
# ============================================================
all_entries = [control, *treatments]
all_labels = [e.label for e in all_entries]
heatmap_best = np.zeros((len(lengths), len(all_entries)))
for i, L in enumerate(lengths):
    means = [np.mean(e.data[L]) for e in all_entries]
    best = min(means)
    for j, m in enumerate(means):
        heatmap_best[i, j] = (m - best) / best * 100

fig, ax = plt.subplots(figsize=(max(8, len(all_entries) * 1.5 + 2), max(4, len(lengths) * 0.6 + 2)))
im = ax.imshow(heatmap_best, cmap='YlOrRd', aspect='auto', vmin=0)
for i in range(len(lengths)):
    for j in range(len(all_entries)):
        val = heatmap_best[i, j]
        if val == 0:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, color='#2ecc71', zorder=2))
            ax.text(j, i, "\u2713 best", ha='center', va='center', fontsize=9,
                    color='white', fontweight='bold', zorder=3)
        else:
            ax.text(j, i, f"{val:.1f}%", ha='center', va='center', fontsize=9,
                    color='white' if val > heatmap_best.max() * 0.6 else 'black')

ax.set_xticks(range(len(all_entries)))
ax.set_xticklabels(all_labels, rotation=45, ha='right')
ax.set_yticks(range(len(lengths)))
ax.set_yticklabels([str(L) for L in lengths])
ax.set_xlabel("Entry")
ax.set_ylabel("Array Length")
ax.set_title("% Change vs Best (per length)\nWhite = best, Red = worst", fontweight='bold')
fig.colorbar(im, ax=ax, label='% slower than best', shrink=0.8)
plt.tight_layout()
save_fig(plt, "heatmap_vs_best.png")

# ============================================================
# 9. Small multiples bar chart (% vs best)
# ============================================================
all_entries = [control, *treatments]
all_labels = [e.label for e in all_entries]
all_colors = [DATA_COLORS[i] for i in range(len(all_entries))]

# Compute all % vs best values and global max
grid = np.zeros((len(lengths), len(all_entries)))
win_counts = [0] * len(all_entries)
for i, L in enumerate(lengths):
    means = [np.mean(e.data[L]) for e in all_entries]
    best = min(means)
    best_idx = means.index(best)
    win_counts[best_idx] += 1
    for j, m in enumerate(means):
        grid[i, j] = (m - best) / best * 100
global_max = grid.max() if grid.max() > 0 else 1

n_rows = len(lengths)
n_cols = len(all_entries)
cell_h = 1.0
col_w = 1.6  # width allocated per column
max_bar_w = col_w * 0.9  # max width a bar can be
min_bar_w = 0.06  # thin line for best
pad = 0.05

fig, ax = plt.subplots(figsize=(max(8, n_cols * col_w + 2), max(6, n_rows * 0.65 + 3)))

# Draw segments: width encodes magnitude, centered in each column
for i in range(n_rows):
    for j in range(n_cols):
        val = grid[i, j]
        cx = j * col_w + col_w / 2
        cy = i * cell_h + cell_h / 2

        if val == 0:
            # Best: thin line + label
            w = min_bar_w
            ax.add_patch(plt.Rectangle((cx - w / 2, i * cell_h + pad), w, cell_h - 2 * pad,
                                       fill=True, facecolor='#2ecc71', edgecolor='#2ecc71',
                                       linewidth=1.5, zorder=2))
            ax.text(cx, cy, "best", ha='center', va='center', fontsize=8,
                    fontweight='bold', color='#2ecc71',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1), zorder=3)
        else:
            frac = val / global_max if global_max > 0 else 0
            w = min_bar_w + frac * (max_bar_w - min_bar_w)
            ax.add_patch(plt.Rectangle((cx - w / 2, i * cell_h + pad), w, cell_h - 2 * pad,
                                       fill=True, facecolor=all_colors[j], alpha=0.8,
                                       edgecolor=darken_color(all_colors[j]), linewidth=0.5, zorder=2))
            ax.text(cx, cy, f"{val:.1f}%", ha='center', va='center', fontsize=7,
                    color='white' if frac > 0.25 else '#333333', fontweight='bold', zorder=3)

    # Row divider
    ax.axhline(i * cell_h, color='#dddddd', linewidth=0.5, zorder=0)

# Vertical spine lines connecting segments
for j in range(n_cols):
    cx = j * col_w + col_w / 2
    ax.plot([cx, cx], [0, n_rows * cell_h], color='#cccccc', linewidth=0.5, zorder=0)

# Wins row
wins_y = n_rows * cell_h + cell_h * 0.4
ax.axhline(wins_y - 0.1, color='#999999', linewidth=1, zorder=0)
for j in range(n_cols):
    cx = j * col_w + col_w / 2
    ax.text(cx, wins_y + 0.2, f"{win_counts[j]} wins", ha='center', va='center',
            fontsize=11, fontweight='bold', color=all_colors[j], zorder=3)

# Axes setup
ax.set_xlim(-0.3, n_cols * col_w + 0.3)
ax.set_ylim(-0.3, wins_y + 0.6)
ax.invert_yaxis()

ax.set_xticks([j * col_w + col_w / 2 for j in range(n_cols)])
ax.set_xticklabels(all_labels, fontsize=11, fontweight='bold')
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

ax.set_yticks([i * cell_h + cell_h / 2 for i in range(n_rows)])
ax.set_yticklabels([str(L) for L in lengths], fontsize=9, fontweight='bold')
ax.set_ylabel("Array Length", fontsize=11)
ax.set_title("% Slower than Best — wider = worse, thin = best\n", fontsize=13, fontweight='bold', pad=20)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(length=0)

plt.tight_layout()
save_fig(plt, "bars_vs_best.png")
