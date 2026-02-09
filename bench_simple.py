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
alpha = 0.05

means_128, means_256 = [], []
ci_lo_128, ci_hi_128 = [], []
ci_lo_256, ci_hi_256 = [], []

for L in lengths:
    a, b = d128[L], d256[L]
    m_a, m_b = np.mean(a), np.mean(b)
    ci_a = stats.t.interval(1 - alpha, len(a) - 1, loc=m_a, scale=stats.sem(a))
    ci_b = stats.t.interval(1 - alpha, len(b) - 1, loc=m_b, scale=stats.sem(b))
    means_128.append(m_a)
    means_256.append(m_b)
    ci_lo_128.append(ci_a[0])
    ci_hi_128.append(ci_a[1])
    ci_lo_256.append(ci_b[0])
    ci_hi_256.append(ci_b[1])

means_128 = np.array(means_128)
means_256 = np.array(means_256)
ci_lo_128 = np.array(ci_lo_128)
ci_hi_128 = np.array(ci_hi_128)
ci_lo_256 = np.array(ci_lo_256)
ci_hi_256 = np.array(ci_hi_256)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
x = np.arange(len(lengths))
labels = [str(L) for L in lengths]

# Top: absolute times with CIs
ax1.errorbar(x - 0.15, means_128, yerr=[means_128 - ci_lo_128, ci_hi_128 - means_128],
             fmt='o-', capsize=4, label='128-bit loads', color=CONTROL)
ax1.errorbar(x + 0.15, means_256, yerr=[means_256 - ci_lo_256, ci_hi_256 - means_256],
             fmt='s-', capsize=4, label='256-bit loads', color=TREATMENT)
ax1.set_ylabel("Time (units from benchmark)")
ax1.set_title("AVX/FMA f32 Benchmark: 128-bit vs 256-bit loads")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=45, ha='right')

# Bottom: difference (256 - 128) with Welch CI
diffs = []
diff_ci_lo = []
diff_ci_hi = []
for L in lengths:
    a, b = d128[L], d256[L]
    d = np.mean(b) - np.mean(a)
    se = np.sqrt(stats.sem(a)**2 + stats.sem(b)**2)
    df_w = (stats.sem(a)**2 + stats.sem(b)**2)**2 / (
        stats.sem(a)**4 / (len(a)-1) + stats.sem(b)**4 / (len(b)-1))
    t_crit = stats.t.ppf(1 - alpha/2, df_w)
    diffs.append(d)
    diff_ci_lo.append(d - t_crit * se)
    diff_ci_hi.append(d + t_crit * se)

diffs = np.array(diffs)
diff_ci_lo = np.array(diff_ci_lo)
diff_ci_hi = np.array(diff_ci_hi)

ax2.errorbar(x, diffs, yerr=[diffs - diff_ci_lo, diff_ci_hi - diffs],
             fmt='D-', capsize=4, color='#2ca02c')
ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax2.set_ylabel("Difference (256-bit minus 128-bit)")
ax2.set_xlabel("Array Length")
ax2.set_title("Mean Difference with 95% CI (positive = 256-bit slower)")
ax2.grid(True, alpha=0.3)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=45, ha='right')

plt.tight_layout()
plt.savefig("bench_simple.png", dpi=150)
print("Saved bench_simple.png")
