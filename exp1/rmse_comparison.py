import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Per-subject RMSE values from kalman.py output ──────────────────────
subjects = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']

raw    = [16.508, 15.638, 11.824, 14.356, 16.746, 13.410, 16.040,  8.988,  7.909, 16.251]
ema    = [ 7.449,  1.512,  3.682,  3.823,  2.862,  0.770,  7.798,  0.830, 10.862,  3.611]
euro   = [11.255, 13.474, 15.139,  8.248, 13.309,  7.433,  9.514,  4.731,  3.321,  6.947]
kalman = [ 8.829,  6.195,  9.371,  3.215,  6.780,  2.345,  7.263,  1.894,  6.070,  3.938]
bvic   = [ 0.220,  0.400,  1.327,  3.252,  1.474,  0.237,  3.822,  0.343,  6.281,  0.431]

# ── Medians (horizontal reference lines) ────────────────────────────────────
med_raw    = np.median(raw)
med_ema    = np.median(ema)
med_euro   = np.median(euro)
med_kalman = np.median(kalman)
med_bvic   = np.median(bvic)

# ── Plot setup ───────────────────────────────────────────────────────────────
x      = np.arange(len(subjects))
width  = 0.15
fig, ax = plt.subplots(figsize=(14, 6))

colors = {
    'Raw':    '#aaaaaa',
    'EMA':    '#e05c5c',
    'Euro':   '#e8a020',
    'Kalman': '#5588cc',
    'B-VIC':  '#2a9d2a',
}

b1 = ax.bar(x - 2*width, raw,    width, label='Raw Input',       color=colors['Raw'],    alpha=0.85)
b2 = ax.bar(x - 1*width, ema,    width, label='Standard EMA',    color=colors['EMA'],    alpha=0.85)
b3 = ax.bar(x,           euro,   width, label='1€ Filter',       color=colors['Euro'],   alpha=0.85)
b4 = ax.bar(x + 1*width, kalman, width, label='Kalman Filter',   color=colors['Kalman'], alpha=0.85)
b5 = ax.bar(x + 2*width, bvic,   width, label='B-VIC (Ours)',    color=colors['B-VIC'],  alpha=0.90,
            edgecolor='#1a6e1a', linewidth=0.8)

# ── Median lines ─────────────────────────────────────────────────────────────
ax.axhline(med_bvic,   color=colors['B-VIC'],  linestyle='--', linewidth=1.2, alpha=0.7)
ax.axhline(med_kalman, color=colors['Kalman'], linestyle='--', linewidth=1.0, alpha=0.5)

# ── S9 outlier annotation ─────────────────────────────────────────────────────
s9_idx = subjects.index('S9')
ax.annotate('S9: boundary\ncondition\n(6.28 deg/s)',
            xy=(s9_idx + 2*width, bvic[s9_idx]),
            xytext=(s9_idx + 2*width + 0.55, bvic[s9_idx] + 1.2),
            fontsize=8, color='#1a6e1a',
            arrowprops=dict(arrowstyle='->', color='#1a6e1a', lw=1.2),
            ha='left')

# ── Median label on right axis ────────────────────────────────────────────────
ax.text(len(subjects) - 0.35, med_bvic + 0.15,
        f'B-VIC median = {med_bvic:.3f}', fontsize=8,
        color=colors['B-VIC'], va='bottom')
ax.text(len(subjects) - 0.35, med_kalman + 0.15,
        f'Kalman median = {med_kalman:.3f}', fontsize=8,
        color=colors['Kalman'], va='bottom')

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(subjects, fontsize=11)
ax.set_ylabel('Static Jitter RMSE (deg/s)', fontsize=12)
ax.set_xlabel('Subject', fontsize=12)
ax.set_title('Per-Subject Static Jitter RMSE Across All Algorithms (N=10)',
             fontsize=13, fontweight='bold')
ax.set_ylim(0, 20)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

plt.tight_layout()
plt.savefig('rmse_per_subject.png', dpi=300, bbox_inches='tight')
print("Saved: rmse_per_subject.png")
plt.close()