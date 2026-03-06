import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os

# --- SUBJECT FILES (All 10 subjects) ---
SUBJECT_FILES = [
    ('Subject 1', 'subject_1_data.csv'),
    ('Subject 2', 'subject_2_data.csv'),
    ('Subject 3', 'subject_3_data.csv'),
    ('Subject 4', 'subject_4_data.csv'),
    ('Subject 5', 'subject_5_data.csv'),
    ('Subject 6', 'subject_6_data.csv'),  
    ('Subject 7', 'subject_7_data.csv'),
    ('Subject 8', 'subject_8_data.csv'),
    ('Subject 9', 'subject_9_data.csv'),
    ('Subject 10', 'subject_10_data.csv'),
]

class OneEuroFilter:
    def __init__(self, initial_val, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff; self.beta = beta; self.d_cutoff = d_cutoff
        self.x_prev = initial_val; self.dx_prev = 0.0; self.t_prev = 0.0
    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)
    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev
    def filter(self, x, t):
        t_e = max(t - self.t_prev, 0.016)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(self.smoothing_factor(t_e, self.d_cutoff), dx, self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)
        self.x_prev = x_hat; self.dx_prev = dx_hat; self.t_prev = t
        return x_hat

def process_continuous_for_graph(df, window_frames=35):
    quiet_end = df['Raw_Input'].rolling(window=window_frames).std().idxmin()
    quiet_start = quiet_end - window_frames
    bias = df['Raw_Input'].iloc[quiet_start:quiet_end].mean()

    raw_cal = df['Raw_Input'].values - bias
    times = df['Timestamp_s'].values

    ema_val = raw_cal[0]
    bvic_val = raw_cal[0]
    euro = OneEuroFilter(initial_val=raw_cal[0])

    out_ema, out_bvic, out_euro = [], [], []

    # B-VIC with hysteresis + zero-equilibrium static mode.
    THRESHOLD = 150.0
    CONFIRM_FRAMES = 3
    DECAY_STATIC = 0.95
    ALPHA_STATIC = 0.02
    ALPHA_DYNAMIC = 0.60

    dynamic_counter = 0
    in_dynamic_mode = False

    for val, t in zip(raw_cal, times):
        ema_val = ema_val + 0.05 * (val - ema_val)
        out_ema.append(ema_val)
        out_euro.append(euro.filter(val, t))

        error = abs(val - bvic_val)
        if error > THRESHOLD:
            dynamic_counter += 1
        else:
            dynamic_counter = 0

        if dynamic_counter >= CONFIRM_FRAMES:
            in_dynamic_mode = True
        elif error <= THRESHOLD:
            in_dynamic_mode = False

        if in_dynamic_mode:
            bvic_val = bvic_val + ALPHA_DYNAMIC * (val - bvic_val)
        else:
            bvic_val = DECAY_STATIC * bvic_val + ALPHA_STATIC * val
        out_bvic.append(bvic_val)

    df['Calibrated_Raw'] = raw_cal
    df['EMA'] = out_ema
    df['Euro'] = out_euro
    df['BVIC'] = out_bvic

    static_win = df.iloc[quiet_start:quiet_end]
    peak_idx = df['Calibrated_Raw'].abs().idxmax()
    dyn_win = df.iloc[max(0, peak_idx - 30) : min(len(df), peak_idx + 50)]

    return static_win, dyn_win

# --- LOAD ALL SUBJECTS ---
print(f"Loading {len(SUBJECT_FILES)} subject files...")
subject_data = []

for name, filename in SUBJECT_FILES:
    if not os.path.exists(filename):
        print(f"  WARNING: {filename} not found, skipping...")
        continue
    try:
        df = pd.read_csv(filename)
        static_win, dyn_win = process_continuous_for_graph(df)
        subject_data.append((name, static_win, dyn_win))
        print(f"  {name}: Processed successfully")
    except Exception as e:
        print(f"  ERROR processing {filename}: {e}")

print(f"\nSuccessfully loaded {len(subject_data)} subjects")

# --- PRINT STATS FOR ALL SUBJECTS ---
print("\n--- B-VIC Static Window Stats (All Subjects) ---")
for name, stat, _ in subject_data:
    b = stat['BVIC']
    print(f"{name}: mean={b.mean():.2f}, std={b.std():.2f}, range=[{b.min():.2f}, {b.max():.2f}]")

# --- PLOT HELPER ---
def plot_panel(ax, data, title, is_dynamic=False):
    line_w = 4 if is_dynamic else 1.5
    ax.plot(data['Timestamp_s'], data['Calibrated_Raw'], label='Raw Input',
            color='lightgray', linewidth=line_w, alpha=0.7)
    ax.plot(data['Timestamp_s'], data['EMA'], label='Standard EMA',
            color='red', linestyle='--', linewidth=1.5)
    ax.plot(data['Timestamp_s'], data['Euro'], label='1 Euro Filter',
            color='orange', linewidth=1.5)
    ax.plot(data['Timestamp_s'], data['BVIC'], label='B-VIC (Ours)',
            color='green', linewidth=2.5)
    ax.set_title(title, fontsize=11, loc='left')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Ang. Vel. (deg/s)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left' if is_dynamic else 'upper right', fontsize=8)

# --- GRAPH 1: FULL GRID (All Subjects - Static & Dynamic) ---
n_subjects = len(subject_data)
fig, axes = plt.subplots(n_subjects, 2, figsize=(16, 3 * n_subjects))
fig.suptitle(f'Multi-Subject Validation (N={n_subjects}): Static Jitter vs Dynamic Response',
             fontsize=16, fontweight='bold', y=0.995)

for i, (name, stat, dyn) in enumerate(subject_data):
    plot_panel(axes[i, 0], stat, f'({chr(97+i*2)}) {name}: Static Retention')
    plot_panel(axes[i, 1], dyn, f'({chr(98+i*2)}) {name}: Ballistic Flick', is_dynamic=True)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig('empirical_validation_all_subjects.png', dpi=300, bbox_inches='tight')
print("\nSaved: empirical_validation_all_subjects.png")

# --- GRAPH 2: SUMMARY COMPARISON (4 Representative Subjects) ---
# Pick first 4 subjects for a cleaner 2x4 grid
if len(subject_data) >= 4:
    fig2, axes2 = plt.subplots(2, 4, figsize=(20, 8))
    fig2.suptitle('Representative Subject Comparison: Static vs Dynamic Performance',
                  fontsize=16, fontweight='bold', y=0.98)
    
    for i in range(4):
        name, stat, dyn = subject_data[i]
        plot_panel(axes2[0, i], stat, f'{name}: Static')
        plot_panel(axes2[1, i], dyn, f'{name}: Dynamic', is_dynamic=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('empirical_validation_summary.png', dpi=300, bbox_inches='tight')
    print("Saved: empirical_validation_summary.png")

# --- GRAPH 3: STATIC-ONLY COMPARISON (All Subjects in Grid) ---
cols = 5
rows = (n_subjects + cols - 1) // cols
fig3, axes3 = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
fig3.suptitle('Static Jitter Suppression Across All Subjects', fontsize=16, fontweight='bold', y=0.99)

axes3_flat = axes3.flatten() if n_subjects > 1 else [axes3]
for i, (name, stat, _) in enumerate(subject_data):
    ax = axes3_flat[i]
    ax.plot(stat['Timestamp_s'], stat['Calibrated_Raw'], label='Raw', color='lightgray', alpha=0.7)
    ax.plot(stat['Timestamp_s'], stat['BVIC'], label='B-VIC', color='green', linewidth=2)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('deg/s', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

# Hide unused subplots
for j in range(i + 1, len(axes3_flat)):
    axes3_flat[j].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('static_comparison_grid.png', dpi=300, bbox_inches='tight')
print("Saved: static_comparison_grid.png")

print("\nSUCCESS: All graphs generated.")
