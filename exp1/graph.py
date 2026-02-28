import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

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
    #
    # Two fixes over the naive implementation:
    #
    # 1. HYSTERESIS: require CONFIRM_FRAMES consecutive frames above
    #    threshold before entering dynamic mode. Prevents single sensor
    #    spike frames from causing a full mode switch and displacing the
    #    filter state.
    #
    # 2. ZERO-EQUILIBRIUM STATIC FILTER: instead of y[n] = y[n-1] + a*(x-y[n-1])
    #    (which has its equilibrium at x, the noisy raw signal), we use:
    #    y[n] = DECAY * y[n-1] + ALPHA * x[n]
    #    This has its equilibrium at 0 (since true angular velocity during
    #    a static hold IS zero). The filter still tracks small tremor via
    #    the ALPHA * x term, but DECAY < 1 ensures the output is always
    #    attracted back to zero. Eliminates accumulated drift from prior
    #    motion entering the static window.
    THRESHOLD = 150.0
    CONFIRM_FRAMES = 3
    DECAY_STATIC = 0.95      # zero-equilibrium decay (< 1)
    ALPHA_STATIC = 0.02      # tremor-follow weight in static mode
    ALPHA_DYNAMIC = 0.60     # high-response weight in dynamic mode
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
            # Zero-equilibrium: pulls toward 0 while still tracking tremor
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

df1 = pd.read_csv('subject1_data.csv')
df2 = pd.read_csv('subject2_data.csv')

s1_stat, s1_dyn = process_continuous_for_graph(df1)
s2_stat, s2_dyn = process_continuous_for_graph(df2)

for name, stat in [('S1', s1_stat), ('S2', s2_stat)]:
    b = stat['BVIC']
    print(f"{name} BVIC static: mean={b.mean():.2f}, std={b.std():.2f}, "
          f"range=[{b.min():.2f}, {b.max():.2f}]")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Multi-Subject Bimodal Targeting Protocol Validation',
             fontsize=18, fontweight='bold', y=0.98)

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
    ax.set_title(title, fontsize=14, loc='left')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Angular Velocity (deg/s)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left' if is_dynamic else 'upper right')

plot_panel(axes[0, 0], s1_stat, '(a) Subject 1: Static Retention (Jitter Suppression)')
plot_panel(axes[0, 1], s1_dyn,  '(b) Subject 1: Ballistic Flick (Latency Analysis)', is_dynamic=True)
plot_panel(axes[1, 0], s2_stat, '(c) Subject 2: Static Retention (Jitter Suppression)')
plot_panel(axes[1, 1], s2_dyn,  '(d) Subject 2: Ballistic Flick (Latency Analysis)', is_dynamic=True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('empirical_validation_grid_fixed.png', dpi=300, bbox_inches='tight')
print("SUCCESS: Fixed graph saved.")
