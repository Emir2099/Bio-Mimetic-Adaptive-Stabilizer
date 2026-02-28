import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# ── 1. Filter Implementations ──────────────────────────────────────────────────

class OneEuroFilter:
    def __init__(self, initial_val, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff
        self.x_prev     = initial_val
        self.dx_prev    = 0.0
        self.t_prev     = 0.0

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def filter(self, x, t):
        t_e    = max(t - self.t_prev, 0.016)
        dx     = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(
                     self.smoothing_factor(t_e, self.d_cutoff), dx, self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a      = self.smoothing_factor(t_e, cutoff)
        x_hat  = self.exponential_smoothing(a, x, self.x_prev)
        self.x_prev  = x_hat
        self.dx_prev = dx_hat
        self.t_prev  = t
        return x_hat


# ── 2. Full Processing Pipeline ────────────────────────────────────────────────

def process(df, window_frames=35):
    """
    1. Find the quietest 35-frame window → estimate sensor DC bias
    2. Subtract bias from entire recording (calibration)
    3. Run all filters over the full calibrated signal
    4. Return the processed DataFrame
    """
    # ── Bias calibration ──
    quiet_end   = df['Raw_Input'].rolling(window=window_frames).std().idxmin()
    quiet_start = quiet_end - window_frames
    bias        = df['Raw_Input'].iloc[quiet_start:quiet_end].mean()
    raw_cal     = df['Raw_Input'].values - bias
    times       = df['Timestamp_s'].values

    # ── Initialise filters ──
    ema_val  = raw_cal[0]
    bvic_val = raw_cal[0]
    euro     = OneEuroFilter(initial_val=raw_cal[0])

    out_ema, out_bvic, out_euro = [], [], []

    # ── B-VIC hysteresis parameters ──
    THRESHOLD     = 150.0   # deg/s  — ~9× the noise floor (~16 deg/s)
    CONFIRM_FRAMES = 3      # frames above threshold needed to enter Action mode
    ALPHA_DYN     = 0.60   # Action mode gain     (low impedance)
    ALPHA_STA     = 0.02   # Stabilisation gain   (high impedance)
    GAMMA         = 0.95   # Zero-equilibrium decay for static mode

    dynamic_counter  = 0
    in_dynamic_mode  = False

    for val, t in zip(raw_cal, times):

        # Standard EMA (fixed alpha = 0.05)
        ema_val = ema_val + 0.05 * (val - ema_val)
        out_ema.append(ema_val)

        # 1 Euro Filter
        out_euro.append(euro.filter(val, t))

        # B-VIC ──────────────────────────────────────────────────────────────
        error = abs(val - bvic_val)          # kinetic energy proxy

        # Hysteresis counter: must see CONFIRM_FRAMES consecutive high-energy
        # frames before entering Action mode.  A single spike won't flip it.
        if error > THRESHOLD:
            dynamic_counter += 1
        else:
            dynamic_counter = 0

        if dynamic_counter >= CONFIRM_FRAMES:
            in_dynamic_mode = True
        elif error <= THRESHOLD:
            in_dynamic_mode = False
        # (else: hold current mode during the ramp-up window)

        if in_dynamic_mode:
            # Action mode: high passthrough, near-zero latency
            bvic_val = bvic_val + ALPHA_DYN * (val - bvic_val)
        else:
            # Stabilisation mode: zero-equilibrium update
            # Equilibrium is 0 (true angular velocity during a static hold),
            # NOT the noisy instantaneous sensor reading.
            bvic_val = GAMMA * bvic_val + ALPHA_STA * val

        out_bvic.append(bvic_val)
        # ─────────────────────────────────────────────────────────────────────

    df['Calibrated_Raw'] = raw_cal
    df['EMA']            = out_ema
    df['Euro']           = out_euro
    df['BVIC']           = out_bvic
    return df, quiet_start, quiet_end


# ── 3. Latency Measurement ─────────────────────────────────────────────────────

def measure_latency(df, name="Subject"):
    """
    Isolates the ballistic flick window (±N frames around the raw signal peak)
    and computes peak-to-peak latency for each filter relative to the raw peak.

    L_dyn = t_peak(filtered) - t_peak(raw)

    Positive  → filter output peak lags  the raw peak  (bad: swimming effect)
    Zero      → filter output peak is in the same frame (ideal)
    Negative  → filter output peak leads the raw peak  (physically impossible
                with a causal filter; would indicate data artefact)
    """
    # Find the global peak of the raw calibrated signal
    peak_idx  = df['Calibrated_Raw'].abs().idxmax()
    dyn_start = max(0,          peak_idx - 30)
    dyn_end   = min(len(df)-1,  peak_idx + 50)
    dyn       = df.iloc[dyn_start:dyn_end].copy()

    # Peak timestamps for each signal
    raw_peak_t  = dyn.loc[dyn['Calibrated_Raw'].abs().idxmax(), 'Timestamp_s']
    ema_peak_t  = dyn.loc[dyn['EMA'].abs().idxmax(),            'Timestamp_s']
    euro_peak_t = dyn.loc[dyn['Euro'].abs().idxmax(),           'Timestamp_s']
    bvic_peak_t = dyn.loc[dyn['BVIC'].abs().idxmax(),           'Timestamp_s']

    # Convert to milliseconds
    ema_lat  = (ema_peak_t  - raw_peak_t) * 1000
    euro_lat = (euro_peak_t - raw_peak_t) * 1000
    bvic_lat = (bvic_peak_t - raw_peak_t) * 1000

    # Sampling rate info
    dt_all    = np.diff(df['Timestamp_s'].values)
    mean_dt   = np.mean(dt_all)   * 1000   # ms
    median_dt = np.median(dt_all) * 1000   # ms

    print(f"\n{'='*55}")
    print(f"  Latency Analysis — {name}")
    print(f"{'='*55}")
    print(f"  Sampling: mean {mean_dt:.1f} ms  |  median {median_dt:.1f} ms")
    print(f"  One frame ≈ {median_dt:.1f} ms  →  sub-frame threshold")
    print(f"\n  Raw peak:  t = {raw_peak_t:.4f} s  |  "
          f"|velocity| = {dyn['Calibrated_Raw'].abs().max():.1f} deg/s")
    print(f"\n  {'Filter':<18} {'Peak time (s)':<16} {'Latency (ms)':<15} Sub-frame?")
    print(f"  {'-'*60}")
    for label, pt, lat in [
        ("Standard EMA",     ema_peak_t,  ema_lat),
        ("1 Euro Filter",    euro_peak_t, euro_lat),
        ("B-VIC (proposed)", bvic_peak_t, bvic_lat),
    ]:
        sub = "YES ✓" if abs(lat) < median_dt else "NO  ✗"
        print(f"  {label:<18} {pt:<16.4f} {lat:<15.1f} {sub}")
    print(f"{'='*55}\n")

    return dyn, {
        'raw_peak_t':  raw_peak_t,
        'ema_lat_ms':  ema_lat,
        'euro_lat_ms': euro_lat,
        'bvic_lat_ms': bvic_lat,
        'median_dt_ms': median_dt,
    }


# ── 4. Optional: Plot the Flick Window ────────────────────────────────────────

def plot_flick(dyn, latency_info, name="Subject"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dyn['Timestamp_s'], dyn['Calibrated_Raw'],
            color='lightgray', linewidth=3, alpha=0.8, label='Raw Input')
    ax.plot(dyn['Timestamp_s'], dyn['EMA'],
            color='red',    linestyle='--', linewidth=1.5, label='Standard EMA')
    ax.plot(dyn['Timestamp_s'], dyn['Euro'],
            color='orange', linewidth=1.5, label='1 Euro Filter')
    ax.plot(dyn['Timestamp_s'], dyn['BVIC'],
            color='green',  linewidth=2.5, label='B-VIC (Ours)')

    # Annotate the raw peak
    ax.axvline(latency_info['raw_peak_t'], color='black',
               linestyle=':', linewidth=1, label='Raw peak')

    ax.set_title(f'{name}: Ballistic Flick — Latency Analysis', fontsize=13)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angular Velocity (deg/s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'latency_{name.replace(" ","_")}.png', dpi=200)
    plt.show()
    print(f"  Plot saved → latency_{name.replace(' ','_')}.png")


# ── 5. Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Load your CSV files here ──
    # Expected columns: Timestamp_s, Raw_Input
    # (add more subjects by extending the list below)
    subjects = [
        ("subject1_data.csv", "Subject 1"),
        ("subject2_data.csv", "Subject 2"),
    ]

    for csv_path, label in subjects:
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"  [SKIP] {csv_path} not found.")
            continue

        df, qs, qe = process(df)
        dyn, info  = measure_latency(df, name=label)
        plot_flick(dyn, info, name=label)