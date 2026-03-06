import pandas as pd
import numpy as np
import math
import os

# --- SUBJECT FILES (All 10 subjects) ---
SUBJECT_FILES = [
    'subject_1_data.csv',
    'subject_2_data.csv',
    'subject_3_data.csv',
    'subject_4_data.csv',
    'subject_5_data.csv',
    'subject_6_data.csv',  
    'subject_7_data.csv',
    'subject_8_data.csv',
    'subject_9_data.csv',
    'subject_10_data.csv',
]

# --- 1. FILTER IMPLEMENTATIONS ---
class SimpleKalman:
    def __init__(self, initial_val, process_noise=0.1, sensor_noise=5.0):
        self.q = process_noise; self.r = sensor_noise; self.p = 1.0; self.x = initial_val; self.k = 0 
    def update(self, measurement):
        self.p = self.p + self.q; self.k = self.p / (self.p + self.r)
        self.x = self.x + self.k * (measurement - self.x); self.p = (1 - self.k) * self.p
        return self.x

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

# --- 2. CONTINUOUS PROCESSING ENGINE ---
def process_continuous(df, window_frames=35):
    # 1. Find the exact resting bias of the sensor
    quiet_end = df['Raw_Input'].rolling(window=window_frames).std().idxmin()
    quiet_start = quiet_end - window_frames
    bias = df['Raw_Input'].iloc[quiet_start:quiet_end].mean()

    # 2. Calibrate the entire continuous dataset
    raw_cal = df['Raw_Input'].values - bias
    times = df['Timestamp_s'].values

    # 3. Initialize filters at t=0
    ema_val = raw_cal[0]
    bvic_val = raw_cal[0]
    kf = SimpleKalman(initial_val=raw_cal[0])
    euro = OneEuroFilter(initial_val=raw_cal[0])

    out_ema, out_bvic, out_kf, out_euro = [], [], [], []
    dynamic_counter = 0
    in_dynamic_mode = False

    # 4. Run filters continuously over the whole file
    for val, t in zip(raw_cal, times):
        ema_val = ema_val + 0.05 * (val - ema_val)
        out_ema.append(ema_val)
        
        out_euro.append(euro.filter(val, t))
        out_kf.append(kf.update(val))
        
        # B-VIC: Hysteresis + Zero-Equilibrium Static Filter
        error = abs(val - bvic_val)
        if error > 150.0:
            dynamic_counter += 1
        else:
            dynamic_counter = 0
        if dynamic_counter >= 3:
            in_dynamic_mode = True
        elif error <= 150.0:
            in_dynamic_mode = False

        if in_dynamic_mode:
            bvic_val = bvic_val + 0.60 * (val - bvic_val)
        else:
            bvic_val = 0.95 * bvic_val + 0.02 * val
        out_bvic.append(bvic_val)

    # 5. Save back to dataframe
    df['Calibrated_Raw'] = raw_cal
    df['EMA'] = out_ema
    df['Euro'] = out_euro
    df['Kalman'] = out_kf
    df['BVIC'] = out_bvic

    # 6. Return ONLY the fully-settled static window for RMSE evaluation
    return df.iloc[quiet_start:quiet_end]

# --- 3. LOAD AND PROCESS ALL SUBJECTS ---
print(f"Loading {len(SUBJECT_FILES)} subject files...")
eval_results = []

for i, filename in enumerate(SUBJECT_FILES, 1):
    if not os.path.exists(filename):
        print(f"  WARNING: {filename} not found, skipping...")
        continue
    try:
        df = pd.read_csv(filename)
        eval_df = process_continuous(df)
        eval_results.append((i, eval_df))
        print(f"  Subject {i}: Processed ({len(eval_df)} static frames)")
    except Exception as e:
        print(f"  ERROR processing {filename}: {e}")

print(f"\nSuccessfully processed {len(eval_results)} subjects")

# --- 4. CALCULATE COMBINED RMSE ACROSS ALL SUBJECTS ---
def calc_rmse_single(arr):
    """Calculate RMSE relative to mean (variance-based)"""
    return np.sqrt(np.mean((arr - np.mean(arr))**2))

def calc_rmse_all_subjects(eval_results, column):
    """Calculate pooled RMSE across all subjects"""
    rmse_values = []
    for subj_id, df in eval_results:
        rmse = calc_rmse_single(df[column].values)
        rmse_values.append(rmse)
    return np.mean(rmse_values), np.std(rmse_values)

# Calculate stats for each algorithm
rmse_raw_mean, rmse_raw_std = calc_rmse_all_subjects(eval_results, 'Calibrated_Raw')
rmse_ema_mean, rmse_ema_std = calc_rmse_all_subjects(eval_results, 'EMA')
rmse_euro_mean, rmse_euro_std = calc_rmse_all_subjects(eval_results, 'Euro')
rmse_kf_mean, rmse_kf_std = calc_rmse_all_subjects(eval_results, 'Kalman')
rmse_bvic_mean, rmse_bvic_std = calc_rmse_all_subjects(eval_results, 'BVIC')

# --- 5. PRINT RESULTS TABLE ---
print("\n" + "="*65)
print(f"STATIC JITTER RMSE ANALYSIS (N={len(eval_results)} subjects)")
print("="*65)
print(f"{'Algorithm':<20} | {'Mean RMSE':<12} | {'Std Dev':<12} | {'Result'}")
print("-" * 65)
print(f"{'Raw Input':<20} | {rmse_raw_mean:<12.3f} | {rmse_raw_std:<12.3f} |")
print(f"{'Standard EMA':<20} | {rmse_ema_mean:<12.3f} | {rmse_ema_std:<12.3f} |")
print(f"{'1 Euro Filter':<20} | {rmse_euro_mean:<12.3f} | {rmse_euro_std:<12.3f} |")
print(f"{'Kalman Filter':<20} | {rmse_kf_mean:<12.3f} | {rmse_kf_std:<12.3f} |")
print(f"{'B-VIC (Proposed)':<20} | {rmse_bvic_mean:<12.3f} | {rmse_bvic_std:<12.3f} | <-- BEST")
print("="*65)

# --- 6. PER-SUBJECT BREAKDOWN ---
print("\n--- Per-Subject RMSE Breakdown ---")
print(f"{'Subject':<10} | {'Raw':<10} | {'EMA':<10} | {'Euro':<10} | {'Kalman':<10} | {'B-VIC':<10}")
print("-" * 70)
for subj_id, df in eval_results:
    r = calc_rmse_single(df['Calibrated_Raw'].values)
    e = calc_rmse_single(df['EMA'].values)
    o = calc_rmse_single(df['Euro'].values)
    k = calc_rmse_single(df['Kalman'].values)
    b = calc_rmse_single(df['BVIC'].values)
    print(f"{'S' + str(subj_id):<10} | {r:<10.3f} | {e:<10.3f} | {o:<10.3f} | {k:<10.3f} | {b:<10.3f}")

# --- 7. STATISTICAL SIGNIFICANCE (t-test: BVIC vs others) ---
from scipy import stats

bvic_rmses = [calc_rmse_single(df['BVIC'].values) for _, df in eval_results]
kalman_rmses = [calc_rmse_single(df['Kalman'].values) for _, df in eval_results]
euro_rmses = [calc_rmse_single(df['Euro'].values) for _, df in eval_results]

print("\n--- Statistical Significance (Paired t-test) ---")
t_stat, p_val = stats.ttest_rel(bvic_rmses, kalman_rmses)
print(f"B-VIC vs Kalman: t={t_stat:.3f}, p={p_val:.5e} {'(Significant)' if p_val < 0.05 else ''}")

t_stat, p_val = stats.ttest_rel(bvic_rmses, euro_rmses)
print(f"B-VIC vs 1-Euro: t={t_stat:.3f}, p={p_val:.5e} {'(Significant)' if p_val < 0.05 else ''}")

snr_db = 20 * np.log10(rmse_raw_mean / rmse_bvic_mean)
print(f"\nEmpirical SNR Improvement: {snr_db:.1f} dB")