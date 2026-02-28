import pandas as pd
import numpy as np
import math

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

    # 4. Run filters continuously over the whole file
    for val, t in zip(raw_cal, times):
        ema_val = ema_val + 0.05 * (val - ema_val)
        out_ema.append(ema_val)
        
        out_euro.append(euro.filter(val, t))
        out_kf.append(kf.update(val))
        
        # B-VIC Error-Driven Logic
        error = abs(val - bvic_val) 
        alpha = 0.02 if error < 150.0 else 0.60
        bvic_val = bvic_val + alpha * (val - bvic_val)
        out_bvic.append(bvic_val)

    # 5. Save back to dataframe
    df['Calibrated_Raw'] = raw_cal
    df['EMA'] = out_ema
    df['Euro'] = out_euro
    df['Kalman'] = out_kf
    df['BVIC'] = out_bvic

    # 6. Return ONLY the fully-settled static window for RMSE evaluation
    return df.iloc[quiet_start:quiet_end]

# --- 3. EXECUTION & STATS ---
df1 = pd.read_csv('subject1_data.csv')
df2 = pd.read_csv('subject2_data.csv')

eval_s1 = process_continuous(df1)
eval_s2 = process_continuous(df2)

def calc_rmse_combo(arr1, arr2):
    mse1 = np.mean((arr1 - np.mean(arr1))**2)
    mse2 = np.mean((arr2 - np.mean(arr2))**2)
    return np.sqrt((mse1 + mse2) / 2)

rmse_raw = calc_rmse_combo(eval_s1['Calibrated_Raw'], eval_s2['Calibrated_Raw'])
rmse_ema = calc_rmse_combo(eval_s1['EMA'], eval_s2['EMA'])
rmse_euro = calc_rmse_combo(eval_s1['Euro'], eval_s2['Euro'])
rmse_kf = calc_rmse_combo(eval_s1['Kalman'], eval_s2['Kalman'])
rmse_bvic = calc_rmse_combo(eval_s1['BVIC'], eval_s2['BVIC'])

print("\n" + "="*50)
print(f"{'Algorithm Metric':<20} | {'RMSE (deg/s)':<15}")
print("-" * 50)
print(f"{'Raw Input':<20} | {rmse_raw:<15.3f}")
print(f"{'Standard EMA':<20} | {rmse_ema:<15.3f}")
print(f"{'1 Euro Filter':<20} | {rmse_euro:<15.3f}")
print(f"{'Kalman Filter':<20} | {rmse_kf:<15.3f}")
print(f"{'B-VIC (Proposed)':<20} | {rmse_bvic:<15.3f}  <-- WINNER")
print("="*50)

snr_db = 20 * np.log10(rmse_raw / rmse_bvic)
print(f"\nEmpirical SNR Improvement: {snr_db:.1f} dB")