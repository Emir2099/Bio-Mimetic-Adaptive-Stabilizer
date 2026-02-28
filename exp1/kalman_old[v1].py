import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- 1. KALMAN FILTER IMPLEMENTATION ---
class SimpleKalman:
    def __init__(self, process_noise=1e-5, sensor_noise=1e-1, estimated_error=1.0, initial_value=0):
        self.q = process_noise # Process noise covariance
        self.r = sensor_noise  # Measurement noise covariance
        self.p = estimated_error # Estimation error covariance
        self.x = initial_value # Value
        self.k = 0 # Kalman gain

    def update(self, measurement):
        # Prediction update
        self.p = self.p + self.q

        # Measurement update
        self.k = self.p / (self.p + self.r)
        self.x = self.x + self.k * (measurement - self.x)
        self.p = (1 - self.k) * self.p
        return self.x

# --- 2. LOAD DATA ---
try:
    df = pd.read_csv('experiment_data.csv')
    raw_signal = df['Raw_Input'].values
    timestamps = df['Timestamp_s'].values
except FileNotFoundError:
    print("Error: 'experiment_data.csv' not found. Please run gen.py and record data first.")
    exit()

# --- 3. MONTE CARLO SIMULATION (N=1000) ---
n_trials = 1000  # UPDATED: Increased from 20 to 1000 as per guide's request
rmse_results = {'EMA': [], 'OneEuro': [], 'Kalman': [], 'BVIC': []}

print(f"--- Running {n_trials} Monte Carlo Simulations (This may take a moment)... ---")

# Seed for reproducibility (Guide requested "Explicitly state seed")
np.random.seed(42) 

for i in range(n_trials):
    # Add Gaussian thermal noise to simulate repeated trials
    # Sigma=0.5 matches your paper description
    synthetic_noise = np.random.normal(0, 0.5, size=len(raw_signal)) 
    trial_input = raw_signal + synthetic_noise
    
    # Run Filters
    kf = SimpleKalman(process_noise=0.1, sensor_noise=5.0)
    
    val_ema = 0
    val_bvic = 0
    
    trial_ema = []
    trial_kf = []
    trial_bvic = []
    
    for t, val in enumerate(trial_input):
        # EMA
        val_ema = val_ema + 0.05 * (val - val_ema)
        trial_ema.append(val_ema)
        
        # Kalman
        trial_kf.append(kf.update(val))
        
        # BVIC LOGIC (Matches your paper/Unity script)
        # Threshold scaled because CSV data is x1000 (0.15 * 1000 = 150)
        energy = abs(val) 
        alpha = 0.02 if energy < 150.0 else 0.6 
        val_bvic = val_bvic + alpha * (val - val_bvic)
        trial_bvic.append(val_bvic)
    
    # Calculate Static RMSE (First 30% of data assumed static)
    limit = int(len(trial_input) * 0.3)
    
    rmse_results['EMA'].append(np.sqrt(np.mean(np.array(trial_ema[:limit])**2)))
    rmse_results['Kalman'].append(np.sqrt(np.mean(np.array(trial_kf[:limit])**2)))
    rmse_results['BVIC'].append(np.sqrt(np.mean(np.array(trial_bvic[:limit])**2)))

# --- 4. PRINT STATS FOR LATEX TABLE (Mean, SD, 95% CI) ---
print("\n" + "="*60)
print(f"{'Metric':<15} | {'Mean':<10} | {'SD':<10} | {'95% CI [Lower, Upper]'}")
print("-" * 60)

stats_summary = {}

for method, values in rmse_results.items():
    if len(values) > 0:
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Calculate 95% Confidence Interval
        # CI = Mean +/- (1.96 * Standard Error)
        std_error = std_val / np.sqrt(n_trials)
        ci_lower = mean_val - (1.96 * std_error)
        ci_upper = mean_val + (1.96 * std_error)
        
        stats_summary[method] = (mean_val, std_val, ci_lower, ci_upper)
        
        print(f"{method:<15} | {mean_val:<10.3f} | {std_val:<10.3f} | [{ci_lower:.3f}, {ci_upper:.3f}]")

# T-Test (BVIC vs Kalman)
t_stat, p_val = stats.ttest_rel(rmse_results['BVIC'], rmse_results['Kalman'])
print("-" * 60)
print(f"Paired t-test (BVIC vs Kalman): p-value = {p_val:.5e}")
if p_val < 0.001:
    print("RESULT: Statistically Significant (p < 0.001)")
print("="*60 + "\n")


# --- 5. TREMOR FREQUENCY ANALYSIS (Requested by Guide) ---
print("--- Running Frequency Characterization (4-6Hz vs 8-12Hz) ---")
# Simulate pure sine waves at different frequencies to prove robustness
frequencies = [5, 10] # 5Hz (Parkinsonian), 10Hz (Physiological)
fs = 60 # Sampling rate
duration = 2.0
t = np.linspace(0, duration, int(fs*duration))

for freq in frequencies:
    # Generate synthetic tremor
    tremor_signal = 100 * np.sin(2 * np.pi * freq * t) # Amplitude 100
    
    # Run B-VIC on it
    bvic_out = []
    val_bvic = 0
    for val in tremor_signal:
        energy = abs(val)
        alpha = 0.02 if energy < 150.0 else 0.6
        val_bvic = val_bvic + alpha * (val - val_bvic)
        bvic_out.append(val_bvic)
        
    rmse = np.sqrt(np.mean(np.array(bvic_out)**2))
    attenuation_db = 20 * np.log10(np.std(tremor_signal) / np.std(bvic_out))
    
    print(f"Frequency {freq}Hz: RMSE = {rmse:.2f} | Attenuation = {attenuation_db:.1f} dB")


# --- 6. GENERATE KALMAN GRAPH ---
kf_clean = SimpleKalman(process_noise=0.1, sensor_noise=5.0)
kf_output = [kf_clean.update(x) for x in raw_signal]

plt.figure(figsize=(10, 5))
plt.plot(timestamps, raw_signal, label='Raw Input', color='lightgray', alpha=0.5)
plt.plot(timestamps, kf_output, label='Kalman Filter', color='blue', linestyle='--')
plt.plot(timestamps, df['BVIC_Output'], label='B-VIC', color='green', linewidth=2)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Angular Velocity (deg/s)', fontsize=12)
plt.legend()
plt.title("B-VIC vs Kalman Filter (Baseline)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("kalman_comparison.png", dpi=300)
print("\nGraph saved: kalman_comparison.png")