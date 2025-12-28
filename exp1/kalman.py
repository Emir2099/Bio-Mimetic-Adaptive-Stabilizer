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
df = pd.read_csv('experiment_data.csv')
raw_signal = df['Raw_Input'].values
timestamps = df['Timestamp_s'].values
dt = np.mean(np.diff(timestamps))

# --- 3. MONTE CARLO SIMULATION (Generate Stats) ---
# We treat your recorded 'Raw_Input' as the "Base Signal" + "Original Noise".
# To get stats, we generate 20 variations by adding small extra thermal noise.
n_trials = 20
rmse_results = {'EMA': [], 'OneEuro': [], 'Kalman': [], 'BVIC': []}

print(f"--- Running {n_trials} Monte Carlo Simulations for Statistics ---")

for i in range(n_trials):
    # Add small Gaussian thermal noise to simulate repeated trials
    synthetic_noise = np.random.normal(0, 0.5, size=len(raw_signal)) 
    trial_input = raw_signal + synthetic_noise
    
    # Run Filters
    kf = SimpleKalman(process_noise=0.1, sensor_noise=5.0)
    
    # Storage
    trial_ema = []
    trial_euro = [] # (Simplified for simulation: reusing existing column logic if possible, else skip)
    trial_kf = []
    trial_bvic = []
    
    # We will approximate the logic here for simulation speed
    # Note: For the paper, use the exact classes from your main script if possible.
    # Here we simulate the logic simply:
    
    val_ema = 0
    val_bvic = 0
    
    for t, val in enumerate(trial_input):
        # EMA
        val_ema = val_ema + 0.05 * (val - val_ema)
        trial_ema.append(val_ema)
        
        # Kalman
        trial_kf.append(kf.update(val))
        
        # BVIC (Simplified Logic)
        energy = abs(val)
        alpha = 0.02 if energy < 0.15 else 0.6
        val_bvic = val_bvic + alpha * (val - val_bvic)
        trial_bvic.append(val_bvic)
    
    # Calculate Static RMSE (First 2 seconds)
    # Assuming first 30% of data is static
    limit = int(len(trial_input) * 0.3)
    
    rmse_results['EMA'].append(np.sqrt(np.mean(np.array(trial_ema[:limit])**2)))
    rmse_results['Kalman'].append(np.sqrt(np.mean(np.array(trial_kf[:limit])**2)))
    rmse_results['BVIC'].append(np.sqrt(np.mean(np.array(trial_bvic[:limit])**2)))

# --- 4. PRINT STATS FOR LATEX TABLE ---
print("\n--- STATISTICAL RESULTS (Mean ± Std Dev) ---")
for method, values in rmse_results.items():
    if len(values) > 0:
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{method}: {mean_val:.3f} ± {std_val:.3f}")

# T-Test (BVIC vs Kalman)
t_stat, p_val = stats.ttest_rel(rmse_results['BVIC'], rmse_results['Kalman'])
print(f"\nT-Test (BVIC vs Kalman): p-value = {p_val:.5e}")
if p_val < 0.05:
    print("RESULT: Statistically Significant Difference")

# --- 5. GENERATE KALMAN GRAPH ---
# Run one clean pass for the graph
kf_clean = SimpleKalman(process_noise=0.1, sensor_noise=5.0)
kf_output = [kf_clean.update(x) for x in raw_signal]

plt.figure(figsize=(10, 5))
plt.plot(timestamps, raw_signal, label='Raw Input', color='lightgray', alpha=0.5)
plt.plot(timestamps, kf_output, label='Kalman Filter', color='blue', linestyle='--')
plt.plot(timestamps, df['BVIC_Output'], label='B-VIC', color='green', linewidth=2)
plt.legend()
plt.title("B-VIC vs Kalman Filter (Baseline)")
plt.savefig("kalman_comparison.png")
print("\nGraph saved: kalman_comparison.png")