import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the data
df = pd.read_csv('experiment_data.csv')

# 2. Define Time Windows
# Static Phase: A stable region before the movement (e.g., 1.0s to 3.0s)
static_df = df[(df['Timestamp_s'] > 1.0) & (df['Timestamp_s'] < 3.0)]

# Dynamic Phase: The specific moment of the "Flick" (e.g., 4.8s to 5.5s)
dynamic_df = df[(df['Timestamp_s'] > 4.8) & (df['Timestamp_s'] < 5.5)]

# --- GRAPH 1: STATIC STABILITY ---
plt.figure(figsize=(10, 5))

# Plot Lines
plt.plot(static_df['Timestamp_s'], static_df['Raw_Input'], label='Raw Input', color='lightgray', alpha=0.7)
plt.plot(static_df['Timestamp_s'], static_df['Standard_EMA'], label='Standard EMA', color='red', linewidth=1.5)
plt.plot(static_df['Timestamp_s'], static_df['OneEuro'], label='1 Euro Filter', color='orange', linewidth=1.5)
plt.plot(static_df['Timestamp_s'], static_df['BVIC_Output'], label='B-VIC', color='green', linewidth=2.5) # Thicker green line

# Styling
plt.title('Static Stability Test (Tremor Suppression)', fontsize=14)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Angular Velocity (deg/s)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save
plt.savefig('static_stability_clean.png', dpi=300)
print("Saved: static_stability_clean.png")

# --- GRAPH 2: DYNAMIC RESPONSE ---
plt.figure(figsize=(10, 5))

# Plot Lines
plt.plot(dynamic_df['Timestamp_s'], dynamic_df['Raw_Input'], label='Raw Input', color='lightgray', linewidth=4, alpha=0.6)
plt.plot(dynamic_df['Timestamp_s'], dynamic_df['Standard_EMA'], label='Standard EMA', color='red', linestyle='--', linewidth=1.5)
plt.plot(dynamic_df['Timestamp_s'], dynamic_df['OneEuro'], label='1 Euro Filter', color='orange', linewidth=1.5)
plt.plot(dynamic_df['Timestamp_s'], dynamic_df['BVIC_Output'], label='B-VIC', color='green', linewidth=2.5)

# Styling
plt.title('Dynamic Response Test (Latency Analysis)', fontsize=14)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Angular Velocity (deg/s)', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save
plt.savefig('dynamic_response_clean.png', dpi=300)
print("Saved: dynamic_response_clean.png")

plt.show()