# Bio-Mimetic Adaptive Stabilizer (B-VIC)

A novel bio-mimetic variable impedance control system for real-time motion stabilization, inspired by how the human nervous system dynamically adjusts muscle stiffness based on movement context.

## 🎯 Overview

This project implements **B-VIC (Bio-Mimetic Variable Impedance Control)**, an adaptive signal filtering algorithm that mimics biological motor control mechanisms. Unlike traditional fixed-parameter filters, B-VIC dynamically adjusts its smoothing behavior based on detected motion energy:

- **Low Energy (Stable/Idle)**: High stiffness for maximum tremor suppression
- **High Energy (Action/Movement)**: Low stiffness for responsive tracking

This creates a system that is simultaneously **stable during rest** and **responsive during intentional movement**.

## ✨ Key Features

- **Adaptive Filtering**: Real-time stiffness adjustment based on motion energy detection
- **Real-time Visualization**: Pygame-based display comparing raw input vs. filtered output
- **Phone Sensor Integration**: Connects to smartphone gyroscope via HTTP (using sensor streaming apps)
- **Academic Benchmarking**: Comparison against 1-Euro Filter and standard EMA
- **Data Logging**: CSV export for experimental analysis and paper figures

## 📁 Project Structure

```
Bio-Mimetic Adaptive Stabilizer/
│
├── updated.py          # Main application with real-time visualization
├── vr_demo.py          # Simplified demo showing core concept
├── vr_input.py         # Utility script for testing phone connectivity
│
└── exp1/               # Experiment & Analysis Tools
    ├── gen.py          # Data logger comparing B-VIC vs 1-Euro Filter
    ├── kalman.py       # Monte Carlo simulation with Kalman filter comparison
    ├── graph.py        # Publication-ready graph generator
    └── experiment_data.csv  # Sample experimental data
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install pygame requests numpy pandas matplotlib scipy
```

### Phone Sensor Setup

1. Install a sensor streaming app on your smartphone (e.g., "Sensor Server", "Phyphox", or similar)
2. Start the HTTP server on your phone
3. Update the `PHONE_URL` in the Python files to match your phone's IP address:
   ```python
   PHONE_URL = "http://YOUR_PHONE_IP:8080"
   ```
4. Ensure your phone and computer are on the same network

### Running the Application

**Main Visualization:**
```bash
python updated.py
```

**Simple Demo:**
```bash
python vr_demo.py
```

**Test Phone Connection:**
```bash
python vr_input.py
```

## 📊 Running Experiments

### Recording Data

1. Run the data logger:
   ```bash
   cd exp1
   python gen.py
   ```
2. Press **R** to start/stop recording
3. Data is saved to `experiment_data.csv`

### Generating Graphs

```bash
cd exp1
python graph.py
```

This generates publication-ready figures:
- `static_stability_clean.png` - Tremor suppression comparison
- `dynamic_response_clean.png` - Latency analysis

### Statistical Analysis

```bash
cd exp1
python kalman.py
```

Runs Monte Carlo simulations comparing B-VIC against:
- Standard EMA (Exponential Moving Average)
- 1-Euro Filter
- Kalman Filter

## ⚙️ Algorithm Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `STABLE_ALPHA` | 0.02 | Smoothing factor during stable/idle state |
| `ACTION_ALPHA` | 0.60 | Smoothing factor during action/movement |
| `ENERGY_THRESHOLD` | 0.15 | Motion energy threshold for state switching |
| `STANDARD_ALPHA` | 0.05 | Fixed alpha for comparison EMA filter |

## 🧬 The Bio-Mimetic Concept

The algorithm is inspired by the human neuromuscular system:

1. **Proprioceptive Sensing**: Detect current motion "energy" (magnitude)
2. **Impedance Modulation**: Adjust virtual "stiffness" based on context
3. **Smooth Transitions**: Gradual stiffness changes prevent discontinuities

```python
# Core Algorithm Logic
energy = abs(sensor_input)

if energy < ENERGY_THRESHOLD:
    target_alpha = STABLE_ALPHA    # High stiffness (tremor suppression)
    mode = "STABILIZING"
else:
    target_alpha = ACTION_ALPHA    # Low stiffness (responsive tracking)
    mode = "ACTION"

# Smooth stiffness transition
current_stiffness = lerp(current_stiffness, target_alpha, 0.1)

# Apply adaptive filter
output = lerp(previous_output, target, current_stiffness)
```

## 📈 Results

B-VIC demonstrates:
- **Superior static stability**: Lower RMSE during idle/tremor conditions
- **Better dynamic response**: Reduced latency compared to heavily smoothed filters
- **Adaptive behavior**: Automatic mode switching without manual tuning

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@article{bvic2026,
  title={Bio-Mimetic Variable Impedance Control for Real-Time Motion Stabilization},
  author={[Your Name]},
  journal={IEEE Access},
  year={2026}
}
```

## 📝 License

This project is for academic and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
