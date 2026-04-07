# Bio-Mimetic Adaptive Stabilizer (B-VIC)

Bio-Mimetic Variable Impedance Control (B-VIC) for real-time stabilization of gyroscope motion signals, with a full experimental pipeline across 10 subjects.

## Overview

This repository contains:

- Real-time B-VIC demos connected to phone gyro streams
- A data logger for collecting trial CSV files
- Multi-subject evaluation scripts comparing B-VIC against EMA, 1-Euro, and Kalman filters
- Figure generation scripts for static jitter suppression and dynamic response analysis

The latest implementation is based on continuous processing with:

- Bias calibration from low-variance windows
- Hysteresis-based mode switching for B-VIC
- Zero-equilibrium static behavior to reduce drift

## Repository Layout

```
Bio-Mimetic Adaptive Stabilizer/
└── exp1/
        ├── gen.py
        ├── graph.py
        ├── kalman.py
        ├── latency_analysis.py
        ├── rmse_comparison.py
        ├── subject_1_data.csv
        ├── subject_2_data.csv
        ├── subject_3_data.csv
        ├── subject_4_data.csv
        ├── subject_5_data.csv
        ├── subject_6_data.csv
        ├── subject_7_data.csv
        ├── subject_8_data.csv
        ├── subject_9_data.csv
        └── subject_10_data.csv
```

## What Each Script Does

- `exp1/gen.py`: Data logger (press `R` to start/stop recording) writing experiment CSV format.
- `exp1/kalman.py`: Multi-subject RMSE and paired t-test analysis (N=10).
- `exp1/graph.py`: Multi-subject figure generation (all-subject and summary plots).

## Requirements

Install dependencies:

```bash
pip install pygame requests numpy pandas matplotlib scipy
```

## Phone Streaming Setup (for `updated.py` / `gen.py`)

1. Start your phone sensor HTTP stream app.
2. Ensure phone and PC are on the same network.
3. Update `PHONE_URL` in scripts that fetch live data.

Example:

```python
PHONE_URL = "http://YOUR_PHONE_IP:8080"
```

## Running Real-Time Demos

From repository root:

```bash
python updated.py
```

## Data Collection Workflow

From `exp1`:

```bash
python gen.py
```

Inside the Pygame window:

- Press `R` to start recording
- Press `R` again to stop recording

Recorded CSV columns:

- `Timestamp_s`
- `Raw_Input`
- `Standard_EMA`
- `OneEuro`
- `BVIC_Output`
- `Stiffness_Alpha`
- `Mode_State`

## Multi-Subject Analysis (Current Implementation)

### 1) Statistical Evaluation

From `exp1`:

```bash
python kalman.py
```

Current `kalman.py` behavior:

- Loads all 10 files: `subject_1_data.csv` ... `subject_10_data.csv`
- Performs per-subject calibration and continuous filtering
- Computes static-window RMSE statistics for:
    - Raw input
    - Standard EMA
    - 1-Euro
    - Kalman
    - B-VIC
- Prints:
    - Mean RMSE ± Std Dev across subjects
    - Per-subject breakdown table
    - Paired t-tests (`B-VIC vs Kalman`, `B-VIC vs 1-Euro`)
    - Empirical SNR improvement

### 2) Figure Generation

From `exp1`:

```bash
python graph.py
```

Current `graph.py` outputs:

- `empirical_validation_all_subjects.png`
    - Full grid: static and dynamic windows for all subjects
- `empirical_validation_summary.png`
    - Compact summary for representative subjects
- `static_comparison_grid.png`
    - Static-only comparison across all subjects

## Core B-VIC Logic (Summary)

B-VIC switches between static and dynamic behavior based on motion error and hysteresis:

- Dynamic mode entry requires consecutive threshold crossings
- Static mode uses decay + low alpha to pull toward zero while tracking fine tremor

Typical parameters used in analysis scripts:

- `THRESHOLD = 150.0`
- `CONFIRM_FRAMES = 3`
- `ALPHA_DYNAMIC = 0.60`
- `DECAY_STATIC = 0.95`
- `ALPHA_STATIC = 0.02`

## Notes

- Run analysis scripts from the `exp1` directory so relative file paths resolve correctly.
- If a subject file is missing, scripts will print a warning and skip it.
- For reproducible publication figures, keep subject CSV schema unchanged.

## License

This repository is currently intended for academic and research use.
