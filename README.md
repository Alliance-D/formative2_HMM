# Human Activity Recognition Using Hidden Markov Models

**Formative 2 | Group Members:** Alliance (Google Pixel 5) & Emmanuel (Samsung S21)

---

## Project Overview

This project builds a Hidden Markov Model (HMM) pipeline to recognise four human activities — **Still, Standing, Walking, and Jumping** — from smartphone accelerometer and gyroscope signals collected using the Sensor Logger app.

---

## Repository Structure

```
├── data/
│   ├── train/
│   │   ├── alliance/
│   │   │   ├── still/
│   │   │   ├── standing/
│   │   │   ├── walking/
│   │   │   └── jumping/
│   │   └── emmanuel/
│   │       ├── Still/
│   │       ├── Standing/
│   │       ├── Walking/
│   │       └── Jumping/
│   └── test/
│       ├── alliance/
│       │   ├── still/
│       │   ├── standing/
│       │   ├── walking/
│       │   └── jumping/
│       └── emmanuel/
│           ├── Still/
│           ├── Standing/
│           ├── Walking/
│           └── Jumping/
├── notebook.ipynb       ← Main implementation notebook
├── report.pdf           ← Project report
├── README.md
```

Each activity folder contains timestamped session subfolders (e.g. `2026-03-05_19-57-38/`) with:
- `Accelerometer.csv` — x, y, z acceleration (m/s²)
- `Gyroscope.csv` — x, y, z angular velocity (rad/s)

---

## Dataset Summary

| Split | Alliance | Emmanuel | Total |
|-------|----------|----------|-------|
| Train | 24 recordings (6 per activity) | 26 recordings (6–7 per activity) | **50** |
| Test  | 12 recordings (3 per activity) | 12 recordings (3 per activity) | **24** |

- **Devices:** Alliance — Google Pixel 5 (~53 Hz) | Emmanuel — Samsung S21 (~100 Hz)
- **Target sampling rate:** 50 Hz (resampled from both devices via linear interpolation)
- **Recording duration:** 9–10 seconds per sample

---

## Pipeline Summary

```
Raw CSVs → Merge Accel + Gyro → Resample to 50 Hz
        → Sliding Window (1s, 50% overlap)
        → Extract 44 Features (time + frequency domain)
        → Z-score Normalise
        → Train Gaussian HMM (Baum-Welch, ε = 1e-4)
        → Decode with Viterbi
        → Evaluate on unseen test data
```

---

## Features Extracted (44 total)

| Domain | Features | Count |
|--------|----------|-------|
| Time | Mean, Variance, Std, RMS per axis (6 axes) | 24 |
| Time | Signal Magnitude Area (accel + gyro) | 2 |
| Time | Cross-axis correlations (6 pairs) | 6 |
| Frequency (FFT) | Dominant frequency per axis | 6 |
| Frequency (FFT) | Spectral energy per axis | 6 |
| **Total** | | **44** |

---

## How to Run

### Requirements

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install hmmlearn scikit-learn pandas numpy matplotlib seaborn scipy jupyter
```

### Run the Notebook

```bash
jupyter notebook notebook.ipynb
```

Or open directly in VS Code with the Jupyter extension — select the `venv` kernel.

### Expected Output Files

After running all cells the notebook saves:
- `raw_signals.png` — raw sensor signals per activity
- `convergence.png` — Baum-Welch training convergence curve
- `transition_matrix.png` — learned transition probability heatmap
- `emission_means.png` — emission means per state (first 18 features)
- `decoded_sequence.png` — true vs Viterbi decoded labels on training set
- `confusion_matrix.png` — test set confusion matrix
- `feature_distributions.png` — feature distributions per activity

---

## Task Allocation

| Task | Owner |
|------|-------|
| Data collection – Alliance's training & test samples | Alliance |
| Data collection – Emmanuel's training & test samples | Emmanuel |
| Sections 0–2: Data loading, preprocessing, feature extraction | Alliance |
| Sections 3–4: HMM setup, Baum-Welch training, Viterbi, visualisations | Emmanuel |
| Sections 5–6: Evaluation, analysis | Alliance |
| Report & README | Alliance |

---

## Results

Evaluated on **464 unseen test windows** from 24 recordings collected in a separate session.

| Activity | No. of Windows | Sensitivity | Specificity | Accuracy |
|----------|---------------|-------------|-------------|----------|
| Still    | 118           | 0.793       | 0.859       | 0.80     |
| Standing | 115           | 0.122       | 0.980       | 0.12     |
| Walking  | 116           | 0.609       | 0.742       | 0.79     |
| Jumping  | 115           | 0.797       | 0.861       | 0.61     |
| **Overall** | **464**    | —           | —           | **0.582** |

**Key finding:** Still and Jumping were best recognised. Standing was the hardest activity to classify, its low-variance gyroscope signal closely resembles Walking, causing frequent misclassification between the two.