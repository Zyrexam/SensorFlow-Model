# SensorFlow-Model

**Federated Learning for Human Activity Recognition from Wearable Sensors**

A multi-algorithm federated learning system that recognizes 12 human activities using dual-sensor data (smartwatch + earable). Built with [Flower](https://flower.ai/) and TensorFlow/Keras. Compares **FedAvg**, **FedProx**, **FedPer**, **FedRep**, and **ClusterFL** across 6 real-world clients.

---

## Key Results

| Algorithm | Rounds | Accuracy | Loss (start → end) |
|-----------|--------|----------|-------------------|
| **FedProx** (μ=0.01) | 10 | **99.6%** | 0.92 → 0.20 |
| **FedAvg** | 10 | 97.6% | 0.29 → 0.09 |
| **FedAvg** | 20 | 98.1% | 0.36 → 0.07 |
| **FedAvg** | 30 | 98.4% | 0.26 → 0.06 |
| **FedAvg** | 40 | **98.8%** | 0.48 → 0.05 |
| **ClusterFL** (k=2) | 10 | 93.5% | 0.52 → 0.19 |
| **FedRep** (Train) | 10 | ~99.8% | — |
| **FedRep** (Eval) | 20 | 78.5% | — |
| **FedPer** | 50 | 88.3% | 2.10 → 0.35 |

---

## Dataset

**6 users**, each with sensor recordings in separate `Client_X` folders. Data comes from two devices:

| Sensor | Features |
|--------|----------|
| **Smartwatch** | Gyroscope (x,y,z) + Accelerometer (x,y,z) = 6 |
| **Earable** | Gyroscope (x,y,z) + Accelerometer (x,y,z) = 6 |

Input: sliding windows of **20 timesteps** (stride 5) → shape `(20, 12)`. Z-score normalized per sensor.

### Activity Classes (12 total)

0. Sitting + Typing on Desk
1. Sitting + Taking Notes
2. Standing + Writing on Whiteboard
3. Standing + Erasing Whiteboard
4. Sitting + Talking + Waving Hands
5. Standing + Talking + Waving Hands
6. Sitting + Drinking Water
7. Sitting + Drinking Coffee
8. Standing + Drinking Water
9. Standing + Drinking Coffee
10. Sitting + Scrolling on Phone
11. Scrolling on Phone

---

## FL Algorithms

| Algorithm | Strategy | Accuracy |
|-----------|----------|----------|
| **FedAvg** | Weighted averaging of client updates | 98.8% (40 rds) |
| **FedProx** | FedAvg + proximal term to penalize local drift (μ=0.01) | **99.6%** (10 rds) |
| **FedPer** | Personalized head layers (PyTorch ANN) | 88.3% (50 rds) |
| **FedRep** | Alternate backbone/head freezing rounds | 78.5% eval (20 rds) |
| **ClusterFL** | K-Means client clustering + per-cluster aggregation | 93.5% (10 rds) |

---

## Model Architecture

**Active model** — Combined-Input Conv1D:
```
Input (20,12) → Conv1D(32,k=3) → MaxPool → Dropout(0.2)
             → Conv1D(64,k=3) → MaxPool → Dropout(0.2)
             → Flatten → Dense(64) → Dropout(0.3) → Dense(12, Softmax)
```

**Alternate** — Dual-branch CNN + GatedSensorFusion + BiLSTM (commented out in `task.py`).

---

## Few-Shot Personalization

Adapts the global model per user. Samples K examples/class (K=5,10,20), fine-tunes 3 epochs, measures Acc/F1/Recall before and after. Triggered via `evaluate_personalization_on_clients()` in `personalize.py`.

---

## Installation

```bash
pip install -e .
```

## Usage

```bash
flwr run .      # Run FL simulation (6 clients, 10 rounds)
```

Edit `pyproject.toml` to change rounds, epochs, or batch size.

**Switch algorithms**: uncomment the desired section in `server_app.py` + `client_app.py` and comment out the active ClusterFL section.

---

## Project Structure

```
SensorFlow-Model/
├── pyproject.toml               # Config & dependencies
├── final_model.keras/.tflite    # Trained models
├── convergence_*.json/.png      # Training history & plots
├── myData/Client_1..6/          # Per-user sensor data
├── TestData/                    # Held-out test data
└── sensorflow_model/
    ├── server_app.py            # Server strategy (ClusterFL active)
    ├── client_app.py            # Client training logic
    ├── task.py                  # Model & data loading
    ├── dataset.py               # CSV → sliding windows
    ├── personalize.py           # Few-shot personalization
    ├── testing.ipynb            # Evaluation notebook
    └── Research_Pipeline.ipynb  # t-SNE & analysis
```
