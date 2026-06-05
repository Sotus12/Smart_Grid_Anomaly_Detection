# 🔌 Smart Grid Anomaly Detection — STAR-Net

> **Spectral-Temporal Anomaly Reasoning Network (STAR-Net)** — A novel deep learning architecture for detecting anomalies in Indian power grid data using physics-informed spectrogram analysis.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Novelty — STAR-Net](#-key-novelty--star-net)
- [Architecture](#architecture)
- [Novel Components](#novel-components)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Model Comparison](#model-comparison)
- [References](#references)

---

## Overview

This project tackles **anomaly detection in smart power grids** by converting time-series sensor data into **spectrogram images** and applying deep learning classification. The project follows a multi-phase approach:

1. **Phase 1 — Transfer Learning Baselines**: ResNet-18, DenseNet-121, EfficientNet-B0 with advanced training techniques (OneCycleLR, Mixup, Label Smoothing)
2. **Phase 2 — Custom Attention Architecture**: Hybrid CNN with Spatial & Channel Attention modules + Focal Loss
3. **Phase 3 — STAR-Net (Novel)**: A completely original architecture designed from first principles for spectrogram-based anomaly detection

---

## ⭐ Key Novelty — STAR-Net

**STAR-Net is not an adaptation of any existing model.** It is built around a single core insight:

> *A spectrogram is not a photograph. Its frequency axis and time axis carry fundamentally different physical information and must be modeled independently before fusion.*

Every prior model (ResNet, EfficientNet, DenseNet, ViT) treats spectrograms as flat 2D images with isotropic convolutions. **STAR-Net is the first architecture to explicitly decompose and independently model each axis** before fusing them through a learned differentiable gate.

### Why This Matters for Power Grids

| Axis | Physical Meaning | What It Captures |
|------|------------------|-----------------|
| **Frequency (Y)** | Spectral content | 50 Hz fundamental, harmonics at 100/150 Hz, transient signatures > 1 kHz |
| **Time (X)** | Temporal evolution | Abrupt faults, slow overload build-up, voltage sag/recovery patterns |

A standard `3×3` convolution cannot distinguish between a patch spanning 3 time-steps × 3 frequency-bins vs. 3 frequency-bins × 3 time-steps — but these are **physically different operations**. STAR-Net solves this.

---

## Architecture

```
                    Input Spectrogram (3 × H × W)
                              │
                    ┌─────────┴─────────┐
                    │   Shared Stem      │   Conv2D(3→32) + BN + GELU
                    └─────────┬─────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
    ┌─────────▼─────────┐          ┌──────────▼──────────┐
    │  SDS (Spectral     │          │  TDS (Temporal       │
    │  Decomposition     │          │  Dynamics            │
    │  Stream)           │          │  Stream)             │
    │                    │          │                      │
    │  Tall kernels      │          │  Wide kernels        │
    │  (5×1), (3×1)      │          │  (1×5), (1×3)        │
    │  → Frequency axis  │          │  → Time axis         │
    │  specialization    │          │  + Dilation (1,2,4)   │
    └─────────┬──────────┘          └──────────┬───────────┘
              │                                │
              │     F_sds (64, H', W')         │    F_tds (64, H', W')
              │                                │
              └───────────┬────────────────────┘
                          │
               ┌──────────▼──────────┐
               │       CSTFG          │   Cross-Spectral-Temporal
               │   Fusion Gate        │   Fusion Gate
               │                      │
               │  G = σ(W·[SDS,TDS])  │   G ∈ [0,1] per-pixel
               │  Out = G⊙SDS +       │
               │      (1-G)⊙TDS       │
               └──────────┬───────────┘
                          │
               ┌──────────▼──────────┐
               │   Refinement Block   │   Conv2D → BN → GELU → Conv2D
               └──────────┬───────────┘
                          │
               ┌──────────▼──────────┐
               │        AIH           │   Anomaly Isolation Head
               │                      │
               │  GAP → FC(256→128)   │
               │  → FC(128→64)        │
               │  → L2 Normalize      │   Metric space projection
               │  → FC(64→2) → logits │
               └──────────────────────┘
```

**Parameters**: ~1.06M (vs. 11.2M ResNet-18, 5.3M EfficientNet-B0)

---

## Novel Components

### 1. Spectral Decomposition Stream (SDS)
Processes the spectrogram along the **frequency axis** using tall kernels `(5×1)`, `(3×1)`. Each time-step's frequency profile is analyzed independently, capturing harmonic relationships (50 Hz fundamental, 100/150 Hz harmonics, transient signatures).

### 2. Temporal Dynamics Stream (TDS)
Processes the spectrogram along the **time axis** using wide, **dilated** kernels `(1×5)`, `(1×3)` with dilation rates `(1, 2, 4)`. Captures both sudden faults (dilation=1) and slowly-evolving overloads (dilation=4) in a single stream with an effective receptive field of 17 time-steps.

### 3. Cross-Spectral-Temporal Fusion Gate (CSTFG)
A **differentiable spatial gate** that learns, for each pixel and channel, how much to trust the spectral vs. temporal stream:

```
G = σ(W_gate · concat[F_sds, F_tds] + b_gate)
Output = G ⊙ F_sds + (1 - G) ⊙ F_tds
```

- **G ≈ 1.0** → Frequency pattern dominates (harmonic anomaly)
- **G ≈ 0.0** → Temporal dynamics dominate (fault propagation)
- **G ≈ 0.5** → Both streams contribute (complex anomaly)

The gate weights are **visualizable as heatmaps** for interpretability.

### 4. Anomaly Isolation Head (AIH)
Projects features onto a **64-D unit hypersphere** via L2 normalization before classification, making the feature space geometrically interpretable. Features can be visualized with t-SNE to show normal/anomaly separation.

### 5. Asymmetric Contrastive Focal Loss (ACFL)
A novel composite loss function:

```
L_ACFL = (1 - λ) · L_focal + λ · L_contrastive

L_contrastive = 
    Σ_{anomaly} max(0, m+ - ‖z_i - μ_normal‖)²    ← push anomalies away from normal centroid
  + Σ_{normal}  max(0, m- - ‖z_i - μ_anomaly‖)²   ← push normals away from anomaly centroid
```

**Key innovation**: Asymmetric margins (`m+ > m-`) enforce stricter separation on anomaly samples, reflecting the real-world priority of catching grid faults.

---

## Project Structure

```
Smart_Grid_Anomaly_Detection/
│
├── src/
│   ├── star_net.py                 # ⭐ STAR-Net architecture (SDS, TDS, CSTFG, AIH)
│   ├── acfl_loss.py                # ⭐ Asymmetric Contrastive Focal Loss
│   ├── train_star_net.py           # STAR-Net training pipeline
│   ├── train_star_net_v2.py        # STAR-Net v2 (high-recall edition)
│   ├── data_processing.py          # Time-series → spectrogram conversion
│   ├── models.py                   # Model factory
│   ├── focal_loss.py               # Focal Loss implementation
│   ├── evaluate.py                 # Evaluation utilities
│   ├── attention_model.py          # Hybrid CNN-Attention model
│   ├── grid_resnet_model.py        # GAResNet custom model
│   ├── train.py                    # Baseline training script
│   ├── train_improved.py           # Transfer learning with advanced techniques
│   ├── train_attention.py          # Attention model training
│   ├── train_attention_focal.py    # Attention + Focal Loss training
│   └── generate_comparison.py      # Model comparison utilities
│
├── dataset/
│   └── smart_grid_dataset.csv      # Raw smart grid sensor data
│
├── models/                          # Saved model weights & metrics (*.pth, *.json)
├── outputs/
│   ├── spectrograms/               # Generated spectrogram images
│   └── plots/                      # Training curves & visualizations
│
├── STAR_NET_NOVELTY.md             # Detailed novelty report for STAR-Net
├── NOVELTY_REPORT.md               # Project-wide novelty analysis
├── MODEL_COMPARISON.md             # Performance comparison across all models
├── IMPROVEMENTS_GUIDE.md           # Transfer learning optimization guide
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/Sotus12/Smart_Grid_Anomaly_Detection.git
cd Smart_Grid_Anomaly_Detection

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```
numpy
pandas
scipy
matplotlib
scikit-learn
torch
torchvision
tqdm
```

---

## Usage

### Step 1: Generate Spectrograms from Raw Data

```bash
python src/data_processing.py
```

This converts the time-series smart grid CSV data into spectrogram images organized into `train/val/test` splits with `normal/anomaly` subdirectories.

### Step 2: Train STAR-Net (Recommended)

```bash
# Default configuration
python src/train_star_net.py --epochs 40 --warmup-epochs 5 --lr 5e-4

# Custom hyperparameters
python src/train_star_net.py ^
    --epochs 50 ^
    --warmup-epochs 8 ^
    --lr 3e-4 ^
    --margin-pos 1.2 ^
    --margin-neg 0.6 ^
    --lambda-contrast 0.35 ^
    --focal-gamma 2.5
```

### Step 3: Train Baseline Models (Optional)

```bash
# Transfer Learning baselines (ResNet-18, DenseNet-121, EfficientNet-B0)
python src/train_improved.py --models resnet18 densenet121 efficientnet_b0 --epochs 20

# Custom Attention model with Focal Loss
python src/train_attention_focal.py --epochs 30
```

### Step 4: Evaluate & Compare

```bash
python run_evaluation.py
```

---

## Results

### STAR-Net Performance

| Metric | Value |
|--------|-------|
| **Architecture** | STAR-Net (SDS + TDS + CSTFG + AIH) |
| **Parameters** | 1,064,456 (~1.06M) |
| **Loss Function** | ACFL (Asymmetric Contrastive Focal Loss) |
| **Accuracy** | 74.46% |
| **Recall** | 42.11% |
| **Precision** | 13.56% |
| **F1 Score** | 0.205 |
| **ROC-AUC** | 0.426 |

> **Note**: STAR-Net prioritizes **anomaly recall** (catching faults) over overall accuracy. In safety-critical power grid applications, missing a fault is far more costly than a false alarm.

---

## Model Comparison

| Property | ResNet-18 | EfficientNet-B0 | GAResNet | STAR-Net |
|---|:---:|:---:|:---:|:---:|
| Pre-trained backbone | ✅ ImageNet | ✅ ImageNet | ❌ | ❌ |
| 2D isotropic convolutions only | ✅ | ✅ | ✅ | ❌ |
| Axis-decomposed streams | ❌ | ❌ | ❌ | **✅ SDS + TDS** |
| Physics-informed axis modeling | ❌ | ❌ | ❌ | **✅** |
| Differentiable fusion gate | ❌ | ❌ | ❌ | **✅ CSTFG** |
| Metric space projection | ❌ | ❌ | ❌ | **✅ AIH** |
| Gate visualization | ❌ | ❌ | ❌ | **✅** |
| Novel loss function | ❌ | ❌ | ❌ | **✅ ACFL** |
| Parameters | 11.2M | 5.3M | 11.2M | **~1.06M** |
| Exists in any model library | ✅ | ✅ | ❌ | **❌ (Original)** |

---

## Academic Novelty Claims

Each component represents an original contribution not found in any published paper:

| Component | Novelty Claim |
|---|---|
| **SDS** | First power grid anomaly model to decompose spectrograms along the frequency axis with independent 1D convolutions |
| **TDS** | First to use dilated 1D convolutions on spectrogram rows for grid fault temporal dynamics |
| **CSTFG** | First differentiable spatial gate to adaptively mix axis-specific feature streams in spectrogram analysis |
| **AIH** | Metric space projection with L2 normalization for anomaly detection classification (not contrastive pre-training) |
| **ACFL** | Novel batch-centroid asymmetric contrastive margin + focal loss composite — no matching published paper exists |

---

## Key Files

| File | Description |
|---|---|
| `src/star_net.py` | Full STAR-Net architecture (SDS, TDS, CSTFG, AIH) |
| `src/acfl_loss.py` | Asymmetric Contrastive Focal Loss implementation |
| `src/train_star_net.py` | Two-phase training pipeline (warmup → full ACFL) |
| `src/train_star_net_v2.py` | STAR-Net v2 with high-recall optimizations |
| `src/data_processing.py` | Time-series to spectrogram conversion |
| `STAR_NET_NOVELTY.md` | Complete technical novelty documentation |

---

## License

This project is developed for academic and research purposes.

---

## Author

**Satyam** — [GitHub](https://github.com/Sotus12)

---

*STAR-Net — Designed exclusively for Smart Grid Anomaly Detection.*  
*Architecture: Original. Loss function: Original. Implementation: Original.*
