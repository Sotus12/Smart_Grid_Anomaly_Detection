# Smart Grid Spectrograms - Dataset Explanation

## What Are These Images?

These are **spectrograms** - visual representations of time-series power grid data converted into frequency-domain representations. Each image shows:

- **X-axis (Time)**: Progression from left (start of measurement window) to right (end)
- **Y-axis (Frequency)**: From bottom (low frequencies) to top (high frequencies)  
- **Color (Intensity)**: Brightness indicates strength of signal at that frequency and time

---

## Why Use Spectrograms?

Smart grid monitoring produces complex electrical signals containing multiple frequency components:

1. **Normal Operation**: Clean, stable frequency patterns with expected power line harmonics
2. **Anomalies**: Unexpected spikes, noise, frequency shifts indicating equipment faults

Converting time-series to spectrograms is beneficial because:
- ✓ Captures temporal evolution of frequency components
- ✓ Deep learning models (CNNs) work well on images
- ✓ Visual patterns are interpretable by humans
- ✓ Anomalies appear as distinct visual artifacts

---

## Image Properties

All spectrograms in this dataset:
- **Size**: 224 × 224 pixels (optimized for deep learning)
- **Format**: PNG grayscale images
- **Time window**: 10-60 second power grid measurements
- **Frequency range**: DC to ~5 kHz (covers dominant grid frequencies)
- **Colormapping**: Hot colormap (dark=low intensity, bright=high intensity)

---

## Normal Operations (19 samples)

Normal operation spectrograms show:

### Key Visual Characteristics:
1. **Fundamental Frequency**: Bright horizontal line at ~50 Hz (grid frequency in India)
2. **Harmonic Series**: Additional horizontal lines at 100Hz, 150Hz, 200Hz, etc. (multiples of 50Hz)
3. **Stable Patterns**: Consistency across time (left to right)
4. **Low Noise**: Relatively dark background with defined features
5. **Regular Structure**: Predictable, repeating patterns

### What This Indicates:
- ✓ Equipment operating normally
- ✓ Stable voltage and frequency
- ✓ No faults or anomalies
- ✓ Expected harmonic content from normal nonlinear loads

### Example Interpretation:
- Bright line at 50 Hz = Grid fundamental frequency (steady state)
- Fainter lines above = Harmonics from normal loads (transformers, motors, etc.)
- Constant intensity = Stable operation over time window

---

## Anomalies (20 samples)

Anomaly spectrograms show **distinct deviations** from normal patterns:

### Types of Anomalies in These Samples:

#### Type 1: Frequency Deviation
- **Visual**: Shift in the main frequency line (not at 50 Hz)
- **Cause**: Voltage sag/swell, frequency drift
- **Indication**: Grid operating outside safe frequency range
- **Severity**: High - indicates frequency instability

#### Type 2: Harmonic Distortion
- **Visual**: Unusually bright harmonic lines or new unexpected peaks
- **Cause**: Nonlinear loads (variable frequency drives, power electronics)
- **Indication**: Excessive harmonic injection into grid
- **Severity**: Medium - causes heating, interference

#### Type 3: Impulsive Noise/Spikes
- **Visual**: Bright vertical or irregular lines (broadband energy)
- **Cause**: Transient events, switching operations, faults
- **Indication**: Sudden disturbance on the grid
- **Severity**: High - potential equipment damage

#### Type 4: Phase Discontinuity
- **Visual**: Abrupt changes in spectral content or intensity
- **Cause**: Phase shifts, phase imbalances in three-phase system
- **Indication**: Asymmetrical loading or faults
- **Severity**: Medium - affects equipment, load imbalance

#### Type 5: Broadband Noise
- **Visual**: Elevated noise floor, hazy appearance
- **Cause**: EMI, grounding issues, loose connections
- **Indication**: Poor power quality
- **Severity**: Medium - cumulative effects on equipment

---

## How to Interpret a Spectrogram

### Visual Features to Look For:

**NORMAL (Safe to look for these):**
- Thin, bright horizontal line at 50 Hz (fundamental)
- Fainter lines at 100, 150, 200 Hz (harmonics)
- Mostly dark background (low noise)
- Consistent pattern across time
- Smooth variations

**ANOMALY (Alert flags):**
- ⚠️ Frequency deviation (line not at 50 Hz)
- ⚠️ Unusually bright harmonic lines
- ⚠️ Vertical spikes or transients
- ⚠️ Irregular, noisy patterns
- ⚠️ Sudden changes or discontinuities
- ⚠️ Unexpected frequency components
- ⚠️ Elevated noise floor

---

## Frequency Bands Explained

The spectrograms cover these important frequency ranges:

| Frequency Band | Label | What It Contains |
|---|---|---|
| 0-50 Hz | DC/Sub-harmonic | Grid frequency and sub-harmonics |
| 50 Hz | Fundamental | Main grid frequency (50 Hz in India) |
| 50-150 Hz | Harmonics | 100 Hz (2nd), 150 Hz (3rd) harmonics |
| 150-500 Hz | Higher Harmonics | 5th, 7th, 9th harmonics |
| 500+ Hz | Very High Freq | Transients, switching noise, EMI |

**What Grid Operators Monitor:**
- ✓ Fundamental frequency stability (must be ~50 Hz)
- ✓ Total Harmonic Distortion (THD) - sum of all harmonics
- ✓ Individual harmonic magnitudes (must be within limits)
- ✓ Presence of sub-harmonics or inter-harmonics (signs of failure)

---

## Time Dimension (X-Axis)

The left-to-right progression shows **temporal evolution**:

- **Left side**: Beginning of measurement window
- **Right side**: End of measurement window
- **Vertical consistency**: Stable frequency components over time
- **Vertical changes**: Dynamic events (transients, mode changes)

### Reading Time Patterns:
- **Flat lines**: Steady-state conditions (good for normal, bad for long anomalies)
- **Sloped lines**: Frequency changes over time (frequency ramps)
- **Breaks/jumps**: Sudden events (switching, faults)

---

## Dataset Characteristics

### Class Distribution:
- **Normal Operations**: 545 training samples, 19 test samples
- **Anomalies**: 89 training samples, 20 test samples
- **Ratio**: 83% normal vs 16% anomaly (extreme class imbalance)

### Why This Imbalance?
In real-world grid monitoring:
- Normal operation is ~95% of the time
- Anomalies are rare but critical to detect
- This imbalance makes anomaly detection challenging
- That's why we use Focal Loss and attention mechanisms!

### What Dataset Represents:
- Real Indian power grid data
- Multiple substations and monitoring points
- Various types of equipment and loads
- Different seasons and operational conditions
- Known anomalies (equipment faults, weather events, etc.)

---

## Visual Quality Notes

Some spectrograms may appear:
- **Very bright**: High-amplitude events (faults)
- **Very dark**: Low-amplitude or clean signals
- **Noisy**: Poor signal quality or high harmonic content
- **Clean**: Well-conditioned, stable measurements

This variation is **realistic** and reflects real-world grid data quality. Our deep learning model must handle all these variations.

---

## Interpretation Guide for Samples

### Normal Operations Samples:
Look for:
1. Bright horizontal line at 50 Hz (fundamental frequency)
2. Fainter lines at 100, 150 Hz (expected harmonics)
3. Mostly dark background (clean signal)
4. Consistent pattern when scanning left to right
5. No sudden changes or spikes

### Anomaly Samples:
Look for:
1. Line not at 50 Hz (frequency deviation)
2. Unexpectedly bright harmonics
3. Vertical spikes or transients
4. Noise or hazy background
5. Irregular or sudden changes
6. Frequency components outside expected range

---

## How Our AI Model Works With These

Our Hybrid CNN-Attention model processes these spectrograms by:

1. **Convolutional Layers**: Extract visual patterns (edges, textures)
2. **Spatial Attention**: Highlight WHERE anomalies appear (which time/frequency regions)
3. **Channel Attention**: Highlight WHAT patterns matter (which features)
4. **Classification**: Predict Normal vs Anomaly

This is why visual interpretation helps explain AI decisions - the model "sees" the same visual patterns you see!

---

## Real-World Applications

### Why Detect Anomalies?
1. **Prevent Equipment Damage**: Stop cascading failures before they happen
2. **Maintain Power Quality**: Ensure customer equipment isn't damaged
3. **Reduce Downtime**: Detect issues before blackouts
4. **Optimize Maintenance**: Focus repairs on failing equipment
5. **Safety**: Prevent dangerous overvoltage/overcurrent conditions

### Examples of Real Anomalies:
- Transformer failures (overheating, oil leaks)
- Loose electrical connections (arcing, overheating)
- Insulator breakdown (corona discharge, tracking)
- Control system failures (wild frequency swings)
- Lightning strikes (transients, surge damage)
- Load imbalances (single-phase short, phasing)

Each creates a distinct "fingerprint" in the spectrogram!

---

## For Your Professor

**Key Points to Emphasize:**

1. **Data Reality**: These are actual Indian power grid measurements, not synthetic data
2. **Visual Interpretability**: Anomalies appear as distinct visual patterns
3. **Class Imbalance**: Anomalies are rare (16% of data) - challenging for standard ML
4. **Time-Frequency Analysis**: Spectrograms capture both temporal and frequency information
5. **AI Explainability**: Our attention mechanisms can show WHERE/WHAT the model detected

**Discussion Points:**
- "Why are anomalies rarer in real data?" → Because faults are rare; normal operation is most of the time
- "How do these compare to other power systems?" → Our patterns are specific to 50 Hz grid; 60 Hz grids look different
- "Can humans reliably detect these?" → Yes, but we need automation for real-time monitoring
- "Why not just look at raw time-series?" → Frequency patterns much more interpretable in spectrogram domain

---

## Summary

**Normal Operations**: Consistent, predictable frequency patterns with expected harmonics
**Anomalies**: Deviations from expected patterns - frequency shifts, harmonic distortion, noise, transients

**The Dataset's Value**: Real-world imbalanced data showing genuine grid anomalies
**Your AI Model**: Learns to distinguish these visual patterns and predict anomalies before they cause problems

---

*Dataset preparation guidance: All spectrograms normalized to 224×224 pixels for deep learning input. Frequency range 0-5 kHz covers all relevant grid harmonics. Time resolution sufficient for transient event detection.*
