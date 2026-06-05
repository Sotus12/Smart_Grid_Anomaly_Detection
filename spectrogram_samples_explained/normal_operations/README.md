# Normal Operations - Spectrogram Samples

## Overview

This folder contains **23 representative samples** of **normal, healthy power grid operation** from the Indian grid monitoring dataset. These spectrograms show what "normal" looks like - the baseline patterns our AI model must learn to distinguish from anomalies.

---

## What Makes These "Normal"?

Normal operation spectrograms share common characteristics:

### Visual Indicators of Normal Operation:

1. **Fundamental Frequency Line (50 Hz)**
   - Bright, sharp horizontal line in lower frequency region
   - Position is stable at ~50 Hz (± 0.2 Hz tolerance)
   - Represents the main grid frequency
   - Should be the most prominent feature

2. **Harmonic Series**
   - Additional horizontal lines at 100 Hz, 150 Hz, 200 Hz, 250 Hz, etc.
   - Each is a multiple of the 50 Hz fundamental
   - Progressively fainter as frequency increases
   - Caused by nonlinear loads (transformers, motors, rectifiers)
   - Within acceptable limits (Total Harmonic Distortion < 5%)

3. **Clean Noise Floor**
   - Mostly dark background between frequency lines
   - Low baseline energy across spectrum
   - Indicates good power quality
   - No chaotic or fuzzy appearance

4. **Temporal Stability**
   - Pattern consistent from left to right (across time)
   - Smooth variations when they occur
   - No sudden jumps or abrupt changes
   - Indicates steady-state operation

5. **Consistent Brightness Across Frequencies**
   - Brightness primarily at harmonic frequencies
   - Not elevated in unexpected places
   - Follows predictable power spectrum shape

---

## Sample Categories

The 23 normal samples represent different operational scenarios:

### Light Load Period (Low Demand)
- **Samples**: normal_00.png, normal_05.png
- **Characteristics**: Dim overall, minimal harmonic content, very clean
- **When this occurs**: Night hours, early morning, weekends
- **Why important**: Shows AI must recognize normal at different load levels
- **Visual pattern**: Thin, barely visible lines with dark background

### Standard Daily Operation
- **Samples**: normal_01.png, normal_02.png, normal_06.png
- **Characteristics**: Clear 50 Hz line, moderate brightness, textbook example
- **When this occurs**: Regular daytime operation
- **Why important**: "Textbook normal" - what professors expect to see
- **Visual pattern**: Bright fundamental, faint harmonics, clean background

### Peak Load Hours (High Demand)
- **Samples**: normal_10.png, normal_11.png, normal_12.png, normal_13.png
- **Characteristics**: Very bright, strong harmonic lines, but still stable
- **When this occurs**: Morning (6-9 AM), Evening (6-9 PM)
- **Why important**: Normal ≠ constant brightness; load varies daily
- **Visual pattern**: Intense lines maintained across time window

### Industrial Area Operations
- **Samples**: normal_03.png, normal_04.png, normal_07.png, normal_08.png
- **Characteristics**: Heavy harmonic content, complex structure, but predictable
- **When this occurs**: Industrial zones have different load profiles
- **Why important**: Different areas have different "normal" patterns
- **Visual pattern**: Multiple bright harmonic lines with structured pattern

### Variable Speed Drives & Inverters
- **Samples**: normal_09.png, normal_14.png, normal_15.png, normal_16.png
- **Characteristics**: Complex harmonic structure, but orderly and consistent
- **When this occurs**: Modern equipment (HVAC, solar, EV chargers)
- **Why important**: New grid equipment creates different but still-normal patterns
- **Visual pattern**: Many harmonics with predictable spacing, no transients

### Grid Support Equipment
- **Samples**: normal_17.png, normal_18.png, normal_19.png, normal_20.png, normal_21.png, normal_22.png
- **Characteristics**: Stabilizing equipment active, compensation signatures visible
- **When this occurs**: During peak demand, renewable injection, or instability
- **Why important**: FACTS devices and capacitors have distinctive signatures
- **Visual pattern**: Controlled, intentional patterns from active equipment

---

## Detailed Sample Guide

### Sample normal_00.png
- **Description**: Very clean; thin 50Hz line; minimal harmonics
- **Load Level**: Very light
- **Scenario**: Low demand period (night/weekend)
- **Key Feature**: Minimal energy - looks mostly dark
- **Why Show This**: Demonstrates normal at minimum load
- **Anomaly Check**: ✓ 50 Hz is stable, ✓ No spikes, ✓ Clean background

### Sample normal_01.png
- **Description**: Standard daily operation; clear 50Hz; faint harmonics
- **Load Level**: Moderate
- **Scenario**: Typical daytime operation
- **Key Feature**: Sharp 50 Hz line with visible but faint harmonics
- **Why Show This**: Classic "textbook normal" pattern
- **Anomaly Check**: ✓ Perfect example for comparison
- **BEST FOR**: Teaching - use this as the reference

### Sample normal_02.png
- **Description**: Morning peak; bright 50Hz; multiple harmonics
- **Load Level**: High
- **Scenario**: Morning demand surge (6-9 AM)
- **Key Feature**: Increased brightness in fundamental and harmonics
- **Why Show This**: Normal can be bright if load is legitimate
- **Anomaly Check**: ✓ Brightness is due to high demand, not fault

### Sample normal_03.png
- **Description**: Industrial area; strong harmonics; heavy loads
- **Load Level**: Very high
- **Scenario**: Industrial facility during operation
- **Key Feature**: Many bright harmonic lines due to industrial equipment
- **Why Show This**: Industrial "normal" differs from residential
- **Anomaly Check**: ✓ Pattern is complex but still structured and stable

### Sample normal_04.png
- **Description**: Evening steady state; stable pattern; moderate brightness
- **Load Level**: Moderate-High
- **Scenario**: Evening hours (peak demand beginning)
- **Key Feature**: Consistent pattern with gradual brightness
- **Why Show This**: Shows sustained normal operation
- **Anomaly Check**: ✓ Stability across time window is key indicator

### Sample normal_05.png
- **Description**: Light load period; dim overall; minimal harmonics
- **Load Level**: Very light
- **Scenario**: Low demand (off-peak)
- **Key Feature**: Minimal visible content, mostly dark
- **Why Show This**: Another light-load example for variety
- **Anomaly Check**: ✓ Dim doesn't mean anomalous if 50Hz is stable

### Sample normal_06.png
- **Description**: Steady state reference; textbook healthy operation
- **Load Level**: Moderate
- **Scenario**: Reference example for teaching
- **Key Feature**: Perfect representation of healthy operation
- **Why Show This**: Gold standard for "normal"
- **Anomaly Check**: ✓ Use as reference for all comparisons
- **PROFESSOR'S TIP**: Show this first, then compare others to it

### Sample normal_07.png
- **Description**: Commercial loads; HVAC and elevator harmonics
- **Load Level**: Moderate-High
- **Scenario**: Commercial building (office/mall) during operation
- **Key Feature**: Specific harmonic signature from HVAC and elevators
- **Why Show This**: Commercial areas have distinctive normal signatures
- **Anomaly Check**: ✓ Different from industrial but still normal

### Sample normal_08.png
- **Description**: Motor-dominant loads; motor harmonic signature
- **Load Level**: High
- **Scenario**: Motor-heavy facility (factory, pump station)
- **Key Feature**: Characteristic motor harmonic pattern (3rd, 5th, 7th dominant)
- **Why Show This**: Motors create specific harmonic patterns
- **Anomaly Check**: ✓ Predictable motor signature indicates normal operation

### Sample normal_09.png
- **Description**: Mixed residential; moderate brightness and complex pattern
- **Load Level**: Moderate
- **Scenario**: Residential neighborhood with diverse loads
- **Key Feature**: Complex but balanced harmonic content
- **Why Show This**: Residential areas have varied but still-normal patterns
- **Anomaly Check**: ✓ Complexity doesn't indicate fault if pattern is stable

### Sample normal_10.png
- **Description**: High-power transmission; very bright; clear harmonics
- **Load Level**: Very high (transmission level)
- **Scenario**: Main transmission line during peak demand
- **Key Feature**: Very bright due to transmission voltage level
- **Why Show This**: Transmission-level "normal" is brighter than distribution
- **Anomaly Check**: ✓ Brightness is expected at transmission level

### Sample normal_11.png
- **Description**: Sustained peak load; consistently bright across time
- **Load Level**: Very high (sustained)
- **Scenario**: Prolonged peak demand period
- **Key Feature**: Consistently bright pattern throughout time window
- **Why Show This**: Sustained peaks are normal during certain hours
- **Anomaly Check**: ✓ Consistency across time indicates controlled load

### Sample normal_12.png
- **Description**: Evening ramp-up; gradual increase in harmonic content
- **Load Level**: Increasing from moderate to high
- **Scenario**: Transition from day to evening peak demand
- **Key Feature**: Gradual left-to-right increase in brightness
- **Why Show This**: Gradual changes are normal; sudden changes are anomalies
- **Anomaly Check**: ✓ Smooth transition, no abrupt jumps

### Sample normal_13.png
- **Description**: Industrial peak hours; very bright strong harmonics
- **Load Level**: Very high
- **Scenario**: Industrial facility at peak operation
- **Key Feature**: Intense harmonic lines during high-load period
- **Why Show This**: Industrial sites during peak are legitimately bright
- **Anomaly Check**: ✓ Brightness and complexity are expected

### Sample normal_14.png
- **Description**: Variable speed drive operation; complex harmonic structure
- **Load Level**: Moderate-High
- **Scenario**: VFD (Variable Frequency Drive) equipment running
- **Key Feature**: Complex harmonic pattern with controlled structure
- **Why Show This**: Modern VFDs create different but normal patterns
- **Anomaly Check**: ✓ VFD harmonics are predictable and acceptable

### Sample normal_15.png
- **Description**: Regenerative braking event; inverter returning power
- **Load Level**: Variable (bidirectional power flow)
- **Scenario**: Electric vehicle or renewable energy storage injecting power
- **Key Feature**: Inverter signature with controlled harmonic content
- **Why Show This**: Modern grids support distributed energy resources
- **Anomaly Check**: ✓ Inverter patterns are well-behaved when operating normally

### Sample normal_16.png
- **Description**: Distributed solar injection; inverter signature during day
- **Load Level**: Variable (depends on solar output)
- **Scenario**: Solar panel array injecting power during daylight
- **Key Feature**: Solar inverter harmonic signature
- **Why Show This**: Solar is increasingly common in modern grids
- **Anomaly Check**: ✓ Solar inverter patterns are recognized as normal

### Sample normal_17.png
- **Description**: Grid stabilizing equipment active; compensation signatures
- **Load Level**: Variable
- **Scenario**: FACTS device or capacitor bank engaged
- **Key Feature**: Active control signatures from stabilization equipment
- **Why Show This**: Grid support equipment creates intentional patterns
- **Anomaly Check**: ✓ These patterns indicate the grid is being actively managed

### Sample normal_18.png
- **Description**: Transition to night operation; gradually dimming pattern
- **Load Level**: Decreasing from peak to low
- **Scenario**: Evening-to-night transition (9 PM - 11 PM)
- **Key Feature**: Gradual decrease in brightness over time window
- **Why Show This**: Daily transitions are smooth and normal
- **Anomaly Check**: ✓ Smooth dimming is different from sudden drops

### Sample normal_19.png
- **Description**: Off-peak low load; low amplitude; clean 50Hz (extra sample)
- **Load Level**: Very low
- **Scenario**: Off-peak period (late night/early morning)
- **Key Feature**: Minimal brightness with clear 50 Hz fundamental
- **Why Show This**: Demonstrates normal at absolute minimum load
- **Anomaly Check**: ✓ Still shows clean 50 Hz despite low brightness

### Sample normal_20.png
- **Description**: Solar injection variation; daytime inverter pattern (extra sample)
- **Load Level**: Moderate (solar-dependent)
- **Scenario**: Solar generation with cloud variations
- **Key Feature**: Solar inverter responding to variable cloud cover
- **Why Show This**: Real solar generation is variable but normal
- **Anomaly Check**: ✓ Variations are within inverter's normal operating range

### Sample normal_21.png
- **Description**: Grid support equipment active; capacitors/reactors engaged (extra sample)
- **Load Level**: Variable
- **Scenario**: Capacitor banks switching on for reactive power support
- **Key Feature**: Capacitor bank operation signatures
- **Why Show This**: Grid support creates distinctive patterns
- **Anomaly Check**: ✓ These are intentional control actions, not anomalies

### Sample normal_22.png
- **Description**: Extra normal sample; typical variation (auto-added)
- **Load Level**: Moderate
- **Scenario**: General representation of typical normal operation
- **Key Feature**: Representative of average operational state
- **Why Show This**: Provides additional diversity in normal examples
- **Anomaly Check**: ✓ Falls within normal operational envelope

---

## How to Interpret These Samples

### Step 1: Identify the Fundamental Frequency
Look for the **brightest horizontal line in the lower frequency region** (around 1/4 height from bottom).
- **In normal operations**: Should be at exactly 50 Hz
- **Location**: Should not shift up or down
- **Brightness**: Brightest feature in the entire image

### Step 2: Look for Harmonics
Look for **fainter horizontal lines above the fundamental**.
- **Expected positions**: 100 Hz, 150 Hz, 200 Hz, 250 Hz, etc.
- **Expected brightness**: Each harmonic is fainter than the previous one
- **Expected pattern**: Regular spacing (every 50 Hz)

### Step 3: Check the Noise Floor
Look at the **overall brightness of the background**.
- **In normal operations**: Should be mostly dark
- **Exception**: May be brighter if load is high
- **Look for**: No "haze" or fuzzy appearance

### Step 4: Scan Across Time
Follow the pattern **from left to right** (start to end of measurement).
- **In normal operations**: Pattern should be stable or show gradual changes
- **RED FLAG**: Sudden jumps, breaks, or abrupt shifts

### Step 5: Look for Anomaly Indicators
Quickly scan for **warning signs**:
- ⚠️ Vertical bright lines (transients/spikes)
- ⚠️ Frequency lines NOT at expected positions
- ⚠️ Chaotic or noisy appearance
- ⚠️ Multiple simultaneous anomalies

---

## Teaching Tips

### Comparison Approach
1. Show normal_01.png (classic normal) and anomaly_00.png together
2. Ask: "What's different?"
3. Highlight: "The position of the main line changed - that's a frequency anomaly"
4. Explain: "Our model learns to spot these visual differences"

### Load Variation Lesson
1. Show normal_05.png (very light load) and normal_13.png (very heavy load) together
2. Ask: "Are both normal?"
3. Explain: "Yes! Normal includes a wide range of brightness. The key is **stability** and **structure**, not absolute brightness"

### Industrial vs. Residential
1. Show normal_01.png (simple harmonic structure) and normal_08.png (complex) together
2. Explain: "Different areas have different normal patterns due to different equipment"
3. Key insight: "Our AI must learn all these variations as normal"

### Timeline of Normal Operation
Show in sequence: normal_05.png → normal_12.png → normal_18.png
- Explain: "This is a typical day: night → morning peak → evening decline"
- Insight: "These natural transitions are all normal"

---

## Common Questions Answered

**Q: Why are some samples so bright and others so dark?**
A: Brightness represents load level and power flow. Morning/evening peaks are legitimately bright. Off-peak periods are naturally dim. Both are normal if the pattern is stable and structured.

**Q: What if the 50 Hz line looks off-center?**
A: That could indicate frequency deviation. In these normal samples, the 50 Hz line should always be at the same vertical position. If it moves, that's anomalous.

**Q: Why do some samples have so many harmonics?**
A: Different equipment creates different harmonic patterns. Motors, drives, inverters all have characteristic signatures. As long as they're predictable and within limits, it's normal.

**Q: Can two different areas have completely different "normal" patterns?**
A: Yes! Industrial areas, commercial zones, and residential neighborhoods have very different typical loads and equipment. Each has its own version of "normal."

**Q: Is a very dim sample broken?**
A: No, sometimes measurements capture very low demand periods (late night, holidays). Dim is normal if the 50 Hz line is stable and structure is present.

---

## Key Insights for Deep Learning

### Why Spectrograms Are Perfect for CNNs
- These images contain **spatial patterns** (frequency vs. time)
- **Convolutional layers** can extract features like "bright line at position X"
- **Pooling layers** can recognize "stable pattern" vs. "changing pattern"
- **Attention layers** can highlight "important regions"

### What the AI Actually Learns
From these 23 normal samples (and 545 training samples), the model learns:
1. **Frequency patterns**: "50 Hz should be at position Y"
2. **Harmonic structure**: "Harmonics should be at 100, 150, 200 Hz with decreasing brightness"
3. **Noise baseline**: "Background should be dark except at frequency lines"
4. **Temporal stability**: "Pattern should be consistent across time"
5. **Load variations**: "Brightness can vary, but structure stays the same"

### Why Understanding These is Critical
Before you can understand **why** the model fails on anomalies, you must understand **what** it learns from normal samples. These 23 samples are the foundation!

---

## Next Steps

1. **Look at all 23 samples** - spend 5 minutes just browsing
2. **Pick your favorite normal sample** - what makes it "normal" to you?
3. **Compare to anomalies folder** - see how different they look
4. **Read the DATASET_EXPLANATION.md** - for deeper context
5. **Look at model attention maps** - to see where the AI focuses

---

## Summary

**These 23 samples represent the baseline of normal power grid operation:**
- ✓ 50 Hz fundamental is stable and centered
- ✓ Harmonics at expected positions with expected brightness
- ✓ Clean background with minimal noise
- ✓ Stable patterns across time
- ✓ Load variations within expected ranges
- ✓ No unexpected frequency components
- ✓ No transients or spikes
- ✓ Structured, predictable appearance

**Anomalies will deviate from this baseline in obvious ways** - that's what makes them detectable.

---

*Questions? Start with DATASET_EXPLANATION.md or compare directly with anomalies/README.md to see the contrasts.*
