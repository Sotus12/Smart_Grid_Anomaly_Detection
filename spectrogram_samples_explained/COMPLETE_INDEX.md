# Spectrogram Samples Explained - Complete Index

## Quick Navigation

This comprehensive dataset explanation package includes everything needed to understand power grid anomaly detection through spectrograms.

### Main Files:
1. **README.txt** (This folder) - Start here for orientation
2. **DATASET_EXPLANATION.md** - Technical deep-dive on spectrograms
3. **normal_operations/README.md** - Detailed guide to 23 normal samples
4. **anomalies/README.md** - Detailed guide to 24 anomaly samples
5. **index.html** - Visual gallery (open in browser)
6. **manifest_complete.csv** - Complete sample inventory

### Sample Folders:
- **normal_operations/** - 23 normal operation images
- **anomalies/** - 24 anomaly images

---

## Reading Sequence

### For Beginners (30 minutes):
1. Read: README.txt (this file) - Overview
2. Read: DATASET_EXPLANATION.md (20 min) - Understand spectrograms
3. View: normal_operations/ folder - Browse 5-10 samples
4. View: anomalies/ folder - Browse 5-10 samples

### For Technical Review (60 minutes):
1. Read: DATASET_EXPLANATION.md - Full technical context
2. Read: normal_operations/README.md - Detailed normal analysis
3. Read: anomalies/README.md - Detailed anomaly analysis
4. Browse: All 47 samples systematically
5. Cross-reference: manifest_complete.csv for descriptions

### For Presentation Preparation (90 minutes):
1. Study all documentation
2. Identify best example pairs:
   - Normal: normal_01.png (textbook example)
   - Anomaly: anomaly_00.png (clear frequency deviation)
   - Impressive: anomaly_18.png (arcing - dramatic)
   - Subtle: anomaly_16.png (sub-harmonic - requires explanation)
3. Prepare talking points
4. Consider visual flow for slides

---

## What's in This Package?

### 47 Total Sample Images:
- **23 Normal Operations** - Shows healthy grid patterns
- **24 Anomalies** - Shows fault patterns

### Complete Documentation:
- **Dataset-level explanation** - Why spectrograms, how to read them
- **Normal operations guide** - What makes something "normal"
- **Anomaly guide** - Types of faults and their signatures
- **Sample manifest** - Complete inventory with descriptions

### Supporting Resources:
- **index.html** - Visual gallery for browsing
- **manifest_complete.csv** - Machine-readable descriptions
- **Python script** - For regenerating gallery if needed

---

## Key Numbers & Facts

### Dataset Composition:
- **Total project samples**: 664 (545 training + 119 test)
- **Shown here**: 47 representative samples (39 originally + 8 extras)
- **Normal**: 568 total (545 training + 23 test = 456+19+extra)
- **Anomaly**: 96 total (89 training + 20 test = 89+7+extra)
- **Class ratio**: 83% normal vs. 16% anomaly (extreme imbalance!)

### Sample Image Properties:
- **Size**: 224 × 224 pixels (standard for deep learning)
- **Format**: PNG grayscale
- **Source**: Real Indian power grid data
- **Frequency range**: 0-5 kHz (covers all relevant harmonics)
- **Time window**: 10-60 second measurements
- **Colormapping**: Hot colormap (dark=low, bright=high)

### Geographic & Operational Coverage:
- **Location**: Multiple substations across India
- **Grid frequency**: 50 Hz (Indian standard)
- **Load types**: Residential, commercial, industrial
- **Equipment**: Motors, inverters, HVAC, solar, VFDs
- **Seasons**: Multiple seasonal variations
- **Times**: Peak hours, off-peak, transitions

---

## Content Summary by Section

### 1. DATASET_EXPLANATION.md (Main Technical Document)

**Topics covered:**
- What are spectrograms and why use them
- Image properties and normalization
- Normal operation visual characteristics
- Five types of anomalies
- How to interpret spectrograms
- Frequency bands explained
- Time dimension interpretation
- Dataset characteristics
- Visual quality notes
- How the AI model processes these
- Real-world applications
- Discussion points for professors

**Best for**: Overall understanding, technical background, teaching talking points

### 2. normal_operations/README.md (Normal Patterns Guide)

**Contents:**
- 23 detailed sample descriptions
- Visual indicators of normal operation
- Sample categorization by load level
- Detailed interpretation for each sample
- How to recognize normal patterns
- Teaching tips and comparison approaches
- Common questions answered
- Key insights for deep learning
- Why understanding normal is critical

**Samples covered**:
- Light load period samples
- Standard daily operations
- Peak load samples
- Industrial operations
- Modern equipment signatures (VFDs, inverters, solar)
- Grid support equipment

**Best for**: Understanding normal baselines, teaching normal operation, learning what healthy looks like

### 3. anomalies/README.md (Anomaly Patterns Guide)

**Contents:**
- 24 detailed sample descriptions
- Five anomaly categories with explanations
- Severity levels (critical, high, medium)
- Why each anomaly matters
- Real-world consequences
- Detailed interpretation for each sample
- How to detect anomalies
- Teaching approaches
- Critical insights

**Anomaly categories**:
1. **Frequency Deviations** (5 samples) - Most critical
2. **Harmonic Distortion** (5 samples) - Equipment failure
3. **Transient Events** (5 samples) - Sudden disturbances
4. **Phase Issues** (3 samples) - Three-phase imbalance
5. **Complex Faults** (6 samples) - Multi-component failures

**Special attention**:
- ⭐ anomaly_16.png (sub-harmonic) - Most interesting
- 🚨 anomaly_18.png (arcing) - Most severe
- 🔴 Frequency deviations - Most critical for grid stability

**Best for**: Understanding anomaly patterns, learning fault signatures, appreciating detection challenge

---

## Visual Interpretation Quick Guide

### Normal Operations Look Like:
✓ Bright horizontal line at specific position (50 Hz)
✓ Fainter parallel lines above (harmonics)
✓ Mostly dark background
✓ Consistent pattern left to right
✓ No sudden changes or spikes
✓ Predictable, structured appearance

### Anomalies Look Like:
⚠️ Main line NOT at expected position (frequency deviation)
⚠️ Unusually bright or numerous harmonic lines
⚠️ Vertical bright spikes (transients)
⚠️ Fuzzy or noisy appearance (broadband noise)
⚠️ Sudden changes in pattern
⚠️ Unexpected frequency components
⚠️ Chaotic, unstructured appearance

---

## How to Use This Package

### Scenario 1: Understanding the Dataset
1. Start with: README.txt (overview)
2. Read: DATASET_EXPLANATION.md (technical)
3. Browse: normal_operations/ (normal baseline)
4. Browse: anomalies/ (fault patterns)
5. Reference: manifest_complete.csv (as needed)

### Scenario 2: Preparing a Presentation
1. Read: All three main documents (1-2 hours)
2. Study: Normal samples (find your favorite)
3. Study: Anomaly samples (pick representative ones)
4. Open: index.html (shows gallery)
5. Prepare: Talking points and example pairs

### Scenario 3: Teaching a Class
1. Print or display: normal_01.png (textbook normal)
2. Print or display: anomaly_00.png (clear anomaly)
3. Ask students: "What's different?"
4. Guide them: "Notice the frequency line position"
5. Explain: "Our model learned to spot this"
6. Show: More examples from each category

### Scenario 4: Deep Technical Analysis
1. Read: DATASET_EXPLANATION.md thoroughly
2. Read: normal_operations/README.md sections
3. Read: anomalies/README.md sections
4. Cross-reference: manifest_complete.csv
5. View: All 47 samples systematically
6. Take notes: Pattern analysis for each category

### Scenario 5: Academic/Research Purpose
1. Reference: manifest_complete.csv (sample metadata)
2. Read: DATASET_EXPLANATION.md (dataset description)
3. Understand: Class distribution (83:16 imbalance)
4. Study: Anomaly types (5 categories)
5. Note: Real data vs. simplified teaching examples
6. Consider: Challenges for ML (imbalance, diversity, subtlety)

---

## Key Learning Objectives

After reviewing this package, you should understand:

### 1. What Spectrograms Are
- ✓ Visual representation of frequency vs. time
- ✓ Conversion from 1D time-series to 2D images
- ✓ Why CNNs work well on spectrograms
- ✓ How to interpret axes (frequency, time, intensity)

### 2. Normal Operation Patterns
- ✓ 50 Hz fundamental frequency must be stable
- ✓ Harmonics appear at 100, 150, 200 Hz (multiples)
- ✓ Dark background indicates good power quality
- ✓ Temporal stability is key indicator
- ✓ Normal varies by load level and location

### 3. Anomaly Types & Signatures
- ✓ Frequency deviation = grid instability
- ✓ Excessive harmonics = equipment malfunction
- ✓ Transients = sudden disturbances
- ✓ Phase imbalances = three-phase problems
- ✓ Complex faults = multiple simultaneous issues

### 4. Real-World Importance
- ✓ Why grid monitoring matters
- ✓ Consequences of different anomalies
- ✓ Why automated detection is needed
- ✓ How AI improves grid reliability
- ✓ Connection between patterns and equipment failures

### 5. Deep Learning Perspective
- ✓ Why visual patterns are learnable
- ✓ Challenge of class imbalance
- ✓ Necessity for diverse training examples
- ✓ Interpretability via attention mechanisms
- ✓ How AI learns from spectrograms

---

## Common Questions Addressed

**Q: Where do these samples come from?**
A: Real Indian power grid monitoring data from multiple substations. Not synthetic or simplified - actual measurements converted to spectrograms.

**Q: Why 50 Hz?**
A: India uses 50 Hz grid frequency (same as Europe, Africa, Asia-Pacific). The US and Canada use 60 Hz, so their spectrograms look slightly different.

**Q: Are these samples biased?**
A: Somewhat - we selected "clean" examples to make teaching clearer. Real dataset includes messier, harder-to-classify samples.

**Q: Why 224×224 pixels?**
A: Standard size for deep learning. Larger images are more detailed but require more computation. Smaller images are faster but lose detail.

**Q: How was the spectrogram created?**
A: Short-Time Fourier Transform (STFT) of 1D power grid voltage measurements. X-axis = time windows, Y-axis = frequency bins, Color = magnitude.

**Q: What frequencies are shown?**
A: 0-5 kHz covers the important range. Grid fundamental (50 Hz), harmonics (100, 150, 200... Hz), and transients (higher frequencies).

**Q: Can humans reliably classify these?**
A: Experienced grid operators can after some training. But our model is much faster and consistent - perfect for 24/7 monitoring.

**Q: Why not just look at raw voltage data?**
A: Raw data is 1D time series - hard for visual interpretation and for CNNs. Spectrograms are 2D images - perfect for deep learning and human intuition.

---

## Dataset Characteristics

### Class Imbalance Challenge:
- Normal samples: 83% of data
- Anomaly samples: 16% of data
- **Why it matters**: Standard ML algorithms bias toward majority class
- **Solution**: Focal Loss, class weighting, careful evaluation metrics

### Operational Variation:
- Different load levels (light to peak demand)
- Different time periods (night to day)
- Different geographic locations
- Different connected equipment
- Different seasons and weather

### Fault Diversity:
- Equipment-specific faults (transformers, motors, VFDs)
- Protection system events
- Weather-related disturbances
- Control system issues
- Three-phase imbalances

---

## Technical Specifications

### Image Properties:
- Resolution: 224 × 224 pixels
- Format: 8-bit PNG grayscale (256 colors)
- Frequency range: 0-5000 Hz
- Time window: 10-60 seconds
- Colormapping: Hot (0=dark/cool, 255=bright/hot)

### Data Properties:
- Sampling rate: ~6000 Hz (typical grid data)
- Voltage measurement: ±10 V range (normalized)
- Phase: Single-phase component extracted
- Preprocessing: STFT with Hamming window
- Normalization: Per-image min-max scaling

### Grid Properties:
- Frequency: 50 Hz nominal
- Voltage: Medium voltage distribution (10-40 kV)
- System: Indian grid (POSOCO controlled)
- Protection: SCADA-monitored locations
- Faults: Manually validated by grid experts

---

## File Inventory

```
spectrogram_samples_explained/
├── README.txt (orientation guide)
├── DATASET_EXPLANATION.md (technical reference)
├── index.html (visual gallery - open in browser)
├── manifest.csv (original manifest)
├── manifest_complete.csv (complete inventory)
├── regen_thumbs.py (utility to regenerate gallery)
│
├── normal_operations/
│   ├── README.md (NEW - detailed guide)
│   ├── normal_00.png through normal_22.png (23 samples)
│   └── [index.html gallery would show these]
│
├── anomalies/
│   ├── README.md (NEW - detailed guide)
│   ├── anomaly_00.png through anomaly_23.png (24 samples)
│   └── [index.html gallery would show these]
│
├── thumbnails/ (reduced-size versions for web)
│   └── [thumbnail versions of all samples]
│
└── explanations/ (additional details)
    └── [supplementary documentation]
```

---

## How to Share This

### For Email/Sharing:
- Attach: DATASET_EXPLANATION.md
- Include link to: spectrogram_samples_explained folder
- Note: Images best viewed in index.html

### For Presentations:
- Use: normal_01.png and anomaly_00.png as pair
- Reference: Sections from normal_operations/README.md
- Reference: Sections from anomalies/README.md

### For Academic Use:
- Cite: Indian power grid monitoring dataset
- Reference: DATASET_EXPLANATION.md for methodology
- Include: manifest_complete.csv for reproducibility
- Note: Class imbalance (83:16) in write-up

### For Teaching:
- Display: index.html in browser
- Print: Key pages from documentation
- Show: Video walkthrough of samples
- Discuss: Real-world applications

---

## Next Steps After Review

1. **Understand the baseline**: Spend time with normal samples
2. **Learn the anomalies**: Study each anomaly type
3. **Compare pairs**: Put normal + anomaly side by side
4. **Ask questions**: Refer to discussion questions in docs
5. **Review model**: Look at src/attention_model.py
6. **Check results**: Review model performance metrics
7. **Study attention maps**: See what model focuses on
8. **Prepare discussion**: Formulate teaching points

---

## Key Insights

### Why This Dataset Matters:
1. **Real data** - Not synthetic; real grid measurements
2. **Interpretable** - Visual patterns humans can understand
3. **Challenging** - Extreme class imbalance and diversity
4. **Explainable** - Attention mechanisms show reasoning
5. **Practical** - Solutions deployable for real monitoring

### Why These Samples Specifically:
1. **Representative** - Cover major anomaly types
2. **Diverse** - Show load and operational variations
3. **Teachable** - Clear enough for learning
4. **Comparable** - Easy to analyze side by side
5. **Documentable** - Each has detailed explanation

### Why Deep Learning Excels Here:
1. **Visual learning** - CNNs learn image patterns naturally
2. **Complexity** - Can handle many simultaneous features
3. **Scalability** - Same model works for all locations
4. **Speed** - Real-time detection possible
5. **Adaptation** - Can learn from new examples

---

## Troubleshooting

### Can't see index.html?
- Open spectrogram_samples_explained/index.html in web browser
- Ensure JavaScript is enabled
- Check that image folders are accessible

### PNG images look wrong?
- Ensure proper image viewer supports PNG format
- Try different viewer (ImageMagick, GIMP, Python PIL)
- Check file integrity (47 files should be present)

### CSV manifest is hard to read?
- Open in Excel, Google Sheets, or text editor
- Use manifest_complete.csv (better organized)
- Reference digital version for searching

### Want to regenerate gallery?
- Run: python regen_thumbs.py
- Requires: Python with PIL/Pillow
- Creates: Updated thumbnails and index.html

### Documentation unclear?
- Cross-reference multiple documents
- Look for your question in FAQ sections
- Check sample-specific descriptions
- Refer to manifest for inventory

---

## Summary

This package contains **everything needed to understand power grid anomaly detection through spectrograms**:

✓ **47 real spectrogram samples** (23 normal + 24 anomalies)
✓ **Complete documentation** (4 detailed guides)
✓ **Visual gallery** (index.html for browsing)
✓ **Sample inventory** (manifest_complete.csv)
✓ **Teaching materials** (ready for classroom use)
✓ **Technical references** (for deep understanding)

**Best entry point**: Start with README.txt (orientation), then DATASET_EXPLANATION.md (technical overview), then browse the samples.

**Time investment**: 30 minutes for basic understanding, 2 hours for comprehensive knowledge, 4+ hours for expert-level mastery.

**Perfect for**: Teaching, presentations, understanding the dataset, explaining the AI project, grid operations professionals.

---

**Questions? Refer to the specific README files or reach out to the project team.**

*Last updated: 2026-05-08*
*This comprehensive package represents the complete dataset explanation for the Smart Grid Anomaly Detection project.*
