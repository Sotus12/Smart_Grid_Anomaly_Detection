# Anomalies - Spectrogram Samples

## Overview

This folder contains **24 representative samples** of **power grid anomalies** from the Indian grid monitoring dataset. These spectrograms show what "abnormal" looks like - the deviations from normal operation that indicate equipment faults, power quality problems, or grid disturbances.

These are the patterns our deep learning model must learn to detect automatically. Each anomaly represents a real fault condition that grid operators need to know about immediately.

---

## What Makes These Anomalous?

Anomalies are **deviations from the normal baseline**. Unlike the clean, structured patterns in normal operations, anomalies show:

### Visual Indicators of Anomalies:

1. **Frequency Deviation**
   - Main frequency line NOT at 50 Hz
   - Shifted higher (over-frequency) or lower (under-frequency)
   - Indicates grid instability
   - Can appear as sloped lines (frequency changing over time)

2. **Excessive Harmonic Distortion**
   - Harmonics brighter than expected
   - More harmonic lines than typical
   - New unexpected frequency components
   - Indicates equipment malfunction or fault

3. **Transients/Impulsive Spikes**
   - Sudden vertical bright lines
   - Broadband energy at all frequencies
   - Indicates switching events or fault clearing
   - Can appear sudden and sharp

4. **Broadband Noise**
   - Elevated noise floor (brighter background)
   - Fuzzy, hazy appearance
   - Energy spread across many frequencies
   - Indicates poor power quality or EMI

5. **Phase Discontinuities**
   - Abrupt changes in pattern
   - Left side looks different from right side
   - Indicates events during measurement window
   - Can show mode changes or phase shifts

6. **Sub-harmonics or Inter-harmonics**
   - Frequency components NOT at 50 Hz multiples
   - Appears between expected harmonic lines
   - Indicates specific equipment faults
   - Requires attention and investigation

---

## Critical Severity Levels

### 🔴 CRITICAL (Immediate Action Required)
- **Frequency sag/swell** → Grid voltage instability → Could lead to blackouts
- **Severe transients** → Could damage equipment → Risk of cascade failures
- **Arcing events** → Fire hazard → Immediate shut-down needed
- **Complex multi-anomalies** → Multiple faults → System emergency

### 🟠 HIGH (Urgent Investigation)
- **Persistent frequency deviation** → Equipment not responding → Could escalate
- **Excessive harmonics** → Equipment overheating → Premature failure
- **Sub-harmonics** → Specific equipment failing → Targeted maintenance

### 🟡 MEDIUM (Schedule Maintenance)
- **Interharmonics** → Power quality issue → Affects customer equipment
- **Phase imbalance** → Uneven load → Can cause burnout over time
- **Mild harmonic distortion** → Load issue → Monitor and optimize

---

## Anomaly Categories

### Category 1: Frequency Deviations (anomaly_00 to 04)

These represent the **most critical grid anomalies** - when the fundamental frequency moves away from 50 Hz.

#### Why This Matters:
- Grid frequency MUST stay at 50 Hz (±0.2 Hz)
- Deviation indicates:
  - Imbalance between supply and demand
  - Generator failure or disconnection
  - Load shedding or major load change
  - Control system malfunction
- **Consequence**: Cascade failures across grid

#### Sample Details:

**Sample anomaly_00.png**
- **Description**: Frequency swell; main line above 50Hz; over-frequency event
- **Severity**: CRITICAL 🔴
- **What's happening**: Grid frequency is TOO HIGH (over 50 Hz)
- **Cause**: Sudden loss of load or generator over-supply
- **Visible pattern**: Main frequency line appears HIGHER than normal position
- **Real-world**: Excess power being injected; could damage equipment
- **Action**: Immediate load shedding or generator reduction needed
- **AI Signal**: Line shifts upward on frequency axis

**Sample anomaly_01.png**
- **Description**: Frequency sag; main line below 50Hz; under-frequency event
- **Severity**: CRITICAL 🔴
- **What's happening**: Grid frequency is TOO LOW (below 50 Hz)
- **Cause**: Sudden loss of generation or excessive load
- **Visible pattern**: Main frequency line appears LOWER than normal
- **Real-world**: Grid cannot keep up with demand; risk of blackout
- **Action**: Emergency measures needed; load shedding imminent
- **AI Signal**: Line shifts downward on frequency axis

**Sample anomaly_02.png**
- **Description**: Rapid frequency change; main line moves across time; instability
- **Severity**: CRITICAL 🔴
- **What's happening**: Frequency is CHANGING RAPIDLY during measurement
- **Cause**: Oscillatory instability; control system hunting
- **Visible pattern**: Main line appears to SLOPE or MOVE from left to right
- **Real-world**: Grid is oscillating; could lead to blackout
- **Action**: Stabilizer required; could cascade to wide-area blackout
- **AI Signal**: Non-horizontal fundamental line; motion across time

**Sample anomaly_03.png**
- **Description**: Frequency oscillation; wobbling main line; oscillatory instability
- **Severity**: CRITICAL 🔴
- **What's happening**: Frequency is OSCILLATING repeatedly around 50 Hz
- **Cause**: Poorly damped oscillations; synchronization issues
- **Visible pattern**: Main line appears to WOBBLE or have WAVY shape
- **Real-world**: System in dangerous borderline state
- **Action**: Damping control must be activated immediately
- **AI Signal**: Non-straight fundamental line; wobbling appearance

**Sample anomaly_04.png**
- **Description**: Severe frequency deviation; major shift from nominal
- **Severity**: CRITICAL 🔴
- **What's happening**: Frequency has MOVED FAR from 50 Hz
- **Cause**: Severe supply-demand imbalance; major equipment failure
- **Visible pattern**: Main line is at clearly WRONG position
- **Real-world**: Extreme event; blackout or generator failure
- **Action**: Emergency protocols; system disconnection possible
- **AI Signal**: Fundamental line position drastically wrong

---

### Category 2: Harmonic Distortion (anomaly_05 to 09)

These represent **excessive harmonic pollution** - when unwanted frequency components appear too bright or too many.

#### Why This Matters:
- Harmonics are normal but must be limited
- Excessive harmonics cause:
  - Equipment heating and accelerated aging
  - Transformer overheating
  - Motor efficiency loss
  - Capacitor damage
  - Interference with communication systems
- **Threshold**: Total Harmonic Distortion (THD) should be < 5%

#### Sample Details:

**Sample anomaly_05.png**
- **Description**: Excessive 3rd harmonic; bright 150Hz; possible grounding issue
- **Severity**: HIGH 🟠
- **What's happening**: 150 Hz (3rd harmonic) is TOO BRIGHT
- **Cause**: Grounding problem or three-phase load imbalance
- **Visible pattern**: Line at 150 Hz appears MUCH BRIGHTER than expected
- **Real-world**: Neutral current issues; could cause ground faults
- **Action**: Investigate grounding system; load rebalancing needed
- **Equipment at risk**: Transformers, neutral connections
- **AI Signal**: Specific harmonic line elevated above expected level

**Sample anomaly_06.png**
- **Description**: High 5th harmonic; bright 250Hz; converter or VFD malfunction
- **Severity**: HIGH 🟠
- **What's happening**: 250 Hz (5th harmonic) is EXCESSIVELY BRIGHT
- **Cause**: Variable Frequency Drive (VFD) or power converter malfunction
- **Visible pattern**: Line at 250 Hz is BRIGHTER than other harmonics
- **Real-world**: VFD is faulty or not filtering harmonics properly
- **Action**: VFD maintenance; harmonic filter check needed
- **Equipment at risk**: VFDs, power electronics, capacitors
- **AI Signal**: Odd harmonic line (250 Hz) over-emphasized

**Sample anomaly_07.png**
- **Description**: Multiple harmonics elevated; severe harmonic pollution
- **Severity**: HIGH 🟠
- **What's happening**: MANY harmonics are too bright
- **Cause**: Multiple equipment faults or massive load distortion
- **Visible pattern**: Several harmonic lines appear UNUSUALLY BRIGHT
- **Real-world**: System-wide harmonic pollution; widespread issues
- **Action**: Comprehensive investigation of loads; filtering required
- **Equipment at risk**: Entire distribution system
- **AI Signal**: Multiple harmonic lines across expected spectrum elevated

**Sample anomaly_08.png**
- **Description**: Odd harmonics prominent; unusual harmonic pattern
- **Severity**: MEDIUM 🟡
- **What's happening**: Odd harmonics (100, 150, 250 Hz) dominate
- **Cause**: Nonlinear load operation outside normal parameters
- **Visible pattern**: ODD harmonics bright, EVEN harmonics weak or absent
- **Real-world**: Specific equipment type creating unusual signature
- **Action**: Identify and investigate the specific load
- **Equipment at risk**: Specific customer equipment
- **AI Signal**: Unusual harmonic sequence (not standard pattern)

**Sample anomaly_09.png**
- **Description**: Interharmonics visible; frequency components between harmonics
- **Severity**: MEDIUM 🟡
- **What's happening**: Frequency components appear BETWEEN expected harmonics
- **Cause**: Cycloconverter, AC drives, or other frequency-conversion equipment
- **Visible pattern**: Bright lines appear at UNEXPECTED FREQUENCIES (not at 50 Hz multiples)
- **Real-world**: Specific equipment type present; needs investigation
- **Action**: Equipment identification; potential replacement/upgrade
- **Equipment at risk**: Sensitive electronics, communication systems
- **AI Signal**: Energy at frequencies that shouldn't exist (e.g., 75 Hz instead of 100 Hz)

---

### Category 3: Transient Events (anomaly_10 to 14)

These represent **sudden disturbances** - when sharp spikes or sudden energy bursts appear.

#### Why This Matters:
- Transients can damage sensitive equipment
- Can cause:
  - Equipment shutdown
  - Data corruption
  - Accelerated aging
  - Fire in extreme cases
- Often precede or follow major faults

#### Sample Details:

**Sample anomaly_10.png**
- **Description**: Single sharp transient; vertical spike from switching or fault clearing
- **Severity**: HIGH 🟠
- **What's happening**: ONE sudden bright VERTICAL line across all frequencies
- **Cause**: Switching operation, fault clearing, or circuit breaker action
- **Visible pattern**: Single SPIKE extending from bottom to top of image
- **Real-world**: Equipment switched or fault was cleared
- **Action**: Log event; verify protection system worked
- **Equipment at risk**: Computers, power supplies, sensitive loads
- **AI Signal**: Broadband energy burst at specific time

**Sample anomaly_11.png**
- **Description**: Multiple switching transients; repeated vertical spikes
- **Severity**: HIGH 🟠
- **What's happening**: MULTIPLE VERTICAL spikes at different times
- **Cause**: Repeated switching operations or oscillatory fault clearing
- **Visible pattern**: Several SPIKES distributed across time window
- **Real-world**: Multiple events occurring; intermittent fault likely
- **Action**: Investigate repeated transients; could indicate breaker chatter
- **Equipment at risk**: Repeatedly stressed devices
- **AI Signal**: Multiple distinct transient peaks on time axis

**Sample anomaly_12.png**
- **Description**: Broadband transient; fuzzy bright energy across many frequencies
- **Severity**: HIGH 🟠
- **What's happening**: WIDE spectrum suddenly becomes bright
- **Cause**: Major switching event, lightning strike, or fault inception
- **Visible pattern**: Image shows FUZZY bright region (not thin lines)
- **Real-world**: Severe transient event; could be lightning-induced
- **Action**: Check for equipment damage; storm activity noted
- **Equipment at risk**: Wide range of devices; potential cascade damage
- **AI Signal**: Broadband energy; noise-like appearance in spectrogram

**Sample anomaly_13.png**
- **Description**: Ringing transient; spike followed by damped oscillation
- **Severity**: HIGH 🟠
- **What's happening**: SPIKE followed by DECAYING oscillations
- **Cause**: Sharp transient with natural frequency oscillation of system
- **Visible pattern**: Bright spike followed by repeated WAVY lines
- **Real-world**: System ringing; natural oscillation at grid's natural frequency
- **Action**: Monitor for additional disturbances; verify damping
- **Equipment at risk**: Resonant circuits, capacitors, transformers
- **AI Signal**: Transient followed by sustained frequency activity

**Sample anomaly_14.png**
- **Description**: Repetitive transient; regular interval spikes indicating intermittent fault
- **Severity**: HIGH 🟠
- **What's happening**: SPIKES appear at REGULAR INTERVALS
- **Cause**: Intermittent fault with periodic repetition; unstable connection
- **Visible pattern**: Multiple SPIKES equally spaced across time window
- **Real-world**: Loose connection, intermittent breaker, arcing fault
- **Action**: Find and fix intermittent fault immediately
- **Equipment at risk**: Fault can escalate to total failure
- **AI Signal**: Periodic impulses; regular pattern to transient events

---

### Category 4: Phase and Balance Issues (anomaly_15 to 17)

These represent **three-phase system imbalances** - when the normal symmetric operation is broken.

#### Why This Matters:
- Three-phase systems must be balanced
- Imbalances cause:
  - Negative-sequence currents
  - Motor vibration and overheating
  - Equipment malfunction
  - Uneven power distribution
- Can indicate one phase has an issue

#### Sample Details:

**Sample anomaly_15.png**
- **Description**: Even harmonics present; indicates phase imbalance
- **Severity**: MEDIUM 🟡
- **What's happening**: EVEN harmonics (100, 200 Hz) appear when they shouldn't
- **Cause**: Phase imbalance; uneven three-phase loading
- **Visible pattern**: Lines appear at 100 Hz and other EVEN multiples
- **Real-world**: One or more phases different from others
- **Action**: Check phase loads; balance the system
- **Equipment at risk**: Induction motors, three-phase equipment
- **AI Signal**: Even harmonics in spectrogram (don't occur in balanced systems)

**Sample anomaly_16.png** ⭐ **MOST INTERESTING ANOMALY**
- **Description**: Sub-harmonic component (25Hz); unusual behavior needing investigation
- **Severity**: HIGH 🟠 (but requires investigation)
- **What's happening**: Line appears at 25 Hz (BELOW 50 Hz fundamental!)
- **Cause**: Very specific equipment fault; requires expertise to diagnose
- **Visible pattern**: Bright line at lower frequency than normal fundamental
- **Real-world**: Advanced power electronic equipment fault; rare
- **Action**: Immediate investigation by expert; unique signature
- **Equipment at risk**: Specific equipment; depends on cause
- **AI Signal**: Unexpected frequency (25 Hz); not a standard harmonic
- **Why show this**: "Most operators wouldn't recognize this, but our AI does!"

**Sample anomaly_17.png**
- **Description**: Phase shift discontinuity; abrupt change during measurement
- **Severity**: HIGH 🟠
- **What's happening**: ABRUPT CHANGE in pattern during time window
- **Cause**: Phase shift event; three-phase component phase angle changed
- **Visible pattern**: Left side of image differs from right side; STEP CHANGE
- **Real-world**: Protection relay operated; phase relationship altered
- **Action**: Investigate protection events; check relay settings
- **Equipment at risk**: Synchronous machines, motors dependent on phase
- **AI Signal**: Discontinuity; pattern change across time axis

---

### Category 5: Complex Events (anomaly_18 to 24)

These represent **severe, multi-component faults** - when multiple anomalies occur simultaneously.

#### Why This Matters:
- Real faults are often complex
- Real systems don't just have one problem
- Model must handle combined anomalies
- These are the hardest to detect and most critical to catch

#### Sample Details:

**Sample anomaly_18.png** 🚨 **MOST SEVERE ANOMALY**
- **Description**: Arcing event; chaotic broadband high-energy pattern; critical
- **Severity**: CRITICAL 🔴
- **What's happening**: CHAOTIC, NOISY, VERY BRIGHT everywhere
- **Cause**: Electrical arc; loose connection arcing; equipment breakdown
- **Visible pattern**: Image looks like "WHITE NOISE"; fuzzy; bright throughout
- **Real-world**: FIRE HAZARD; immediate danger
- **Action**: IMMEDIATE DISCONNECTION required; emergency response
- **Equipment at risk**: Complete damage; potential fire
- **AI Signal**: Broadband noise covering entire spectrum; chaotic appearance
- **Why show this**: "This is what an emergency looks like - obvious even to humans"

**Sample anomaly_19.png**
- **Description**: Complex multi-event fault; multiple simultaneous anomalies
- **Severity**: CRITICAL 🔴
- **What's happening**: Multiple problems happening AT SAME TIME
- **Cause**: Cascading failures; multiple components failing
- **Visible pattern**: Combination of all bad patterns; frequency deviation + harmonics + transients
- **Real-world**: System in severe distress
- **Action**: Emergency procedures; system shutdown likely
- **Equipment at risk**: Everything connected
- **AI Signal**: Multiple distinct anomaly signatures combined

**Sample anomaly_20.png**
- **Description**: Intermittent arcing; repeating short spikes (extra sample)
- **Severity**: HIGH 🟠
- **What's happening**: Multiple arcing events separated in time
- **Cause**: Unstable arc; loose connection with partial continuity
- **Visible pattern**: Several bright SPIKES with DARK periods between
- **Real-world**: Intermittent contact; could become full arc soon
- **Action**: Find and clean/tighten connection before full failure
- **Equipment at risk**: Local equipment; potential fire
- **AI Signal**: Multiple transient spikes; regular or irregular spacing

**Sample anomaly_21.png**
- **Description**: Control system hunting; slow oscillations from controller instability (extra sample)
- **Severity**: MEDIUM 🟡
- **What's happening**: Slow oscillations visible in pattern
- **Cause**: Control system (voltage regulator, governor) hunting; not tuned correctly
- **Visible pattern**: Main line appears to WOBBLE SLOWLY
- **Real-world**: Control system unstable; needs retuning
- **Action**: Control system adjustment; tuning parameters need review
- **Equipment at risk**: Control system and controlled equipment
- **AI Signal**: Low-frequency oscillations; underdamped response

**Sample anomaly_22.png**
- **Description**: Combined harmonic and transient; both distortion and spike present (extra sample)
- **Severity**: HIGH 🟠
- **What's happening**: BOTH harmonic pollution AND a sharp spike
- **Cause**: Combination of equipment fault and sudden event
- **Visible pattern**: Multiple bright harmonics PLUS sharp vertical line
- **Real-world**: Multiple simultaneous problems
- **Action**: Investigate both the harmonic source and the spike event
- **Equipment at risk**: Affected equipment; widespread issues
- **AI Signal**: Combination of multiple anomaly types

**Sample anomaly_23.png**
- **Description**: Extra anomaly sample; transient-harmonic mixture (auto-added)
- **Severity**: HIGH 🟠
- **What's happening**: Combination of transient and harmonic distortion
- **Cause**: Equipment failure with cascading effects
- **Visible pattern**: Mixed appearance; some structure + some chaos
- **Real-world**: Complex fault condition
- **Action**: Full investigation needed; multiple potential causes
- **Equipment at risk**: Multiple systems potentially affected
- **AI Signal**: Hybrid appearance; both organized and chaotic elements

---

## Anomaly Detection Challenge

### Why Anomalies Are Hard to Detect:

1. **They're Rare** (only 16% of data)
   - More examples of normal than anomalous
   - Statistical algorithms naturally bias toward normal
   - ML models need special techniques (Focal Loss, class weighting)

2. **They're Diverse** (24 different types shown here, more in reality)
   - No single signature for "all anomalies"
   - Must learn many different failure patterns
   - Can't use simple rule-based detection

3. **They're Subtle** (some are hard to spot)
   - Not all anomalies look obviously wrong
   - anomaly_16.png (sub-harmonic) requires expertise
   - Some barely visible to untrained eye

4. **They're Context-Dependent**
   - What's normal in one area might be anomaly in another
   - Load variations create baseline differences
   - Equipment changes mean "normal" changes

### Why Our Model Matters:

- **24/7 Monitoring**: Humans can't watch constantly
- **Millisecond Detection**: Can respond faster than manual
- **Consistent**: Doesn't get tired or distracted
- **Scalable**: Can monitor thousands of points simultaneously
- **Learning**: Improves with more examples

---

## How to Interpret Anomalies

### Step 1: Scan for Obvious Problems
- Look for **vertical bright lines** (transients)
- Look for **line NOT at 50 Hz** (frequency deviation)
- Look for **very bright/chaotic appearance** (noise/arcing)

### Step 2: Look at Frequency Position
- Is the main line clearly at 50 Hz? (Good)
- Is it shifted? (Frequency deviation - BAD)
- Is it multiple positions? (Frequency changing - VERY BAD)

### Step 3: Examine Harmonic Content
- Are harmonics at expected positions (100, 150, 200 Hz)? (Normal)
- Are they too bright? (Harmonic distortion - BAD)
- Are there lines at unexpected positions? (Interharmonics - BAD)

### Step 4: Look at Noise Floor
- Is background dark? (Good)
- Is background hazy/fuzzy? (Noise/EMI - BAD)
- Is there significant broadband energy? (Transient/Arcing - CRITICAL)

### Step 5: Check Temporal Pattern
- Is pattern stable left to right? (Good)
- Does pattern change or shift? (Event during measurement - BAD)
- Are there sudden vertical spikes? (Transient - BAD)

---

## Teaching with Anomalies

### Quick Demonstration (5 minutes)
1. Show normal_01.png (normal reference)
2. Show anomaly_00.png (frequency swell)
3. Ask: "What's different?"
4. Answer: "The main frequency line moved up - frequency is too high!"
5. Explain: "Our model learned to spot this"

### Anomaly Categories Lesson (20 minutes)
Group anomalies by type and discuss:
- **Frequency issues**: anomaly_00 to 04
- **Harmonics**: anomaly_05 to 09
- **Transients**: anomaly_10 to 14
- **Phase issues**: anomaly_15 to 17
- **Complex**: anomaly_18 to 24

### Real-World Impact (10 minutes)
Discuss consequences:
- Frequency deviation → Could cause blackout
- Arcing → Fire hazard
- Harmonic distortion → Equipment damage
- Transients → Electronics failure

---

## Critical Insights

### What Makes an Anomaly Detectable:

✓ **Visual deviation from normal** - It looks different
✓ **Specific signature patterns** - Each fault has fingerprint
✓ **Interpretable to experts** - Grid operators can validate
✓ **Consistent representation** - Same type always similar

### Why Deep Learning Excels Here:

✓ **Learns visual patterns** - Like humans do, but faster
✓ **Handles many simultaneous features** - Not just one rule
✓ **Robust to variations** - Learns from 89 training examples
✓ **Explains decisions** - Attention maps show where/what

### Why Humans Need AI:

✓ **Humans get tired** - AI is always alert
✓ **Humans are slow** - AI is milliseconds
✓ **Humans are limited** - One person can watch ~20 points
✓ **AI is unlimited** - Can watch thousands

---

## Discussion Questions

**Q: Isn't anomaly_18.png too obvious? Why does it need AI?**
A: True, obvious faults are easy to detect. But most anomalies are subtle. The model must learn to catch the subtle ones AND the obvious ones. Plus, automated 24/7 detection catches things before they become obvious.

**Q: Why are there more normal samples (545) than anomaly samples (89)?**
A: In real grid data, normal operation is ~95% of the time. Anomalies are rare - that's why we need AI to catch them. Training with imbalanced data requires special techniques.

**Q: How does the AI actually learn from these samples?**
A: Via backpropagation on a deep CNN. The model learns filters that respond to "50 Hz line at position X" (normal) vs. "50 Hz line at position Y" (anomaly). Thousands of learned filters combine to make detection.

**Q: Could we just use simple rules like "if frequency ≠ 50 Hz, alarm"?**
A: Rule-based systems work for some anomalies but fail on complex ones. They need manual tuning for each case. AI learns rules automatically. Plus, anomaly_16 (sub-harmonic) isn't caught by simple frequency rules.

**Q: What happens if the model sees an anomaly type it never trained on?**
A: It might fail. That's why having diverse training examples matters. If a completely new fault type appears, the model might not catch it. This is a real limitation of supervised learning.

---

## Summary

**These 24 anomaly samples represent 5 major fault categories:**

1. **Frequency Deviations** (anomaly_00-04): Grid frequency moving away from 50 Hz
2. **Harmonic Distortion** (anomaly_05-09): Excessive harmonic pollution
3. **Transient Events** (anomaly_10-14): Sudden sharp disturbances
4. **Phase Issues** (anomaly_15-17): Three-phase imbalance or sub-harmonics
5. **Complex Faults** (anomaly_18-24): Multiple simultaneous problems

**All are visually distinct from normal operation** - that's what makes them learnable by deep neural networks.

**The challenge isn't detection - humans can see these are different. The challenge is:**
- ✓ Catching them in real-time (24/7 monitoring)
- ✓ Scaling to thousands of locations
- ✓ Handling subtle cases (anomaly_16)
- ✓ Responding before damage escalates
- ✓ Learning patterns automatically (no manual rules)

---

## Next Steps

1. **Compare with normal operations folder** - See the contrast
2. **Spend 10 minutes browsing all 24 samples** - Get intuition
3. **Pick the most interesting anomaly to you** - What draws your attention?
4. **Read DATASET_EXPLANATION.md** - For deeper technical context
5. **Look at model attention maps** - To understand AI's perspective
6. **Review real project results** - See model performance on test data

---

*For questions about specific samples, refer to their individual descriptions above. For overall context, see DATASET_EXPLANATION.md.*
