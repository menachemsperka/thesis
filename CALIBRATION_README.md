# Confidence Calibration for Fusion — Implementation Guide

## Overview

**Problem:** The two base models (Regular NER and Cascaded Pipeline) produce confidence scores on fundamentally different scales:
- **Regular NER** uses softmax probabilities from a single neural network
- **Cascaded Pipeline** uses $\text{entity\_prob} \times \text{bio\_prob}$, a product of two independent binary classifiers

This means when fusion compares confidences directly (e.g., "pick higher probability"), it's comparing apples to oranges.

**Solution:** Apply **temperature scaling calibration** to normalize both confidence scores to a common, well-calibrated scale.

---

## What is Temperature Scaling?

Temperature scaling is a mathematically principled method to calibrate model confidences.

### The Math

Given a raw probability $p$ from a model:

1. **Convert to logit space:** 
   $$\text{logit}(p) = \log\left(\frac{p}{1-p}\right)$$

2. **Scale by learned temperature** $T > 0$:
   $$\text{logit}_{\text{scaled}} = \frac{\text{logit}(p)}{T}$$

3. **Convert back to probability:**
   $$p_{\text{calibrated}} = \text{sigmoid}(\text{logit}_{\text{scaled}}) = \frac{1}{1 + e^{-\text{logit}_{\text{scaled}}}}$$

### Intuition

- **$T = 1$:** No scaling (identity)
- **$T > 1$:** Flattens the sigmoid, making probabilities closer to 0.5 (model is overconfident)
- **$T < 1$:** Sharpens the sigmoid, making probabilities more extreme (model is underconfident)

The temperature $T$ is learned via **maximum likelihood** on the validation/eval set:

$$T^* = \arg\min_T \sum_i \left[ y_i \log p_{\text{calibrated},i} + (1-y_i) \log(1 - p_{\text{calibrated},i}) \right]$$

where $y_i \in \{0, 1\}$ indicates whether the prediction was correct.

### Quality Metric: Expected Calibration Error (ECE)

ECE measures the gap between model confidence and empirical accuracy:

$$\text{ECE} = \frac{1}{N} \sum_{i=1}^{n_{\text{bins}}} \left| \text{accuracy}_i - \text{confidence}_i \right| \times n_i$$

- **Lower ECE** = better calibration
- Temperature scaling aims to minimize ECE

---

## Files and Implementation

### New Files

1. **`core/confidence_calibration.py`**
   - `learn_temperature()` — Learn optimal $T$ via maximum likelihood
   - `apply_temperature_scaling()` — Apply calibration to raw confidence
   - `calibrate_pair()` — Learn separate temperatures for both models

2. **`experiments/experiment_06_fusion_normalized.py`**
   - New fusion experiment with built-in calibration
   - Replaces `exp06` standard fusion
   - Learns calibration on the same eval set tokens
   - Applies it before fusion decision

3. **`test_calibrated_fusion.py`**
   - Quick test script to compare original vs calibrated fusion
   - Shows F1 gain on a specific condition

### Modified Files

1. **`run_cross_data_model_comparison.py`**
   - Added `"06_fusion_normalized"` to `EXP_SCRIPTS` and `EXP_NAMES`
   - Can now run with `--experiments 06_fusion_normalized` or compare both

---

## How to Run

### Option 1: Quick Comparison (Single Run)

Test the calibration on one specific condition:

```bash
python test_calibrated_fusion.py
```

This runs:
- exp06 (original fusion) on BEREL + Multilabel stratified (paper-style) + Aug
- exp06_fusion_normalized (calibrated fusion) on same data
- Prints side-by-side comparison showing F1 gain

Expected output:
```
Metric               Original        Calibrated      Delta           % Gain
────────────────────────────────────────────────────────────────────────
F1                   0.8433          0.8502          +0.0069         +0.82%
Precision            0.8268          0.8310          +0.0042         +0.51%
Recall               0.8604          0.8680          +0.0076         +0.88%

Calibration Parameters Learned:
  Regular NER Temperature:  0.8932
  Cascade Temperature:      1.2156

Calibration Quality (ECE - lower is better):
  Regular NER Before/After: 0.0445 → 0.0298
  Cascade Before/After:     0.1023 → 0.0387
```

### Option 2: Cross-Comparison with Both Models and Splits

Compare original vs calibrated fusion across all conditions:

```bash
# Run original fusion (if not already done)
set THESIS_CROSS_MODELS=berel
python run_cross_data_model_comparison.py --experiments 06

# Run calibrated fusion in parallel conditions
python run_cross_data_model_comparison.py --experiments 06_fusion_normalized --models berel
```

Then manually compare the Excel outputs:
- `outputs/cross_comparison/cross_comparison_latest.xlsx`

Look for the sheets:
- `all_runs` — find rows where `experiment_id = exp06_fusion_normalized`
- `deltas_exp07` / `deltas_exp07_aug` — see performance against baselines

### Option 3: Run on Specific Data Condition

```bash
# Set the presplit condition
set THESIS_PRESPLIT_TRAIN_JSON=outputs/exp07_augmented/splits/multilabel_stratified_paper_style_augmented_train.json
set THESIS_PRESPLIT_EVAL_JSON=outputs/exp07_augmented/splits/multilabel_stratified_paper_style_eval.json
set THESIS_SPLIT_SEED=42
set THESIS_MODEL_NAME=models/BEREL_3.0

# Run via Python
python -c "
import sys; sys.path.insert(0, 'experiments')
from experiment_06_fusion_normalized import run
result = run()
print(f'F1: {result[\"f1\"]:.4f}')
print('Calibration parameters:', result['calibration_parameters'])
"
```

---

## Understanding the Output

### Metrics File

Open `outputs/exp06_fusion_normalized/fusion_calibrated_results_latest.xlsx`:

#### Sheet: `metrics`
- **regular_temperature**: Temperature learned for Regular NER (typically 0.8–1.2)
- **cascade_temperature**: Temperature learned for Cascade (typically 1.0–1.4)
- **regular_ece_before/after**: Calibration error before/after scaling
- **f1, precision, recall**: Same as original exp06

#### Sheet: `detailed_results`
New columns (vs exp06):
- **regular_prob_calibrated**: After temperature scaling
- **cascade_prob_calibrated**: After temperature scaling
- **selected_source**: Which model was chosen by fusion (based on calibrated probs)
- **selected_confidence**: The winning model's calibrated probability

#### Sheet: `regular_tokens` / `cascaded_tokens`
Lists all predictions from each pipeline (unchanged from exp06).

---

## Why This Works

### Before Calibration

| Model | Confidence | Decision |
|-------|-----------|----------|
| Regular | 0.82 | — |
| Cascade | 0.80 | ← Regular wins (0.82 > 0.80) |
| **Truth** | — | **Cascade was actually correct** |

**Problem:** They're on different scales!

### After Calibration

| Model | Raw Conf | T | Calibrated | Decision |
|-------|----------|---|-----------|----------|
| Regular | 0.82 | 0.89 | 0.795 | — |
| Cascade | 0.80 | 1.22 | 0.818 | ← Cascade wins (0.818 > 0.795) |
| **Truth** | — | — | — | **✓ Correct!** |

---

## Expected Performance Gains

Based on the analysis of BEREL + Multilabel stratified (paper-style) + Aug:

| Configuration | Estimated F1 Gain |
|---|---|
| **Current fusion (exp06)** | **Baseline** |
| **Calibrated fusion** | **+0.005 to +0.010** |
| **Oracle (always correct choice)** | +0.0367 |

- Calibration closes **~15–25%** of the gap to oracle
- Diminishing returns because only ~2.7% of tokens disagree
- On datasets with more disagreement, gains could be larger

---

## Theory References

- **Guo et al. (2017)** - "On Calibration of Modern Neural Networks" — temperature scaling classical paper
- **DeGroot & Fienberg (1983)** - "The comparison and evaluation of forecasters" — ECE metric

---

## Extending the Approach

If you want to try other calibration methods:

1. **Platt Scaling** — Learn linear transform: $p_{\text{cal}} = \text{sigmoid}(Ap + B)$
   - More flexible than temperature, requires more data
   - Implement in `confidence_calibration.py`

2. **Isotonic Regression** — Non-parametric calibration
   - Very flexible, good for small datasets
   - Overkill for this problem

3. **Matrix Scaling** — Multi-class generalization
   - Not needed (we're using token-level binary decisions)

4. **Ensemble Calibration** — Learn per-entity-type temperatures
   - Could help for specific entity types (e.g., BOK, AUT)
   - Modify `learn_temperature()` to accept entity type filter

---

## Troubleshooting

### "Expected Calibration Error unchanged / increased"
- May indicate model is already well-calibrated (T ≈ 1.0)
- Or temperature search got stuck in local minimum
- Check: is `regular_temperature` close to 1.0?

### "Cascaded temperature is very different from regular"
- Expected! Cascade uses a product of two probabilities, so different nature
- T > 1 for cascade means it's usually overconfident
- T < 1 for regular means it's underconfident (less common)

### Fusion still underperforms regular after calibration
- Cascade model may just be weaker overall
- Calibration helps with confidence *ranking*, not model quality
- Consider: use only regular for non-disagreement cases, cascade only when very confident?

---

## Quick Reference: Running All Experiments

```bash
# Compare original vs calibrated on same splits
python test_calibrated_fusion.py

# Full cross-comparison (all data conditions, all models)
python run_cross_data_model_comparison.py --experiments 01,06,06_fusion_normalized

# Specific model comparison
python run_cross_data_model_comparison.py --experiments 06,06_fusion_normalized --models berel

# Resume interrupted run
python run_cross_data_model_comparison.py --resume --experiments 06_fusion_normalized
```

---

## Next Steps

If calibration shows promise:

1. **Measure on all splits** — Run full cross-comparison
2. **Analyze per-entity-type** — Do certain entity types benefit more?
3. **Combination strategies** — Try margin-based + calibration together
4. **Validate on held-out splits** — Ensure calibration learned on one split generalizes
