# Statistical Significance Guide for Model Comparison

## Executive Summary

To achieve **α = 0.05 (5% significance level)** for comparing models/methods in your thesis, you need **sufficient paired observations** (runs) to apply statistical tests. The key insight: **seeds are your primary source of variance**, and you need **10-30 seeds minimum** for robust statistical claims.

---

## 1. What You Need to Vary

### 1.1 Seeds (PRIMARY — Most Important)

**What it controls:**
- Neural network weight initialization
- Training data shuffling
- Random decisions during training (dropout, etc.)

**Current default:** `--num-seeds=3` (TOO FEW for significance)

**Recommended:** `--num-seeds=10` minimum, `--num-seeds=20` for robust results

**Command:**
```bash
python run_cross_data_model_comparison.py --num-seeds 20 --models dictabert,berel
```

### 1.2 Data Splits (SECONDARY — Built into exp07)

Your experiment 07 already creates different split strategies, which contribute to observational variance. These are systematic conditions (not random), so they're useful for generalization but don't substitute for seed variance.

### 1.3 Runs vs Seeds — What's the Difference?

| Term | Meaning | Effect |
|------|---------|--------|
| **Seeds** | Different random initializations | Creates paired observations for the same data condition |
| **Splits** | Different train/test partitions | Tests generalization across data conditions |
| **Runs** | Repeating the whole experiment | Same as seeds if you change the seed each time |

**Bottom line:** For significance testing, **seeds = runs**. Each seed creates one paired observation.

---

## 2. Why 3 Seeds is Not Enough

With only 3 observations (seeds), even if Model A beats Model B every time:

- A paired t-test needs more degrees of freedom for power
- p-values will rarely reach < 0.05 unless differences are huge
- Statistically: with 3 observations, you can only detect very large effect sizes

### Sample Size Requirements for α = 0.05

| Effect Size | Required Seeds | Power |
|-------------|----------------|-------|
| Large (d=0.8) | 10 seeds | ~75% |
| Large (d=0.8) | 15 seeds | ~88% |
| Medium (d=0.5) | 20 seeds | ~75% |
| Medium (d=0.5) | 30 seeds | ~87% |
| Small (d=0.2) | 50+ seeds | ~50%+ |

**Practical recommendation: Use 10-20 seeds** for detecting meaningful differences.

---

## 3. Statistical Tests to Use

### 3.1 Paired t-test (Recommended for F1 scores)

Use when comparing two methods across multiple seeds on the same conditions.

```python
from scipy.stats import ttest_rel

# F1 scores from method A across 20 seeds
f1_method_a = [0.72, 0.74, 0.73, ...]  # 20 values
# F1 scores from method B across 20 seeds (same seeds!)
f1_method_b = [0.71, 0.72, 0.73, ...]  # 20 values

t_stat, p_value = ttest_rel(f1_method_a, f1_method_b)
print(f"t = {t_stat:.3f}, p = {p_value:.4f}")

if p_value < 0.05:
    print("Statistically significant difference at α=0.05")
```

### 3.2 Wilcoxon Signed-Rank Test (Non-parametric alternative)

Use when you can't assume normality or have small samples.

```python
from scipy.stats import wilcoxon

stat, p_value = wilcoxon(f1_method_a, f1_method_b)
print(f"W = {stat:.3f}, p = {p_value:.4f}")
```

### 3.3 Bootstrap Confidence Intervals (Most robust)

Use for computing confidence intervals around differences.

```python
import numpy as np

def bootstrap_ci(a, b, n_bootstrap=10000, ci=0.95):
    """Compute bootstrap CI for the mean difference (a - b)."""
    diffs = np.array(a) - np.array(b)
    boot_diffs = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(diffs, size=len(diffs), replace=True)
        boot_diffs.append(np.mean(sample))
    
    lower = np.percentile(boot_diffs, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_diffs, (1 + ci) / 2 * 100)
    return np.mean(diffs), lower, upper

mean_diff, ci_low, ci_high = bootstrap_ci(f1_method_a, f1_method_b)
print(f"Mean Δ = {mean_diff:.4f}, 95% CI = [{ci_low:.4f}, {ci_high:.4f}]")

# If CI doesn't include 0, difference is significant at α=0.05
if ci_low > 0 or ci_high < 0:
    print("Statistically significant (CI excludes 0)")
```

---

## 4. Experiment Design for Significance

### 4.1 Paired Design (REQUIRED)

**Critical:** Use the SAME seed for both methods you're comparing.

```bash
# GOOD: Same seeds for both models
python run_cross_data_model_comparison.py --models dictabert --num-seeds 20
python run_cross_data_model_comparison.py --models berel --num-seeds 20 
# Then pair seed 1 with seed 1, seed 2 with seed 2, etc.

# BETTER: Run together with same seed sequence
python run_cross_data_model_comparison.py --models dictabert,berel --num-seeds 20
```

### 4.2 Run Configuration Recommendations

| Goal | Seeds | Estimated Time | Command Example |
|------|-------|----------------|-----------------|
| Quick sanity check | 3 | ~1 hour | `--num-seeds 3` |
| Moderate confidence | 10 | ~4 hours | `--num-seeds 10` |
| Publication-ready | 20 | ~8 hours | `--num-seeds 20` |
| High-confidence | 30 | ~12 hours | `--num-seeds 30` |

---

## 5. How to Run for Statistical Significance

### 5.1 Full Publication-Quality Run (RECOMMENDED)

```bash
# Run with 20 seeds, save models for fusion reuse
python run_cross_data_model_comparison.py ^
    --num-seeds 20 ^
    --models dictabert,berel ^
    --experiments 01,03,04,05_ready,06_ready,06_svm_ready ^
    --save-models ^
    --base-mode auto

# Or with all experiments including all fusion variants
python run_cross_data_model_comparison.py ^
    --num-seeds 20 ^
    --models dictabert,berel ^
    --experiments 01,03,04,05_ready,06_ready,06_normalized_ready,06_entropy_ready,06_learned_ready,06_ensemble_ready,06_svm_ready ^
    --save-models ^
    --base-mode auto
```

### 5.2 Using Environment Variables (Alternative)

```bash
set THESIS_CROSS_NUM_SEEDS=20
set THESIS_SAVE_TRAINED_MODELS=1
python run_cross_data_model_comparison.py
```

### 5.3 Resume After Interruption

```bash
python run_cross_data_model_comparison.py --resume --num-seeds 20 --save-models
```

### 5.4 Outputs Location

After running, you'll find:
- **JSON results**: `outputs/cross_comparison/cross_comparison_latest.json`
- **Excel results**: `outputs/cross_comparison/cross_comparison_latest.xlsx`
- **Saved models**: `outputs/trained_models/exp{01,04}_{model}_{condition}_seed{N}/`
- **Progress checkpoint**: `outputs/cross_comparison/cross_comparison_progress_latest.json`

### 5.5 Reuse Saved Models for New Fusion Experiments

Once base models are saved, rerun fusion experiments without retraining:

```bash
# Skip base training, just reuse cached artifacts
python run_cross_data_model_comparison.py ^
    --num-seeds 20 ^
    --models dictabert,berel ^
    --experiments 05_ready,06_ready,06_svm_ready,06_normalized_ready ^
    --base-mode reuse
```

---

## 6. Interpreting Results

### 6.1 Reporting Template

In your thesis, report:

```
DictaBERT achieved mean F1 = 0.742 ± 0.015 (SD over 20 seeds), compared to 
BEREL's 0.731 ± 0.018. A paired t-test confirmed the difference was 
statistically significant (t(19) = 2.84, p = 0.010 < 0.05), with DictaBERT 
outperforming BEREL by an average of 1.1 F1 points.
```

### 6.2 Multi-Method Comparison

When comparing >2 methods, use **Friedman test** with **post-hoc Nemenyi test** or **Bonferroni correction**:

```python
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp  # pip install scikit-posthocs

# F1 scores: rows = seeds, columns = methods
data = np.array([
    [0.72, 0.71, 0.74],  # seed 1: [method_A, method_B, method_C]
    [0.74, 0.72, 0.73],  # seed 2
    # ... 20 seeds
])

stat, p = friedmanchisquare(*data.T)
print(f"Friedman χ² = {stat:.3f}, p = {p:.4f}")

if p < 0.05:
    # Post-hoc pairwise comparisons
    posthoc = sp.posthoc_nemenyi_friedman(data)
    print(posthoc)
```

---

## 7. Summary: Quick Checklist

| Requirement | Status | Action |
|-------------|--------|--------|
| Seeds ≥ 10 | ❓ Check `--num-seeds` | Set `--num-seeds 20` |
| Paired design | ❓ Same seeds both methods | Run both models together |
| Correct metric | ✅ F1 score | Use entity-level F1 |
| Statistical test | ❓ Not implemented | Add scipy.stats.ttest_rel |
| Effect size | ❓ Not reported | Calculate Cohen's d |
| Confidence intervals | ❓ Not computed | Use bootstrap CI |

---

## 8. Code Addition: Statistical Analysis Script

Add this to your post-processing to compute significance:

```python
# File: analyze_significance.py
"""Compute pairwise statistical significance from cross_comparison results."""

import json
import numpy as np
from pathlib import Path
from scipy.stats import ttest_rel, wilcoxon
from itertools import combinations

def load_results(json_path):
    """Load cross_comparison JSON results."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('results', [])

def group_by_seed(results, model_id, experiment_id, condition_key):
    """Extract F1 scores across seeds for a specific configuration."""
    f1_scores = []
    for r in results:
        if (r.get('model_id') == model_id and 
            r.get('experiment_id') == experiment_id and
            r.get('condition_key') == condition_key):
            f1 = r.get('f1')
            if f1 is not None:
                f1_scores.append(f1)
    return f1_scores

def pairwise_significance(f1_a, f1_b, alpha=0.05):
    """Compute paired t-test and Wilcoxon test."""
    if len(f1_a) != len(f1_b) or len(f1_a) < 3:
        return None
    
    t_stat, t_pval = ttest_rel(f1_a, f1_b)
    w_stat, w_pval = wilcoxon(f1_a, f1_b)
    mean_diff = np.mean(f1_a) - np.mean(f1_b)
    
    return {
        'mean_diff': mean_diff,
        't_stat': t_stat,
        't_pval': t_pval,
        'w_stat': w_stat,
        'w_pval': w_pval,
        'significant_ttest': t_pval < alpha,
        'significant_wilcoxon': w_pval < alpha,
        'n_pairs': len(f1_a),
    }

if __name__ == '__main__':
    results_path = Path('outputs/cross_comparison/cross_comparison_latest.json')
    results = load_results(results_path)
    
    # Example: Compare dictabert vs berel on exp01, baseline condition
    f1_dictabert = group_by_seed(results, 'dicta-il/dictabert', 'exp01', 'exp07_baseline')
    f1_berel = group_by_seed(results, 'dicta-il/BEREL_3.0', 'exp01', 'exp07_baseline')
    
    sig = pairwise_significance(f1_dictabert, f1_berel)
    if sig:
        print(f"DictaBERT vs BEREL (exp01, baseline):")
        print(f"  Mean Δ = {sig['mean_diff']:.4f}")
        print(f"  Paired t-test: t={sig['t_stat']:.3f}, p={sig['t_pval']:.4f}")
        print(f"  Wilcoxon: W={sig['w_stat']:.3f}, p={sig['w_pval']:.4f}")
        print(f"  Significant at α=0.05: {sig['significant_ttest']}")
```

---

## 9. Quick Reference Commands

### Basic Publication-Quality Run
```bash
python run_cross_data_model_comparison.py --num-seeds 20 --save-models
```

### Full Multi-Model Comparison
```bash
python run_cross_data_model_comparison.py ^
    --num-seeds 20 ^
    --models dictabert,berel ^
    --experiments 01,03,04,05_ready,06_ready,06_normalized_ready,06_entropy_ready,06_learned_ready,06_ensemble_ready,06_svm_ready ^
    --save-models ^
    --base-mode auto
```

### Four Models Comparison (most thorough)
```bash
python run_cross_data_model_comparison.py ^
    --num-seeds 20 ^
    --models dictabert,berel,hero,alephbertgimmel ^
    --experiments 01,04,05_ready,06_ready,06_svm_ready ^
    --save-models ^
    --base-mode auto
```

### Reuse Trained Models, Test New Fusion Methods
```bash
python run_cross_data_model_comparison.py ^
    --experiments 06_normalized_ready,06_entropy_ready,06_learned_ready ^
    --base-mode reuse
```

### Resume Interrupted Run
```bash
python run_cross_data_model_comparison.py --resume --save-models
```

### Check Cached Base Artifacts
```bash
python run_cross_data_model_comparison.py --list-base-cache
```

### Rebuild Excel/JSON from Checkpoint (no retraining)
```bash
python run_cross_data_model_comparison.py --rebuild-from-checkpoint
```

---

## 10. Output Files

| File | Location | Contents |
|------|----------|----------|
| **JSON results** | `outputs/cross_comparison/cross_comparison_latest.json` | All metrics, can be parsed for significance testing |
| **Excel results** | `outputs/cross_comparison/cross_comparison_latest.xlsx` | Same data with pivots and delta sheets |
| **Saved models** | `outputs/trained_models/exp{01,04}_{model}_{condition}_seed{N}/` | Reusable trained weights |
| **Progress checkpoint** | `outputs/cross_comparison/cross_comparison_progress_latest.json` | For resume capability |
| **Base index** | `outputs/cross_comparison/cross_comparison_base_ready_index.json` | Cached exp01/exp04 artifacts |

---

## References

1. Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *JMLR*.
2. Dror, R., et al. (2018). The hitchhiker's guide to testing statistical significance in NLP. *ACL*.
3. Berg-Kirkpatrick, T., et al. (2012). An empirical investigation of statistical significance in NLP. *EMNLP*.
