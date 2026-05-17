"""
Quick reference: Running the Better Fusion Strategies

Four new fusion approaches have been implemented and registered:
  1. Entropy-Weighted Fusion     (experiment_06_fusion_entropy)
  2. Learned Weights Fusion      (experiment_06_fusion_learned_weights)
  3. SVM Router Fusion           (experiment_06_fusion_svm)
  4. Ensemble Rules Fusion       (experiment_06_fusion_ensemble_rules)

Plus the original for comparison:
  0. Original Fusion             (experiment_06)
  0b. Calibrated Fusion          (experiment_06_fusion_normalized)

ALL OPTIONS for running via cross-comparison script:
==============================================================================
"""

# Option 1: Quick test - compare all fusion methods on one condition
# (DictaBERT + Multilabel + Aug - best condition)
"""
python run_cross_data_model_comparison.py \
  --experiments 06,06_fusion_entropy,06_fusion_learned_weights,06_fusion_svm,06_fusion_ensemble_rules \
  --models dictabert
"""

# Option 2: Full comparison - all fusion methods on all conditions
"""
python run_cross_data_model_comparison.py \
  --experiments 06,06_fusion_entropy,06_fusion_learned_weights,06_fusion_svm,06_fusion_ensemble_rules
"""

# Option 3: Include calibrated fusion for full comparison
"""
python run_cross_data_model_comparison.py \
  --experiments 06,06_fusion_normalized,06_fusion_entropy,06_fusion_learned_weights,06_fusion_svm,06_fusion_ensemble_rules
"""

# Option 4: Just test SVM router fusion across all models
"""
python run_cross_data_model_comparison.py --experiments 06_fusion_svm
"""

# Option 5: Resume from checkpoint (in case interrupted)
"""
set THESIS_CROSS_EXPERIMENTS=06,06_fusion_entropy,06_fusion_learned_weights,06_fusion_svm,06_fusion_ensemble_rules
python run_cross_data_model_comparison.py --resume
"""

# EXPECTED RESULTS
# ==============================================================================
# Output files will be created in: outputs/cross_comparison/
# - cross_comparison_<timestamp>.json -> detailed results
# - cross_comparison_<timestamp>.xlsx -> spreadsheet with all metrics
# - cross_comparison_latest.json / .xlsx -> latest results (overwritten each run)

# Key metrics to compare:
# - F1 score: absolute performance
# - Delta from Exp06: improvement/degradation vs original
# - Breakdown by model (DictaBERT vs BEREL)
# - Breakdown by condition (baseline, multilabel+aug, etc.)

# QUICK ANALYSIS AFTER RUNNING
# ==============================================================================
# 1. Open cross_comparison_latest.xlsx
# 2. Look at sheets:
#    - all_runs: See all experiment results
#    - model_comparison: Compare models side-by-side
#    - experiment_details: Fusion-specific statistics
# 3. Filter by experiment_id to compare fusion variants:
#    - exp06, exp06_fusion_entropy, exp06_fusion_learned_weights,
#      exp06_fusion_svm, exp06_fusion_ensemble_rules
# 4. Calculate deltas: F1(new_method) - F1(original_fusion)
#    - Positive delta = improvement
#    - Look for consistency: which method helps across most conditions?

print(
    """
Better Fusion Strategies - Ready to Run

QUICK START (full retraining):
  python run_cross_data_model_comparison.py \\
    --experiments 06,06_fusion_entropy,06_fusion_learned_weights,06_fusion_svm,06_fusion_ensemble_rules

FAST START (no retraining — ready-results mode):
  # Step 1: train base models once
  python run_cross_data_model_comparison.py --experiments 01,04

  # Step 2: run all ready fusion variants (seconds each)
  python run_cross_data_model_comparison.py \\
    --experiments 05_ready,06_ready,06_normalized_ready,06_entropy_ready,06_learned_ready,06_ensemble_ready,06_svm_ready

READY-RESULTS MODE:
  Ready variants read Exp01 + Exp04 saved outputs and only perform fusion.
  This lets you iterate on fusion rules/thresholds without waiting hours.

  Ready IDs:
    05_ready             - Cascaded + consistency (from Exp04)
    06_ready             - Confidence comparison
    06_normalized_ready  - Temperature-calibrated
    06_entropy_ready     - Entropy-weighted
    06_learned_ready     - Learned weights (alpha sweep)
    06_ensemble_ready    - Rule-based ensemble
    06_svm_ready         - SVM disagreement router

  Environment variables (optional):
    THESIS_READY_EXP01_XLSX  - path to Exp01 output xlsx
    THESIS_READY_EXP04_XLSX  - path to Exp04 output xlsx

This will:
  - Run original fusion (exp06) OR ready variants
  - Compare across all models and data conditions
  - Save results to outputs/cross_comparison/cross_comparison_latest.xlsx

EXPECTED RUNTIME:
  - Full retraining: ~1-2.5 hours (5 methods x 2 models x 4 conditions)
  - Ready mode: ~1-2 minutes total (fusion-only, no GPU needed)

OUTPUT INTERPRETATION:
  1. Look for F1 improvement vs Exp06 (original)
  2. Check consistency: which method helps most often?
  3. Entropy typically helps uncertainty-heavy cases
  4. Learned weights helps when one source is globally stronger
  5. SVM router helps when disagreement patterns are feature-dependent
  6. Ensemble rules helps when interpretability is prioritized
"""
)
