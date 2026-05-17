"""
test_calibrated_fusion.py — Quick test of the new calibrated fusion experiment

Runs both exp06 (original fusion) and exp06_fusion_normalized (with calibration)
on the BEREL + Multilabel stratified (paper-style) + Aug condition and compares results.

Usage:
    python test_calibrated_fusion.py

The comparison will show:
- Original fusion F1
- Calibrated fusion F1
- Temperature parameters learned during calibration
- Calibration quality (ECE before/after)
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add experiments to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from common import configure_model_environment, suppress_output_if_needed
import experiment_06_fusion_regular_and_cascaded as exp06_original
import experiment_06_fusion_normalized as exp06_calibrated


def run_comparison():
    """Run both fusion methods on BEREL + Multilabel stratified (paper-style) + Aug."""
    
    print("=" * 80)
    print("  TESTING CALIBRATED FUSION NORMALIZATION")
    print("=" * 80)
    print(f"\nTest Date: {datetime.now().isoformat()}")
    print("\nCondition: BEREL 3.0 + Multilabel stratified (paper-style) + Aug")
    print("Dataset: outputs/exp08/augmented_train.json + outputs/exp08/augmented_eval.json")
    
    # Use BEREL model
    os.environ["THESIS_MODEL_NAME"] = str(PROJECT_ROOT / "models" / "BEREL_3.0")
    
    # Use the augmented condition
    presplit_train = PROJECT_ROOT / "outputs" / "exp08" / "augmented_train.json"
    presplit_eval = PROJECT_ROOT / "outputs" / "exp08" / "augmented_eval.json"
    
    if not presplit_train.exists() or not presplit_eval.exists():
        print(f"\nERROR: Presplit files not found!")
        print(f"  Train: {presplit_train} (exists: {presplit_train.exists()})")
        print(f"  Eval:  {presplit_eval} (exists: {presplit_eval.exists()})")
        return
    
    os.environ["THESIS_PRESPLIT_TRAIN_JSON"] = str(presplit_train)
    os.environ["THESIS_PRESPLIT_EVAL_JSON"] = str(presplit_eval)
    os.environ["THESIS_SPLIT_SEED"] = "42"
    
    print(f"  Train: {presplit_train}")
    print(f"  Eval:  {presplit_eval}")
    
    # Run original fusion
    print("\n" + "─" * 80)
    print("  Running ORIGINAL FUSION (exp06)...")
    print("─" * 80)
    try:
        result_orig = exp06_original.run()
        f1_orig = result_orig.get("f1")
        precision_orig = result_orig.get("precision")
        recall_orig = result_orig.get("recall")
        print(f"✓ ORIGINAL FUSION Complete")
        print(f"  F1={f1_orig:.4f}, Precision={precision_orig:.4f}, Recall={recall_orig:.4f}")
    except Exception as e:
        print(f"✗ ORIGINAL FUSION Failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run calibrated fusion
    print("\n" + "─" * 80)
    print("  Running CALIBRATED FUSION (exp06_fusion_normalized)...")
    print("─" * 80)
    try:
        result_cal = exp06_calibrated.run()
        f1_cal = result_cal.get("f1")
        precision_cal = result_cal.get("precision")
        recall_cal = result_cal.get("recall")
        print(f"✓ CALIBRATED FUSION Complete")
        print(f"  F1={f1_cal:.4f}, Precision={precision_cal:.4f}, Recall={recall_cal:.4f}")
    except Exception as e:
        print(f"✗ CALIBRATED FUSION Failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Compare results
    print("\n" + "=" * 80)
    print("  COMPARISON RESULTS")
    print("=" * 80)
    
    print(f"\n{'Metric':<20} {'Original':<15} {'Calibrated':<15} {'Delta':<15} {'% Gain':<10}")
    print("-" * 80)
    
    # F1
    delta_f1 = f1_cal - f1_orig
    pct_f1 = (delta_f1 / f1_orig * 100) if f1_orig != 0 else 0
    print(f"{'F1':<20} {f1_orig:.4f}          {f1_cal:.4f}          {delta_f1:+.4f}          {pct_f1:+.2f}%")
    
    # Precision
    delta_p = precision_cal - precision_orig
    pct_p = (delta_p / precision_orig * 100) if precision_orig != 0 else 0
    print(f"{'Precision':<20} {precision_orig:.4f}          {precision_cal:.4f}          {delta_p:+.4f}          {pct_p:+.2f}%")
    
    # Recall
    delta_r = recall_cal - recall_orig
    pct_r = (delta_r / recall_orig * 100) if recall_orig != 0 else 0
    print(f"{'Recall':<20} {recall_orig:.4f}          {recall_cal:.4f}          {delta_r:+.4f}          {pct_r:+.2f}%")
    
    # Calibration parameters
    print("\n" + "-" * 80)
    cal_params = result_cal.get("calibration_parameters", {})
    if cal_params:
        print(f"\nCalibration Parameters Learned:")
        print(f"  Regular NER Temperature:  {cal_params.get('regular_temperature', 'N/A'):.4f}")
        print(f"  Cascade Temperature:      {cal_params.get('cascade_temperature', 'N/A'):.4f}")
        print(f"\nCalibration Quality (ECE - lower is better):")
        print(f"  Regular NER Before/After: {cal_params.get('regular_ece_before', 'N/A'):.4f} → {cal_params.get('regular_ece_after', 'N/A'):.4f}")
        print(f"  Cascade Before/After:     {cal_params.get('cascade_ece_before', 'N/A'):.4f} → {cal_params.get('cascade_ece_after', 'N/A'):.4f}")
    
    # Export details
    print("\n" + "-" * 80)
    print(f"\nOutput Files:")
    print(f"  Original: {result_orig.get('metrics_file')}")
    print(f"  Calibrated: {result_cal.get('metrics_file')}")
    
    print("\n" + "=" * 80)
    if f1_cal > f1_orig:
        print(f"✓ CALIBRATION IMPROVED FUSION by {delta_f1:+.4f} F1 ({pct_f1:+.2f}%)")
    else:
        print(f"✗ Calibration did not improve in this case (delta={delta_f1:+.4f})")
    print("=" * 80)
    
    # Save comparison
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "condition": "BEREL 3.0 + Multilabel stratified (paper-style) + Aug",
        "original_fusion": {
            "f1": float(f1_orig),
            "precision": float(precision_orig),
            "recall": float(recall_orig),
            "metrics_file": result_orig.get("metrics_file"),
        },
        "calibrated_fusion": {
            "f1": float(f1_cal),
            "precision": float(precision_cal),
            "recall": float(recall_cal),
            "calibration_parameters": cal_params,
            "metrics_file": result_cal.get("metrics_file"),
        },
        "delta": {
            "f1": float(delta_f1),
            "f1_percent_gain": float(pct_f1),
            "precision": float(delta_p),
            "recall": float(delta_r),
        }
    }
    
    comparison_path = PROJECT_ROOT / "outputs" / "calibration_comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nComparison saved to: {comparison_path}")


if __name__ == "__main__":
    run_comparison()
