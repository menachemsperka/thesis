"""Analyze whether calibration fixed the fusion F1 issue."""

import json
from pathlib import Path

json_file = Path("outputs/cross_comparison/cross_comparison_latest.json")
with open(json_file) as f:
    data = json.load(f)

results = data['results']

print("="*100)
print("  ANALYSIS: Did Calibration Fix Fusion F1 Issue?")
print("="*100)
print("\nISSUE: Fusion F1 was lower than individual models it's ensembling\n")

# Best condition for analysis: Multilabel stratified + Aug (most reliable)
conditions_to_check = [
    ("Multilabel stratified + Aug", "DictaBERT"),
    ("Multilabel stratified + Aug", "BEREL 3.0"),
]

for cond, model in conditions_to_check:
    print(f"\n{model}: {cond}")
    print("-" * 100)
    
    filtered = [r for r in results 
                if r['model_name'] == model 
                and r['condition_short'] == cond
                and r['data_source'] == 'exp07+aug']
    
    results_by_exp = {}
    for r in filtered:
        results_by_exp[r['experiment_id']] = r['f1']
    
    if all(k in results_by_exp for k in ['01', '03', '06', '06_fusion_normalized']):
        f1_regular = results_by_exp.get('01', 0)
        f1_cascade = results_by_exp.get('03', 0)
        f1_fusion_orig = results_by_exp.get('06', 0)
        f1_fusion_cal = results_by_exp.get('06_fusion_normalized', 0)
        
        model_best = max(f1_regular, f1_cascade)
        
        print(f"  Regular NER (Exp01):           {f1_regular:.4f}")
        print(f"  Cascaded Pipeline (Exp03):     {f1_cascade:.4f}")
        print(f"  Best individual model:         {model_best:.4f}")
        print()
        print(f"  Fusion Original (Exp06):      {f1_fusion_orig:.4f}  (delta from best: {f1_fusion_orig - model_best:+.4f})")
        print(f"  Fusion Calibrated (Exp06_Cal): {f1_fusion_cal:.4f}  (delta from best: {f1_fusion_cal - model_best:+.4f})")
        print()
        
        # Check if issue is fixed
        if f1_fusion_orig < model_best:
            print(f"  ✗ ORIGINAL FUSION: LOWER than best model (issue EXISTS)")
            if f1_fusion_cal > model_best:
                print(f"  ✓ CALIBRATED FUSION: NOW BEATS best model (ISSUE FIXED!)")
            elif f1_fusion_cal > f1_fusion_orig:
                print(f"  ~ CALIBRATED FUSION: Improved (+{f1_fusion_cal - f1_fusion_orig:+.4f}) but still below models")
            else:
                print(f"  ✗ CALIBRATED FUSION: Still lower, no improvement")
        else:
            print(f"  ✓ ORIGINAL FUSION: Already beats best model")
            if f1_fusion_cal > f1_fusion_orig:
                print(f"  ✓ CALIBRATED FUSION: Even better (+{f1_fusion_cal - f1_fusion_orig:+.4f})")
            elif f1_fusion_cal < f1_fusion_orig:
                print(f"  ~ CALIBRATED FUSION: Slightly worse ({f1_fusion_cal - f1_fusion_orig:+.4f})")
            else:
                print(f"  ≈ CALIBRATED FUSION: Same performance")

print("\n" + "="*100)
print("SUMMARY")
print("="*100)

# Count improvements
exp06_results = [r for r in results if r['experiment_id'] == '06']
exp06_cal_results = [r for r in results if r['experiment_id'] == '06_fusion_normalized']

improvements = 0
regressions = 0

for r06, r06_cal in zip(sorted(exp06_results, key=lambda x: (x['model_name'], x['condition_short'])),
                        sorted(exp06_cal_results, key=lambda x: (x['model_name'], x['condition_short']))):
    if r06_cal['f1'] > r06['f1']:
        improvements += 1
    elif r06_cal['f1'] < r06['f1']:
        regressions += 1

print(f"\nOverall calibration impact (across all conditions):")
print(f"  ✓ Improvements: {improvements}/16 conditions ({improvements/16*100:.0f}%)")
print(f"  ✗ Regressions:  {regressions}/16 conditions ({regressions/16*100:.0f}%)")
print(f"  ≈ No change:    {16-improvements-regressions}/16 conditions")

if improvements > regressions:
    print(f"\n✓ VERDICT: Calibration helps MORE than it hurts")
else:
    print(f"\n✗ VERDICT: Calibration hurts MORE than it helps")
