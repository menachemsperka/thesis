#!/usr/bin/env python3
"""Check if calibration fixed the fusion F1 issue."""

import json
from pathlib import Path

# Load all results
exp01 = json.load(open('outputs/exp01/latest.json'))
exp03 = json.load(open('outputs/exp03/latest.json'))
exp06 = json.load(open('outputs/exp06/latest.json'))
exp06_cal = json.load(open('outputs/exp06_fusion_normalized/latest.json'))

print("="*90)
print("  FUSION F1 vs INDIVIDUAL MODELS: Does Fusion Beat the Best Model?")
print("="*90)
print()

print("Individual Models (DictaBERT, Paper-style split, Augmented):")
print(f"  Exp01 (Regular NER):       F1 = {exp01['f1']:.4f}")
print(f"  Exp03 (Cascaded/AUC-2T):   F1 = {exp03['f1']:.4f}")

best_model_f1 = max(exp01['f1'], exp03['f1'])
best_model_name = "AUC-2T (Cascaded)" if exp03['f1'] > exp01['f1'] else "Regular NER"
print(f"  Best model:                F1 = {best_model_f1:.4f} ({best_model_name})")
print()

print("Fusion Experiments:")
f1_orig = exp06['f1']
f1_cal = exp06_cal['f1']
print(f"  Exp06 (Original Fusion):    F1 = {f1_orig:.4f}")
print(f"  Exp06_Cal (Calibrated):     F1 = {f1_cal:.4f}")
print()

delta_orig = f1_orig - best_model_f1
delta_cal = f1_cal - best_model_f1

print("="*90)
print("VERDICT:")
print("="*90)
print(f"  Original Fusion vs Best Model:  {delta_orig:+.4f}  ({('✓ BEATS' if delta_orig > 0 else '✗ LOSES')})")
print(f"  Calibrated Fusion vs Best Model: {delta_cal:+.4f}  ({('✓ BEATS' if delta_cal > 0 else '✗ LOSES')})")
print()

if delta_orig < -0.001:
    print("  ❌ ORIGINAL ISSUE: Fusion F1 is LOWER than best model")
    if delta_cal > delta_orig:
        improvement = delta_cal - delta_orig
        print(f"  ✓ CALIBRATION IMPROVED: +{improvement:+.4f} F1 gain")
        if delta_cal > -0.001:
            print(f"    → Fusion now BEATS best model!")
        else:
            print(f"    → Fusion still underperforms, but gap reduced")
    else:
        print(f"  ✗ CALIBRATION MADE IT WORSE: {delta_cal - delta_orig:+.4f}")
elif delta_orig > 0.001:
    print("  ✓ Original fusion already BEATS best model")
    if delta_cal > delta_orig:
        print(f"  ✓ Calibration further improved: +{delta_cal - delta_orig:+.4f}")
    elif delta_cal < delta_orig:
        print(f"  ~ Calibration slightly degraded: {delta_cal - delta_orig:+.4f}")
else:
    print("  ≈ Original fusion roughly equal to best model")

print()
print(f"→ SUMMARY: Calibration impact = {delta_cal - delta_orig:+.4f} F1 points")
