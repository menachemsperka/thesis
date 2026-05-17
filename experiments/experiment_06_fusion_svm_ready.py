"""
experiment_06_fusion_svm_ready.py — SVM Router Fusion (Ready)

Trains an SVM disagreement router on Exp01 + Exp04 ready outputs without
retraining any NER models.

The router is trained on disagreement tokens where one source is clearly correct,
then applied to ALL disagreements.  Agreement tokens pass through unchanged.

Note: SVM is both trained and evaluated on the same data set.  For strict
train/test separation use the full training variant (experiment_06_fusion_svm.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fusion_ready_sources import run_ready_fusion

try:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.svm import LinearSVC
except ImportError:
    Pipeline = None  # type: ignore[misc,assignment]


_NUMERIC_FEATURES = [
    "regular_prob",
    "cascade_prob",
    "regular_margin",
    "cascade_margin",
    "prob_diff",
    "abs_prob_diff",
    "max_prob",
]

_CATEGORICAL_FEATURES = [
    "regular_bio",
    "regular_etype",
    "cascade_bio",
    "cascade_etype",
]


def _train_router(merged: pd.DataFrame):
    """Train a LinearSVC router on disagreements where one source is unambiguously correct."""
    disag = merged[merged["disagree"]].copy()
    disag["regular_correct"] = disag["regular_pred_label"].astype(str) == disag["true_label"].astype(str)
    disag["cascade_correct"] = disag["cascade_pred_label"].astype(str) == disag["true_label"].astype(str)

    def _target(row):
        r = bool(row["regular_correct"])
        c = bool(row["cascade_correct"])
        if r and not c:
            return "regular"
        if c and not r:
            return "cascade"
        return None  # ambiguous

    disag["target_source"] = disag.apply(_target, axis=1)
    usable = disag[disag["target_source"].notna()].copy()

    if usable.empty or len(usable["target_source"].unique()) < 2:
        return None, {"trained": False, "reason": "insufficient_training_data"}

    if Pipeline is None:
        return None, {"trained": False, "reason": "sklearn_not_installed"}

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), _NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), _CATEGORICAL_FEATURES),
        ],
    )

    model = Pipeline(
        steps=[
            ("pre", preprocess),
            ("clf", LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=5000)),
        ]
    )

    x_train = usable[_NUMERIC_FEATURES + _CATEGORICAL_FEATURES]
    y_train = usable["target_source"].astype(str)
    model.fit(x_train, y_train)

    return model, {
        "trained": True,
        "router": "LinearSVC",
        "training_samples": len(usable),
        "class_distribution": y_train.value_counts().to_dict(),
    }


def _svm_fusion(merged):
    """Train SVM router, then apply it on all disagreements."""
    router, info = _train_router(merged)

    reg_label = merged["regular_pred_label"].values
    cas_label = merged["cascade_pred_label"].values
    reg_prob = merged["regular_prob"].values
    cas_prob = merged["cascade_prob"].values
    agree = ~merged["disagree"].values

    fused = reg_label.copy()
    source = np.full(len(merged), "agree", dtype=object)
    confidence = np.maximum(reg_prob, cas_prob)

    disag_mask = ~agree
    disag_idx = np.where(disag_mask)[0]

    if router is not None and len(disag_idx) > 0:
        x_disag = merged.iloc[disag_idx][_NUMERIC_FEATURES + _CATEGORICAL_FEATURES]
        predictions = router.predict(x_disag)

        for i, idx in enumerate(disag_idx):
            choice = str(predictions[i])
            if choice == "regular":
                fused[idx] = reg_label[idx]
                source[idx] = "svm_regular"
                confidence[idx] = reg_prob[idx]
            else:
                fused[idx] = cas_label[idx]
                source[idx] = "svm_cascade"
                confidence[idx] = cas_prob[idx]
    else:
        # Fallback: confidence comparison
        for idx in disag_idx:
            if reg_prob[idx] >= cas_prob[idx]:
                fused[idx] = reg_label[idx]
                source[idx] = "fallback_regular"
                confidence[idx] = reg_prob[idx]
            else:
                fused[idx] = cas_label[idx]
                source[idx] = "fallback_cascade"
                confidence[idx] = cas_prob[idx]

    merged["fused_pred_label"] = fused
    merged["selected_source"] = source
    merged["selected_confidence"] = confidence
    merged["svm_router_info"] = str(info)
    return merged


def run() -> dict:
    return run_ready_fusion(
        strategy_fn=_svm_fusion,
        experiment_id="exp06_svm_ready",
        experiment_name="SVM Router Fusion (Ready)",
        description=(
            "No-retraining SVM disagreement router from Exp01 + Exp04 ready outputs. "
            "Trains LinearSVC on disagreement tokens, routes all disagreements."
        ),
        result_basename="fusion_svm_ready",
        extra_info={"router": "LinearSVC"},
    )


if __name__ == "__main__":
    payload = run()
    f1_str = f"{payload['f1']:.4f}" if payload.get("f1") is not None else "N/A"
    print(f"[exp06_svm_ready] F1={f1_str}")
