"""
experiment_10_fusion_svm_ready.py — SVM router fusion on Exp10 CRF outputs
==========================================================================

Parallel to ``experiment_06_fusion_svm_ready.py``, but sources are:

* ``outputs/exp10_regular/`` (BERT-CRF token predictions)
* ``outputs/exp10_cascade/`` (cascaded CRF ``detailed_results``)

The router is a **LinearSVC** trained on **disagreement tokens** where exactly one source is
correct (same target definition as Section 12 of ``theisis overview.md``).

Limitation (important for grading)
----------------------------------
The ready SVM variant **trains and applies the router on the same evaluation split** used to
report F1. Treat it as an exploratory upper bound; a held-out router split would be stricter.

See ``experiments/experiment_10_README.md`` § lab exercises for comparison with confidence fusion.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from fusion_crf_ready_sources import run_ready_fusion_crf

try:
    import joblib
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.svm import LinearSVC
except ImportError:
    joblib = None  # type: ignore[misc,assignment]
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


def _sanitize_for_path(value, fallback="unknown") -> str:
    text = str(value or "").strip().replace("\\", "/")
    if not text:
        text = fallback
    text = text.rstrip("/").split("/")[-1] or fallback
    text = re.sub(r"[^A-Za-z0-9._=-]+", "_", text).strip("_")
    return (text or fallback)[:160]


def _save_router_artifact(router, info: dict) -> str:
    save_models_flag = (os.environ.get("THESIS_SAVE_TRAINED_MODELS") or "").strip() == "1"
    if not save_models_flag:
        return ""

    project_root = Path(__file__).resolve().parents[1]
    save_base = project_root / "outputs" / "trained_models"
    save_base.mkdir(parents=True, exist_ok=True)

    model_short = _sanitize_for_path(os.environ.get("THESIS_MODEL_NAME", "model"), "model")
    condition_key = _sanitize_for_path(os.environ.get("THESIS_CURRENT_CONDITION_KEY", "default"), "default")
    seed = _sanitize_for_path(os.environ.get("THESIS_SPLIT_SEED", "42"), "42")
    save_path = save_base / f"exp10_svm_router_{model_short}_{condition_key}_seed{seed}"
    save_path.mkdir(parents=True, exist_ok=True)

    metadata = {
        "artifact_type": "SVM disagreement router for Exp10 CRF ready fusion",
        "experiment_id": "exp10_svm_ready",
        "model_name_env": os.environ.get("THESIS_MODEL_NAME", ""),
        "condition_key": os.environ.get("THESIS_CURRENT_CONDITION_KEY", "default"),
        "seed": os.environ.get("THESIS_SPLIT_SEED", "42"),
        "numeric_features": _NUMERIC_FEATURES,
        "categorical_features": _CATEGORICAL_FEATURES,
        "router_info": info,
        "router_file": "router.joblib" if router is not None and joblib is not None else "",
        "notes": (
            "Scikit-learn router choosing between Exp10 regular BERT-CRF and "
            "Exp10 cascaded CRF predictions on disagreement tokens."
        ),
    }

    if router is not None and joblib is not None:
        joblib.dump(router, save_path / "router.joblib")

    (save_path / "router_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (save_path / "README.md").write_text(
        "# Exp10 SVM Router Fusion Artifact\n\n"
        "LinearSVC disagreement router for `exp10_svm_ready` (CRF regular + CRF cascade).\n",
        encoding="utf-8",
    )
    print(f"[Router Saved] {save_path}")
    return str(save_path)


def _train_router(merged: pd.DataFrame):
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
        return None

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
    router, info = _train_router(merged)
    router_artifact_path = _save_router_artifact(router, info)

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
                source[idx] = "svm_regular_crf"
                confidence[idx] = reg_prob[idx]
            else:
                fused[idx] = cas_label[idx]
                source[idx] = "svm_cascade_crf"
                confidence[idx] = cas_prob[idx]
    else:
        for idx in disag_idx:
            if reg_prob[idx] >= cas_prob[idx]:
                fused[idx] = reg_label[idx]
                source[idx] = "fallback_regular_crf"
                confidence[idx] = reg_prob[idx]
            else:
                fused[idx] = cas_label[idx]
                source[idx] = "fallback_cascade_crf"
                confidence[idx] = cas_prob[idx]

    merged["fused_pred_label"] = fused
    merged["selected_source"] = source
    merged["selected_confidence"] = confidence
    merged["svm_router_info"] = str(info)
    merged["svm_router_artifact"] = router_artifact_path
    return merged


def run() -> dict:
    return run_ready_fusion_crf(
        strategy_fn=_svm_fusion,
        experiment_id="exp10_svm_ready",
        experiment_name="SVM Router Fusion CRF (Ready)",
        description=(
            "No-retraining SVM disagreement router from Exp10 regular BERT-CRF and "
            "Exp10 cascaded CRF ready outputs. Trains LinearSVC on disagreement tokens."
        ),
        result_basename="fusion_svm_crf_ready",
        extra_info={"router": "LinearSVC", "sources": "exp10_regular+exp10_cascade"},
    )


if __name__ == "__main__":
    payload = run()
    f1_str = f"{payload['f1']:.4f}" if payload.get("f1") is not None else "N/A"
    print(f"[exp10_svm_ready] F1={f1_str}")
