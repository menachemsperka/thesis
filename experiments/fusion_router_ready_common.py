"""
fusion_router_ready_common.py — Shared sklearn disagreement routers for ready fusion.

Used by Exp06 (Exp01 + Exp04) and Exp10 (CRF) ready fusion experiments.
See ``theisis overview.md`` §12 (linear/kernel SVM) and §12B (NB, LR, RF, MLP).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd

from fusion_ready_sources import (
    FUSION_ROUTER_CATEGORICAL_FEATURES,
    FUSION_ROUTER_NUMERIC_FEATURES,
    ROUTER_GAUSSIAN_NB_PARAMS,
    ROUTER_LINEAR_SVC_PARAMS,
    ROUTER_LOGISTIC_REGRESSION_PARAMS,
    ROUTER_MLP_PARAMS,
    ROUTER_RANDOM_FOREST_PARAMS,
    ROUTER_SVC_RBF_PARAMS,
    run_ready_fusion,
)

try:
    import joblib
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.svm import LinearSVC, SVC
except ImportError:
    joblib = None  # type: ignore[misc,assignment]
    Pipeline = None  # type: ignore[misc,assignment]

Track = Literal["exp06", "exp10"]

_NUMERIC_FEATURES = list(FUSION_ROUTER_NUMERIC_FEATURES)
_CATEGORICAL_FEATURES = list(FUSION_ROUTER_CATEGORICAL_FEATURES)


@dataclass(frozen=True)
class ReadyRouterConfig:
    config_key: str
    classifier_label: str
    experiment_id: str
    experiment_name: str
    result_basename: str
    artifact_prefix: str
    source_tag_regular: str
    source_tag_cascade: str
    source_tag_fallback_regular: str
    source_tag_fallback_cascade: str


def _exp06_configs() -> dict[str, ReadyRouterConfig]:
    return {
        "linear_svm": ReadyRouterConfig(
            config_key="linear_svm",
            classifier_label="LinearSVC",
            experiment_id="exp06_svm_ready",
            experiment_name="Linear SVM Router Fusion (Ready)",
            result_basename="fusion_svm_ready",
            artifact_prefix="exp06_svm_linear_router",
            source_tag_regular="svm_regular",
            source_tag_cascade="svm_cascade",
            source_tag_fallback_regular="fallback_regular",
            source_tag_fallback_cascade="fallback_cascade",
        ),
        "kernel_svm": ReadyRouterConfig(
            config_key="kernel_svm",
            classifier_label="SVC_rbf",
            experiment_id="exp06_svm_kernel_ready",
            experiment_name="Kernel SVM (RBF) Router Fusion (Ready)",
            result_basename="fusion_svm_kernel_ready",
            artifact_prefix="exp06_svm_kernel_router",
            source_tag_regular="svm_kernel_regular",
            source_tag_cascade="svm_kernel_cascade",
            source_tag_fallback_regular="fallback_regular",
            source_tag_fallback_cascade="fallback_cascade",
        ),
        "nb": ReadyRouterConfig(
            config_key="nb",
            classifier_label="GaussianNB",
            experiment_id="exp06_nb_ready",
            experiment_name="Naive Bayes Router Fusion (Ready)",
            result_basename="fusion_nb_ready",
            artifact_prefix="exp06_nb_router",
            source_tag_regular="nb_regular",
            source_tag_cascade="nb_cascade",
            source_tag_fallback_regular="fallback_regular",
            source_tag_fallback_cascade="fallback_cascade",
        ),
        "lr": ReadyRouterConfig(
            config_key="lr",
            classifier_label="LogisticRegression",
            experiment_id="exp06_lr_ready",
            experiment_name="Logistic Regression Router Fusion (Ready)",
            result_basename="fusion_lr_ready",
            artifact_prefix="exp06_lr_router",
            source_tag_regular="lr_regular",
            source_tag_cascade="lr_cascade",
            source_tag_fallback_regular="fallback_regular",
            source_tag_fallback_cascade="fallback_cascade",
        ),
        "rf": ReadyRouterConfig(
            config_key="rf",
            classifier_label="RandomForestClassifier",
            experiment_id="exp06_rf_ready",
            experiment_name="Random Forest Router Fusion (Ready)",
            result_basename="fusion_rf_ready",
            artifact_prefix="exp06_rf_router",
            source_tag_regular="rf_regular",
            source_tag_cascade="rf_cascade",
            source_tag_fallback_regular="fallback_regular",
            source_tag_fallback_cascade="fallback_cascade",
        ),
        "mlp": ReadyRouterConfig(
            config_key="mlp",
            classifier_label="MLPClassifier",
            experiment_id="exp06_mlp_ready",
            experiment_name="MLP Router Fusion (Ready)",
            result_basename="fusion_mlp_ready",
            artifact_prefix="exp06_mlp_router",
            source_tag_regular="mlp_regular",
            source_tag_cascade="mlp_cascade",
            source_tag_fallback_regular="fallback_regular",
            source_tag_fallback_cascade="fallback_cascade",
        ),
    }


def _exp10_configs() -> dict[str, ReadyRouterConfig]:
    base = _exp06_configs()
    mapping = {
        "linear_svm": ("exp10_svm_ready", "Linear SVM Router Fusion CRF (Ready)", "fusion_svm_crf_ready", "exp10_svm_linear_router"),
        "kernel_svm": ("exp10_svm_kernel_ready", "Kernel SVM (RBF) Router Fusion CRF (Ready)", "fusion_svm_kernel_crf_ready", "exp10_svm_kernel_router"),
        "nb": ("exp10_nb_ready", "Naive Bayes Router Fusion CRF (Ready)", "fusion_nb_crf_ready", "exp10_nb_router"),
        "lr": ("exp10_lr_ready", "Logistic Regression Router Fusion CRF (Ready)", "fusion_lr_crf_ready", "exp10_lr_router"),
        "rf": ("exp10_rf_ready", "Random Forest Router Fusion CRF (Ready)", "fusion_rf_crf_ready", "exp10_rf_router"),
        "mlp": ("exp10_mlp_ready", "MLP Router Fusion CRF (Ready)", "fusion_mlp_crf_ready", "exp10_mlp_router"),
    }
    out: dict[str, ReadyRouterConfig] = {}
    for key, cfg in base.items():
        exp_id, exp_name, basename, artifact = mapping[key]
        out[key] = ReadyRouterConfig(
            config_key=cfg.config_key,
            classifier_label=cfg.classifier_label,
            experiment_id=exp_id,
            experiment_name=exp_name,
            result_basename=basename,
            artifact_prefix=artifact,
            source_tag_regular=cfg.source_tag_regular.replace("regular", "regular_crf"),
            source_tag_cascade=cfg.source_tag_cascade.replace("cascade", "cascade_crf"),
            source_tag_fallback_regular="fallback_regular_crf",
            source_tag_fallback_cascade="fallback_cascade_crf",
        )
    return out


def _sanitize_for_path(value, fallback="unknown") -> str:
    text = str(value or "").strip().replace("\\", "/")
    if not text:
        text = fallback
    text = text.rstrip("/").split("/")[-1] or fallback
    text = re.sub(r"[^A-Za-z0-9._=-]+", "_", text).strip("_")
    return (text or fallback)[:160]


def _build_classifier(config_key: str):
    if config_key == "linear_svm":
        return LinearSVC(**ROUTER_LINEAR_SVC_PARAMS)
    if config_key == "kernel_svm":
        return SVC(**ROUTER_SVC_RBF_PARAMS)
    if config_key == "nb":
        return GaussianNB(**ROUTER_GAUSSIAN_NB_PARAMS)
    if config_key == "lr":
        return LogisticRegression(**ROUTER_LOGISTIC_REGRESSION_PARAMS)
    if config_key == "rf":
        return RandomForestClassifier(**ROUTER_RANDOM_FOREST_PARAMS)
    if config_key == "mlp":
        return MLPClassifier(**ROUTER_MLP_PARAMS)
    raise ValueError(f"Unknown router config_key: {config_key}")


def _build_preprocess_pipeline(config_key: str) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), _NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), _CATEGORICAL_FEATURES),
        ],
    )


def _train_router(merged: pd.DataFrame, config: ReadyRouterConfig):
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

    model = Pipeline(
        steps=[
            ("pre", _build_preprocess_pipeline(config.config_key)),
            ("clf", _build_classifier(config.config_key)),
        ]
    )

    x_train = usable[_NUMERIC_FEATURES + _CATEGORICAL_FEATURES]
    y_train = usable["target_source"].astype(str)
    model.fit(x_train, y_train)

    return model, {
        "trained": True,
        "router": config.classifier_label,
        "config_key": config.config_key,
        "training_samples": len(usable),
        "class_distribution": y_train.value_counts().to_dict(),
    }


def _save_router_artifact(router, info: dict, config: ReadyRouterConfig) -> str:
    save_models_flag = (os.environ.get("THESIS_SAVE_TRAINED_MODELS") or "").strip() == "1"
    if not save_models_flag:
        return ""

    project_root = Path(__file__).resolve().parents[1]
    save_base = project_root / "outputs" / "trained_models"
    save_base.mkdir(parents=True, exist_ok=True)

    model_short = _sanitize_for_path(os.environ.get("THESIS_MODEL_NAME", "model"), "model")
    condition_key = _sanitize_for_path(os.environ.get("THESIS_CURRENT_CONDITION_KEY", "default"), "default")
    seed = _sanitize_for_path(os.environ.get("THESIS_SPLIT_SEED", "42"), "seed")
    save_path = save_base / f"{config.artifact_prefix}_{model_short}_{condition_key}_seed{seed}"
    save_path.mkdir(parents=True, exist_ok=True)

    metadata = {
        "artifact_type": f"{config.classifier_label} disagreement router (ready fusion)",
        "experiment_id": config.experiment_id,
        "model_name_env": os.environ.get("THESIS_MODEL_NAME", ""),
        "condition_key": os.environ.get("THESIS_CURRENT_CONDITION_KEY", "default"),
        "seed": os.environ.get("THESIS_SPLIT_SEED", "42"),
        "numeric_features": _NUMERIC_FEATURES,
        "categorical_features": _CATEGORICAL_FEATURES,
        "router_info": info,
        "router_file": "router.joblib" if router is not None and joblib is not None else "",
    }

    if router is not None and joblib is not None:
        joblib.dump(router, save_path / "router.joblib")

    (save_path / "router_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (save_path / "README.md").write_text(
        f"# {config.experiment_name}\n\n"
        f"Scikit-learn `{config.classifier_label}` router for `{config.experiment_id}`.\n",
        encoding="utf-8",
    )
    print(f"[Router Saved] {save_path}")
    return str(save_path)


def make_router_fusion_fn(config: ReadyRouterConfig) -> Callable[[pd.DataFrame], pd.DataFrame]:
    def _router_fusion(merged: pd.DataFrame) -> pd.DataFrame:
        router, info = _train_router(merged, config)
        router_artifact_path = _save_router_artifact(router, info, config)

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
                    source[idx] = config.source_tag_regular
                    confidence[idx] = reg_prob[idx]
                else:
                    fused[idx] = cas_label[idx]
                    source[idx] = config.source_tag_cascade
                    confidence[idx] = cas_prob[idx]
        else:
            for idx in disag_idx:
                if reg_prob[idx] >= cas_prob[idx]:
                    fused[idx] = reg_label[idx]
                    source[idx] = config.source_tag_fallback_regular
                    confidence[idx] = reg_prob[idx]
                else:
                    fused[idx] = cas_label[idx]
                    source[idx] = config.source_tag_fallback_cascade
                    confidence[idx] = cas_prob[idx]

        merged["fused_pred_label"] = fused
        merged["selected_source"] = source
        merged["selected_confidence"] = confidence
        merged["router_info"] = str(info)
        merged["router_artifact"] = router_artifact_path
        merged["router_classifier"] = config.classifier_label
        return merged

    return _router_fusion


def exp06_run(config_key: str) -> dict:
    config = _exp06_configs()[config_key]
    return run_ready_fusion(
        strategy_fn=make_router_fusion_fn(config),
        experiment_id=config.experiment_id,
        experiment_name=config.experiment_name,
        description=(
            f"No-retraining {config.classifier_label} disagreement router from Exp01 + Exp04 "
            "ready outputs. Same feature schema as §12; fallback to confidence fusion (§11.3)."
        ),
        result_basename=config.result_basename,
        extra_info={"router": config.classifier_label, "router_config_key": config_key},
    )


def exp10_run(config_key: str) -> dict:
    from fusion_crf_ready_sources import run_ready_fusion_crf

    config = _exp10_configs()[config_key]
    return run_ready_fusion_crf(
        strategy_fn=make_router_fusion_fn(config),
        experiment_id=config.experiment_id,
        experiment_name=config.experiment_name,
        description=(
            f"No-retraining {config.classifier_label} disagreement router from Exp10 regular "
            "BERT-CRF and cascaded CRF ready outputs."
        ),
        result_basename=config.result_basename,
        extra_info={
            "router": config.classifier_label,
            "router_config_key": config_key,
            "sources": "exp10_regular+exp10_cascade",
        },
    )


def exp06_run_factory(config_key: str) -> Callable[[], dict]:
    def run() -> dict:
        return exp06_run(config_key)

    return run


def exp10_run_factory(config_key: str) -> Callable[[], dict]:
    def run() -> dict:
        return exp10_run(config_key)

    return run


# Cross-comparison runner keys (without ``exp`` prefix)
READY_ML_ROUTER_EXP_IDS: frozenset[str] = frozenset(
    {
        "06_svm_ready",
        "06_svm_kernel_ready",
        "06_nb_ready",
        "06_lr_ready",
        "06_rf_ready",
        "06_mlp_ready",
        "10_svm_ready",
        "10_svm_kernel_ready",
        "10_nb_ready",
        "10_lr_ready",
        "10_rf_ready",
        "10_mlp_ready",
    }
)
