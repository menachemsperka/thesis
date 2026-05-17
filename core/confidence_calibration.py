"""
confidence_calibration.py — Learn and apply confidence calibration via temperature scaling.

This module learns optimal temperature parameters for each model on a validation set
and applies temperature scaling at test time to normalize confidence scores to a
common scale.

Temperature scaling: confidence_scaled = sigmoid(logit / T)
where logit = log(p / (1-p)) is the inverse sigmoid of the original probability.

By learning T per model, we calibrate confidence to match the empirical frequency
of correct predictions, making confidences from different models comparable.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit


def _ece(y_true, y_pred_conf, n_bins=10):
    """Expected Calibration Error: mean absolute difference between confidence and accuracy."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ece_val = 0.0
    bin_count = 0
    for i in range(n_bins):
        mask = (y_pred_conf >= bins[i]) & (y_pred_conf < bins[i + 1])
        if mask.sum() > 0:
            acc = y_true[mask].mean()
            conf = y_pred_conf[mask].mean()
            ece_val += np.abs(acc - conf) * mask.sum()
            bin_count += mask.sum()
    return ece_val / max(1, bin_count) if bin_count > 0 else 0.0


def learn_temperature(y_true, y_pred_conf, initial_temp=1.0, solver="bfgs"):
    """
    Learn optimal temperature T via maximum likelihood on validation set.

    Parameters
    ----------
    y_true : array-like, shape (n,)
        Binary correctness labels (1 = correct, 0 = incorrect).
    y_pred_conf : array-like, shape (n,)
        Original confidence scores from model (should be 0-1 probabilities).
    initial_temp : float
        Initial temperature guess (default 1.0 = no scaling).
    solver : str
        scipy optimizer to use (default "bfgs").

    Returns
    -------
    temperature : float
        Learned temperature parameter T > 0.
    ece_before : float
        Expected Calibration Error before temperature scaling.
    ece_after : float
        Expected Calibration Error after temperature scaling.
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred_conf = np.asarray(y_pred_conf, dtype=np.float32)
    y_pred_conf = np.clip(y_pred_conf, 1e-7, 1 - 1e-7)

    # ECE before scaling
    ece_before = _ece(y_true, y_pred_conf)

    # Objective: negative log-likelihood of calibrated confidences
    def nll(temp):
        if temp <= 0:
            return 1e10
        # Apply temperature scaling: convert confidence to logit, scale, convert back
        logits = logit(y_pred_conf)
        calibrated_conf = expit(logits / temp)
        # Avoid log(0)
        calibrated_conf = np.clip(calibrated_conf, 1e-7, 1 - 1e-7)
        # Negative log-likelihood: -sum(y_true * log(conf) + (1-y_true) * log(1-conf))
        nll_val = -(
            y_true * np.log(calibrated_conf) + (1 - y_true) * np.log(1 - calibrated_conf)
        ).mean()
        return nll_val

    # Optimize temperature
    result = minimize(nll, initial_temp, method=solver, bounds=[(0.01, 10.0)])
    temperature = float(result.x[0])

    # ECE after scaling
    logits = logit(y_pred_conf)
    calibrated_conf = np.clip(expit(logits / temperature), 1e-7, 1 - 1e-7)
    ece_after = _ece(y_true, calibrated_conf)

    return temperature, ece_before, ece_after


def apply_temperature_scaling(confidence, temperature):
    """
    Apply learned temperature scaling to confidence scores.

    Parameters
    ----------
    confidence : float or array
        Original confidence score(s), should be in (0, 1).
    temperature : float
        Temperature parameter > 0.

    Returns
    -------
    calibrated : float or array
        Temperature-scaled confidence(s).
    """
    confidence = np.asarray(confidence, dtype=np.float32)
    confidence = np.clip(confidence, 1e-7, 1 - 1e-7)
    logits = logit(confidence)
    calibrated = expit(logits / temperature)
    return float(calibrated) if np.isscalar(confidence) else calibrated


def calibrate_pair(y_true_regular, y_pred_regular_conf, y_true_cascade, y_pred_cascade_conf):
    """
    Learn calibration temperatures for both models independently.

    Parameters
    ----------
    y_true_regular : array-like
        Correctness labels for regular NER predictions.
    y_pred_regular_conf : array-like
        Confidence scores from regular NER.
    y_true_cascade : array-like
        Correctness labels for cascade predictions.
    y_pred_cascade_conf : array-like
        Confidence scores from cascade pipeline.

    Returns
    -------
    result : dict
        Keys: "regular_temp", "cascade_temp", "regular_ece_before", "regular_ece_after",
        "cascade_ece_before", "cascade_ece_after".
    """
    temp_reg, ece_reg_bef, ece_reg_aft = learn_temperature(y_true_regular, y_pred_regular_conf)
    temp_casc, ece_casc_bef, ece_casc_aft = learn_temperature(y_true_cascade, y_pred_cascade_conf)

    return {
        "regular_temperature": float(temp_reg),
        "cascade_temperature": float(temp_casc),
        "regular_ece_before": float(ece_reg_bef),
        "regular_ece_after": float(ece_reg_aft),
        "cascade_ece_before": float(ece_casc_bef),
        "cascade_ece_after": float(ece_casc_aft),
    }
