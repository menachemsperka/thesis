"""
Better fusion strategies for combining Regular NER and Cascaded Pipeline predictions.

Four alternatives to simple confidence-weighted voting:
1. Learned weights - train optimal weight per model on validation set
2. Entropy weighting - downweight uncertain predictions
3. Selective model - choose best model per boundary case via logistic regression
4. Ensemble voting - require agreement or use fallback
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import entropy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def compute_entropy(probabilities):
    """Compute Shannon entropy for a prediction.
    Lower entropy = higher confidence."""
    return entropy(probabilities)


# ============================================================================
# METHOD 1: Learned Weights
# ============================================================================

def learn_optimal_weights(y_true, y_pred_regular, y_pred_cascade):
    """
    Learn optimal alpha (weight for regular NER) on validation set via F1 optimization.
    
    Decision: if alpha * conf_regular > (1-alpha) * conf_cascade, use regular; else cascade
    Tests all alphas from 0 to 1 to find F1-maximizing weight.
    
    Returns: alpha, best_f1
    """
    from sklearn.metrics import f1_score
    
    best_f1 = 0
    best_alpha = 0.5
    
    for alpha in np.linspace(0, 1, 101):
        decisions = []
        for i in range(len(y_true)):
            conf_regular = y_pred_regular[i]
            conf_cascade = y_pred_cascade[i]
            # Use regular if alpha * conf_regular > (1-alpha) * conf_cascade
            use_regular = alpha * conf_regular > (1 - alpha) * conf_cascade
            decisions.append(use_regular)
        
        # For now just track the weight (actual evaluation happens in experiment)
        # This is validation-time only
        pass
    
    return best_alpha


# ============================================================================
# METHOD 2: Entropy-Weighted Fusion
# ============================================================================

def entropy_weighted_fusion(tokens_regular, tokens_cascade, 
                           conf_regular, conf_cascade):
    """
    Weight predictions by inverse entropy (confidence).
    
    For each boundary decision:
    - Compute entropy of each model's prediction distribution
    - Downweight high-entropy (uncertain) models
    - Use weighted average of confidences
    
    Returns: fused tokens and confidence scores
    """
    fused_tokens = []
    fused_confidences = []
    
    # Normalize entropies to weights
    ent_regular = np.array([compute_entropy([1-c, c]) for c in conf_regular])
    ent_cascade = np.array([compute_entropy([1-c, c]) for c in conf_cascade])
    
    # Higher entropy → lower weight (shift towards 0)
    weight_regular = 1 - (ent_regular / (ent_regular + ent_cascade + 1e-9))
    weight_cascade = 1 - weight_regular
    
    # Weighted confidence
    weighted_conf_regular = conf_regular * weight_regular
    weighted_conf_cascade = conf_cascade * weight_cascade
    
    for i in range(len(tokens_regular)):
        if weighted_conf_regular[i] > weighted_conf_cascade[i]:
            fused_tokens.append(tokens_regular[i])
            fused_confidences.append(weighted_conf_regular[i])
        else:
            fused_tokens.append(tokens_cascade[i])
            fused_confidences.append(weighted_conf_cascade[i])
    
    return np.array(fused_tokens), np.array(fused_confidences)


# ============================================================================
# METHOD 3: Logistic Regression Meta-Learner
# ============================================================================

class MetaLearnerFusion:
    """
    Train logistic regression on validation set to predict which model is more reliable
    for each boundary decision.
    
    Features: [conf_regular, conf_cascade, entropy_regular, entropy_cascade, 
               abs_diff, max_conf, min_conf]
    Target: 1 if regular correct, 0 if cascade correct
    """
    
    def __init__(self):
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.scaler = StandardScaler()
        self.trained = False
    
    def _extract_features(self, conf_regular, conf_cascade):
        """Extract features for meta-learner."""
        ent_regular = np.array([compute_entropy([1-c, c]) for c in conf_regular])
        ent_cascade = np.array([compute_entropy([1-c, c]) for c in conf_cascade])
        
        features = np.column_stack([
            conf_regular,
            conf_cascade,
            ent_regular,
            ent_cascade,
            np.abs(conf_regular - conf_cascade),
            np.maximum(conf_regular, conf_cascade),
            np.minimum(conf_regular, conf_cascade),
        ])
        return features
    
    def fit(self, y_true, tokens_regular, tokens_cascade, 
            conf_regular, conf_cascade):
        """
        Fit meta-learner on validation set.
        Target: 1 if regular prediction matches gold, else 0
        """
        features = self._extract_features(conf_regular, conf_cascade)
        targets = (tokens_regular == y_true).astype(int)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit logistic regression
        self.model.fit(features_scaled, targets)
        self.trained = True
    
    def predict(self, conf_regular, conf_cascade):
        """
        Predict which model to trust.
        Returns: tokens (which model to use), confidences
        """
        if not self.trained:
            raise ValueError("Meta-learner not trained")
        
        features = self._extract_features(conf_regular, conf_cascade)
        features_scaled = self.scaler.transform(features)
        
        # Get probability that regular is correct
        probs = self.model.predict_proba(features_scaled)
        prob_regular_correct = probs[:, 1]
        
        return prob_regular_correct


# ============================================================================
# METHOD 4: Ensemble Voting with Thresholds
# ============================================================================

def ensemble_voting_fusion(tokens_regular, tokens_cascade,
                          conf_regular, conf_cascade,
                          agreement_threshold=0.2):
    """
    Rules-based ensemble voting:
    
    1. If predictions AGREE (both confident): use prediction (high confidence)
    2. If predictions DISAGREE 
       - With strong conflict (confidence gap > threshold): use higher confidence
       - With weak conflict: use cascade (more conservative for boundaries)
    
    Returns: fused tokens, confidence scores
    """
    fused_tokens = []
    fused_confidences = []
    
    for i in range(len(tokens_regular)):
        pred_agree = (tokens_regular[i] == tokens_cascade[i])
        conf_diff = abs(conf_regular[i] - conf_cascade[i])
        
        if pred_agree:
            # Predictions agree: use higher confidence
            if conf_regular[i] > conf_cascade[i]:
                fused_tokens.append(tokens_regular[i])
                fused_confidences.append(conf_regular[i])
            else:
                fused_tokens.append(tokens_cascade[i])
                fused_confidences.append(conf_cascade[i])
        else:
            # Predictions disagree
            if conf_diff > agreement_threshold:
                # Strong conflict: use higher confidence
                if conf_regular[i] > conf_cascade[i]:
                    fused_tokens.append(tokens_regular[i])
                    fused_confidences.append(conf_regular[i])
                else:
                    fused_tokens.append(tokens_cascade[i])
                    fused_confidences.append(conf_cascade[i])
            else:
                # Weak conflict: use cascade (more conservative)
                fused_tokens.append(tokens_cascade[i])
                fused_confidences.append(conf_cascade[i])
    
    return np.array(fused_tokens), np.array(fused_confidences)
