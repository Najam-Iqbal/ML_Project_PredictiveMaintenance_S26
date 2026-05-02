"""
router.py — Prediction Router & Business Logic
==============================================
Routes predictions through appropriate model & formats output.
Decouples model loading from prediction logic.
"""

import joblib
import numpy as np
from pathlib import Path

# ── Model paths ──────────────────────────────────────────────────────
_BASE = Path(__file__).parent
MODEL_DIR = _BASE / "Trained_models"

# Load models (lazy loading for faster startup)
_MODELS = {}


def load_model(model_name):
    """Load a model by name (cached after first load)."""
    if model_name not in _MODELS:
        path = MODEL_DIR / f"{model_name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        _MODELS[model_name] = joblib.load(str(path))
    return _MODELS[model_name]


# ── Model mapping with thresholds ───────────────────────────────────
# Format: (priority, detail) -> {failure_model, failure_threshold, failure_type, model, threshold, type, display, failure_model_display, metrics}
MODEL_MAP = {
    ("Minimize missed failures", "Primary cause only"): {
        "failure_model": "binary_decision_tree_feature_engineered_13features_threshold_0p96",
        "failure_threshold": 0.75,
        "failure_type": "binary",
        "model": "binary_decision_tree_feature_engineered_13features_threshold_0p96",
        "threshold": 0.75,
        "type": "binary",
        "display": "Binary DT (13 features, high sensitivity)",
        "failure_model_display": "Binary DT (13 features, high sensitivity)",
        "metrics": {
            "precision_test": 0.4098,
            "recall_test": 0.7353,
            "f1_test": 0.5263,
            "precision_cv_mean": 0.3559,
            "precision_cv_std": 0.027,
            "recall_cv_mean": 0.7415,
            "recall_cv_std": 0.0432,
            "f1_cv_mean": 0.48,
            "f1_cv_std": 0.0266,
            "train_samples": 8000,
            "test_samples": 2000,
        },
    },
    ("Minimize missed failures", "All contributing causes"): {
        "failure_model": "binary_decision_tree_feature_engineered_13features_threshold_0p96",
        "failure_threshold": 0.75,
        "failure_type": "binary",
        "model": "multilabel_decision_tree_multioutput_scaled_original_features",
        "threshold": 0.75,
        "type": "multilabel",
        "display": "Multi-Label DT (concurrent failures)",
        "failure_model_display": "Binary DT (13 features, high sensitivity)",
        "metrics": {
            "precision_test": 0.4098,
            "recall_test": 0.7353,
            "f1_test": 0.5263,
            "precision_cv_mean": 0.3559,
            "precision_cv_std": 0.027,
            "recall_cv_mean": 0.7415,
            "recall_cv_std": 0.0432,
            "f1_cv_mean": 0.48,
            "f1_cv_std": 0.0266,
            "train_samples": 8000,
            "test_samples": 2000,
        },
    },
    ("Minimize unnecessary maintenance", "Primary cause only"): {
        "failure_model": "binary_decision_tree_feature_engineered_13features_threshold_0p96",
        "failure_threshold": 0.96,
        "failure_type": "binary",
        "model": "binary_decision_tree_feature_engineered_13features_threshold_0p96",
        "threshold": 0.96,
        "type": "binary",
        "display": "Binary DT (13 features, high precision)",
        "failure_model_display": "Binary DT (13 features, high precision)",
        "metrics": {
            "precision_test": 0.4098,
            "recall_test": 0.7353,
            "f1_test": 0.5263,
            "precision_cv_mean": 0.3559,
            "precision_cv_std": 0.027,
            "recall_cv_mean": 0.7415,
            "recall_cv_std": 0.0432,
            "f1_cv_mean": 0.48,
            "f1_cv_std": 0.0266,
            "train_samples": 8000,
            "test_samples": 2000,
        },
    },
    ("Minimize unnecessary maintenance", "All contributing causes"): {
        "failure_model": "binary_decision_tree_feature_engineered_13features_threshold_0p96",
        "failure_threshold": 0.96,
        "failure_type": "binary",
        "model": "multilabel_decision_tree_multioutput_scaled_original_features",
        "threshold": 0.96,
        "type": "multilabel",
        "display": "Multi-Label DT (concurrent failures)",
        "failure_model_display": "Binary DT (13 features, high precision)",
        "metrics": {
            "precision_test": 0.6667,
            "recall_test": 0.6176,
            "f1_test": 0.6412,
            "precision_cv_mean": 0.5808,
            "precision_cv_std": 0.0392,
            "recall_cv_mean": 0.6899,
            "recall_cv_std": 0.0387,
            "f1_cv_mean": 0.6299,
            "f1_cv_std": 0.0322,
            "train_samples": 8000,
            "test_samples": 2000,
        },
    },
}

# Failure type mapping
FAILURE_TYPES = {
    0: "No failure",
    1: "Tool Wear Failure (TWF)",
    2: "Heat Dissipation Failure (HDF)",
    3: "Power Loss Failure (PWF)",
    4: "Overstrain Failure (OSF)",
    5: "Random Failure (RNF)",
}


def _binary_gate(model, X_input, threshold):
    """Return binary failure predictions and confidence from a binary model."""
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_input)
        prob_failure = probas[:, 1]
        failure_predicted = prob_failure >= threshold
        confidence = np.where(failure_predicted, prob_failure, 1 - prob_failure)
        return failure_predicted.astype(bool), confidence.astype(float)

    preds = model.predict(X_input)
    failure_predicted = preds == 1
    confidence = np.where(failure_predicted, 1.0, 0.0)
    return failure_predicted.astype(bool), confidence.astype(float)


# ── Prediction function ──────────────────────────────────────────────
def predict(X, business_priority, diagnostic_detail="All contributing causes"):
    """
    Route prediction through appropriate model with custom thresholds.
    Cause labels are only emitted when the binary failure gate predicts failure.
    """

    # Allow callers (e.g., Gradio app) to only choose business priority.
    # If diagnostic_detail is empty/None, default to 'All contributing causes'.
    if diagnostic_detail is None or str(diagnostic_detail).strip() == "":
        diagnostic_detail = "All contributing causes"

    key = (business_priority, diagnostic_detail)
    if key not in MODEL_MAP:
        raise ValueError(f"Unknown configuration: {key}")

    config = MODEL_MAP[key]
    failure_model = load_model(config["failure_model"])
    cause_model = load_model(config["model"])

    failure_threshold = config["failure_threshold"]
    cause_threshold = config["threshold"]
    cause_type = config["type"]
    display_name = config["display"]
    failure_display = config["failure_model_display"]

    failure_input = X
    cause_input = X if cause_type == "binary" else X[:, :8]

    failure_predicted_arr, failure_confidence_arr = _binary_gate(
        failure_model, failure_input, failure_threshold
    )

    cause_probas = cause_model.predict_proba(cause_input) if hasattr(cause_model, "predict_proba") else None
    
    metrics = config.get("metrics", {})

    results = []
    for i in range(len(X)):
        failure_predicted = bool(failure_predicted_arr[i])
        failure_confidence = float(failure_confidence_arr[i])

        if not failure_predicted:
            failure_reason = FAILURE_TYPES[0]
            confidence = failure_confidence
        else:
            if cause_type == "binary":
                # Binary cause models only predict failure/no-failure.
                # Do not assume a specific failure type (was hardcoded to TWF).
                failure_reason = "Failure (unspecified)"
                confidence = failure_confidence
            elif cause_type == "multiclass":
                # Map multiclass prediction to failure label correctly.
                pred = int(cause_model.predict(cause_input[i:i+1])[0])
                if pred == 0:
                    failure_reason = FAILURE_TYPES[0]
                else:
                    failure_reason = FAILURE_TYPES.get(pred, f"Class {pred}")

                if cause_probas is not None:
                    # confidence for the predicted class
                    confidence = float(cause_probas[i][pred])
                else:
                    confidence = failure_confidence
            elif cause_type == "multilabel":
                pred = np.ravel(cause_model.predict(cause_input[i:i+1])[0])
                # Build list of cause labels and per-cause confidences (if available)
                causes = []
                cause_confidences = {}
                for j, flag in enumerate(pred):
                    if flag:
                        label = FAILURE_TYPES.get(j + 1, f"Type {j + 1}")
                        causes.append(label)
                        if cause_probas is not None and hasattr(cause_probas, "__len__"):
                            # cause_probas for multilabel may be a list/array per output
                            try:
                                prob = float(cause_probas[j][i, 1])
                            except Exception:
                                # fallback: attempt to index per-sample probability
                                try:
                                    prob = float(cause_probas[j][i])
                                except Exception:
                                    prob = failure_confidence
                            cause_confidences[label] = prob

                failure_reason = " + ".join(causes) if causes else FAILURE_TYPES[1]
                # overall confidence: max of per-cause confidences if available
                if cause_confidences:
                    confidence = float(max(cause_confidences.values()))
                else:
                    confidence = failure_confidence
            else:
                failure_reason = FAILURE_TYPES[1]
                confidence = failure_confidence

        results.append({
            "failure_predicted": failure_predicted,
            "failure_reason": failure_reason,
            "confidence": confidence,
            "model_name": display_name,
            "failure_model_name": failure_display,
            "failure_threshold": failure_threshold,
            "cause_model_name": display_name,
            "threshold": cause_threshold,
            "metrics": metrics,
        })

    return results
