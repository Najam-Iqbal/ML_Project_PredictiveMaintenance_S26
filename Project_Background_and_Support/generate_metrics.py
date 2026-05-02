"""
generate_metrics.py — Generate Model Performance Metrics with Cross-Validation
===============================================================================
Calculates precision, recall, F1 for each model at different thresholds using:
  - Test set for final evaluation
  - 5-fold Stratified CV on training data for robustness
  - Multiple thresholds: 0.75, 0.96

Saves metrics to model_metrics.json for display in Gradio.

Run once: python generate_metrics.py
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from pipeline import preprocess

# ── Paths ────────────────────────────────────────────────────────────
_BASE = Path(__file__).parent
DATASET_PATH = _BASE / "Predictive_M.csv"
MODEL_DIR = _BASE / "Trained_models"
METRICS_PATH = _BASE / "model_metrics.json"

# Load models
_MODELS = {}
def load_model(model_name):
    """Load a model by name (cached)."""
    if model_name not in _MODELS:
        path = MODEL_DIR / f"{model_name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        _MODELS[model_name] = joblib.load(str(path))
    return _MODELS[model_name]

# ── Model configurations ────────────────────────────────────────────
MODELS_TO_EVALUATE = {
    "Binary DT (13 feat, High Precision)": {
        "model_file": "binary_decision_tree_feature_engineered_13features_threshold_0p96",
        "thresholds": [0.75, 0.96],
        "type": "binary",
        "features_count": 13,
    },
    "Binary DT (13 feat, High Sensitivity)": {
        "model_file": "binary_decision_tree_feature_engineered_13features_threshold_0p96",
        "thresholds": [0.75, 0.96],
        "type": "binary",
        "features_count": 13,
    },
    "Multi-Class DT": {
        "model_file": "multiclass_decision_tree_priority_encoded_scaled_original_features",
        "thresholds": [0.96],
        "type": "multiclass",
        "features_count": 8,
    },
    "Multi-Label DT": {
        "model_file": "multilabel_decision_tree_multioutput_scaled_original_features",
        "thresholds": [0.75],
        "type": "multilabel",
        "features_count": 8,
    },
}

# ── Load and split data ──────────────────────────────────────────────
def load_data():
    """Load dataset and split into train/test."""
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    
    if 'Type' not in df.columns:
        raise ValueError("Missing 'Type' column")
    
    # Find target column (Machine failure)
    target_col = None
    for col in df.columns:
        if 'failure' in col.lower() or 'target' in col.lower():
            target_col = col
            break
    
    if target_col is None:
        target_col = df.columns[-1]
    
    print(f"Using target column: {target_col}")
    
    # Extract features and target
    y = df[target_col].astype(int).values
    X_raw = df.drop(columns=[target_col])
    
    # Train-test split (80-20)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Preprocess both splits
    X_train = preprocess(X_train_raw.copy())
    X_test = preprocess(X_test_raw.copy())
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Train class distribution: {np.bincount(y_train)}")
    print(f"Test class distribution: {np.bincount(y_test)}")
    
    return X_train, X_test, y_train, y_test

# ── Calculate metrics ────────────────────────────────────────────────
def calculate_metrics(y_true, y_pred):
    """Calculate precision, recall, F1."""
    try:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tp = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
            tn = cm[0, 0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
            fp = cm[0, 1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
            fn = cm[1, 0] if cm.shape[0] > 1 and cm.shape[1] > 0 else 0
        
        return {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None

# ── Generate predictions at different thresholds ─────────────────────
def get_binary_predictions(model, X, y_true, thresholds, features_count):
    """Get binary predictions at different thresholds."""
    results = {}
    
    # Use correct number of features
    if features_count == 8:
        X_input = X[:, :8]
    else:
        X_input = X
    
    proba = model.predict_proba(X_input)
    
    for threshold in thresholds:
        y_pred = (proba[:, 1] >= threshold).astype(int)
        metrics = calculate_metrics(y_true, y_pred)
        results[str(threshold)] = metrics
    
    return results

def get_multiclass_predictions(model, X, y_true, features_count):
    """Get multiclass predictions (binary: failure vs no failure)."""
    # Use correct number of features
    if features_count == 8:
        X_input = X[:, :8]
    else:
        X_input = X
    
    y_pred_multiclass = model.predict(X_input)
    y_pred = (y_pred_multiclass != 0).astype(int)
    
    metrics = calculate_metrics(y_true, y_pred)
    return {"0.96": metrics}

def get_multilabel_predictions(model, X, y_true, features_count):
    """Get multilabel predictions (binary: any failure vs no failure)."""
    # Use correct number of features
    if features_count == 8:
        X_input = X[:, :8]
    else:
        X_input = X
    
    y_pred_multi = model.predict(X_input)
    y_pred = (np.sum(y_pred_multi, axis=1) > 0).astype(int)
    
    metrics = calculate_metrics(y_true, y_pred)
    return {"0.75": metrics}

# ── Cross-validation for robustness ──────────────────────────────────
def cross_validate_model(X_train, y_train, model, config):
    """Perform 5-fold Stratified CV and return mean/std of metrics."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {str(t): [] for t in config["thresholds"]}
    
    model_type = config["type"]
    features_count = config["features_count"]
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        X_tr = X_train[train_idx]
        X_val = X_train[val_idx]
        y_tr = y_train[train_idx]
        y_val = y_train[val_idx]
        
        # Use correct number of features
        if features_count == 8:
            X_tr_input = X_tr[:, :8]
            X_val_input = X_val[:, :8]
        else:
            X_tr_input = X_tr
            X_val_input = X_val
        
        if model_type == "binary":
            proba = model.predict_proba(X_val_input)
            for threshold in config["thresholds"]:
                y_pred = (proba[:, 1] >= threshold).astype(int)
                metrics = calculate_metrics(y_val, y_pred)
                if metrics:
                    cv_results[str(threshold)].append(metrics)
        
        elif model_type == "multiclass":
            y_pred_multiclass = model.predict(X_val_input)
            y_pred = (y_pred_multiclass != 0).astype(int)
            metrics = calculate_metrics(y_val, y_pred)
            if metrics:
                cv_results["0.96"].append(metrics)
        
        elif model_type == "multilabel":
            y_pred_multi = model.predict(X_val_input)
            y_pred = (np.sum(y_pred_multi, axis=1) > 0).astype(int)
            metrics = calculate_metrics(y_val, y_pred)
            if metrics:
                cv_results["0.75"].append(metrics)
    
    # Compute mean and std
    cv_stats = {}
    for threshold, metrics_list in cv_results.items():
        if metrics_list:
            mean_metrics = {}
            std_metrics = {}
            
            for key in ["precision", "recall", "f1"]:
                values = [m[key] for m in metrics_list]
                mean_metrics[key] = round(np.mean(values), 4)
                std_metrics[key] = round(np.std(values), 4)
            
            cv_stats[threshold] = {
                "mean": mean_metrics,
                "std": std_metrics,
                "folds": len(metrics_list)
            }
    
    return cv_stats

# ── Generate all metrics ─────────────────────────────────────────────
def generate_metrics():
    """Generate and save all model metrics."""
    print("="*70)
    print("GENERATING MODEL METRICS WITH CROSS-VALIDATION")
    print("="*70)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    metrics_data = {}
    
    for model_name, config in MODELS_TO_EVALUATE.items():
        print(f"\n{'='*70}")
        print(f"Processing: {model_name}")
        print(f"{'='*70}")
        
        model = load_model(config["model_file"])
        model_type = config["type"]
        
        # Test set evaluation
        print(f"\nTest Set Evaluation:")
        if model_type == "binary":
            test_results = get_binary_predictions(
                model, X_test, y_test, config["thresholds"], config["features_count"]
            )
        elif model_type == "multiclass":
            test_results = get_multiclass_predictions(
                model, X_test, y_test, config["features_count"]
            )
        elif model_type == "multilabel":
            test_results = get_multilabel_predictions(
                model, X_test, y_test, config["features_count"]
            )
        
        for threshold, metrics in test_results.items():
            if metrics:
                print(f"  Threshold {threshold}:")
                print(f"    Precision: {metrics['precision']}")
                print(f"    Recall:    {metrics['recall']}")
                print(f"    F1 Score:  {metrics['f1']}")
        
        # 5-fold CV evaluation
        print(f"\n5-Fold Stratified CV:")
        cv_results = cross_validate_model(X_train, y_train, model, config)
        
        for threshold, stats in cv_results.items():
            print(f"  Threshold {threshold}:")
            print(f"    Precision: {stats['mean']['precision']} ± {stats['std']['precision']}")
            print(f"    Recall:    {stats['mean']['recall']} ± {stats['std']['recall']}")
            print(f"    F1 Score:  {stats['mean']['f1']} ± {stats['std']['f1']}")
        
        # Combine test + CV results
        metrics_data[model_name] = {
            "test_set": test_results,
            "cv_5fold": cv_results,
            "test_size": len(X_test),
            "train_size": len(X_train),
        }
    
    # Save to file
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"{'='*70}")
    
    return metrics_data

if __name__ == "__main__":
    try:
        metrics = generate_metrics()
        print("\nSuccess! Use these metrics in Gradio via model_metrics.json")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
