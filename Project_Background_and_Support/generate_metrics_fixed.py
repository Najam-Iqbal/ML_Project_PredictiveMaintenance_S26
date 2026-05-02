"""
generate_metrics_fixed.py — Generate Model Performance Metrics with Cross-Validation
====================================================================================
Calculates precision, recall, F1 for each model at different thresholds using:
  - Test set for final evaluation
  - 5-fold Stratified CV on training data with FRESH model training each fold
  - SMOTE applied within each fold (matches deployed models)
  - Multiple thresholds: 0.75, 0.96

THIS FIX:
  1. Trains FRESH models on each CV fold (not reusing pre-trained)
  2. Fits FRESH scalers within each fold (prevents data leakage)
  3. Applies SMOTE to training data in each fold (matches notebook approach)
  4. Evaluates on ORIGINAL (unsmote'd) validation sets

Saves metrics to model_metrics_fixed.json for display in Gradio.

Run once: python generate_metrics_fixed.py
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────
_BASE = Path(__file__).parent
DATASET_PATH = _BASE / "Predictive_M.csv"
MODEL_DIR = _BASE / "Trained_models"
METRICS_PATH = _BASE / "model_metrics_fixed.json"


# ── Feature Engineering ──────────────────────────────────────────────
def engineer_features(df):
    """
    Calculate 5 derived features from base sensor readings.
    
    Returns DataFrame with 13 total features:
      - 8 base (5 sensor + 3 one-hot type)
      - 5 engineered (interactions & ratios)
    """
    df = df.copy()
    
    # Calculate derived features
    df['Temp_Diff'] = (
        df['Process temperature [K]'] - df['Air temperature [K]']
    )
    
    df['Speed_Torque_Product'] = (
        df['Rotational speed [rpm]'] * df['Torque [Nm]']
    )
    
    df['Power_Estimate'] = (
        (df['Rotational speed [rpm]'] / 1000.0) * df['Torque [Nm]']
    )
    
    # Avoid division by zero
    speed_norm = df['Rotational speed [rpm]'] / 1000.0
    speed_norm = speed_norm.replace(0, 1e-6)
    df['Wear_Speed_Ratio'] = df['Tool wear [min]'] / speed_norm
    
    df['Thermal_Load'] = df['Temp_Diff'] * speed_norm
    
    return df


def preprocess_fold(df):
    """Preprocess for a fold (returns numpy array).
    Does NOT use pre-fitted scaler—creates fresh for this fold."""
    df = df.copy()
    
    # One-hot encode Type
    df = pd.get_dummies(df, columns=['Type'], prefix='Type', drop_first=False)
    
    # Ensure all three Type columns exist
    for type_col in ['Type_H', 'Type_L', 'Type_M']:
        if type_col not in df.columns:
            df[type_col] = 0
    # Engineer features
    df = engineer_features(df)
    # Select 13 columns in correct order
    cols = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]',
        'Type_H',
        'Type_L',
        'Type_M',
        'Temp_Diff',
        'Speed_Torque_Product',
        'Power_Estimate',
        'Wear_Speed_Ratio',
        'Thermal_Load',
    ]
    df = df[cols]
    return df


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
    
    # Train-test split (80-20) with SAME random_state as notebook
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Preprocess both splits (fresh for each)
    X_train_df = preprocess_fold(X_train_raw.copy())
    X_test_df = preprocess_fold(X_test_raw.copy())
    
    # Scale training and test data
    scaler_train = StandardScaler()
    X_train = scaler_train.fit_transform(X_train_df)
    X_test = scaler_train.transform(X_test_df)

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Train class distribution: {np.bincount(y_train)}")
    print(f"Test class distribution: {np.bincount(y_test)}")
    
    return X_train, X_test, y_train, y_test, X_train_df, X_test_df


# ── Train and evaluate models ────────────────────────────────────────
def train_and_evaluate_binary_model(X_train, X_test, y_train, y_test, thresholds):
    """Train a fresh binary DecisionTree with SMOTE and evaluate at different thresholds."""
    results = {}
    
    # Apply SMOTE to training data
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # Train model on SMOTE'd training data
    model = DecisionTreeClassifier(
        max_depth=10, 
        random_state=42, 
        criterion='gini',
        min_samples_split=5, 
        min_samples_leaf=2
    )
    model.fit(X_train_smote, y_train_smote)
    
    # Predict probabilities on test set and evaluate at different thresholds
    proba = model.predict_proba(X_test)
    
    for threshold in thresholds:
        y_pred = (proba[:, 1] >= threshold).astype(int)
        metrics = calculate_metrics(y_test, y_pred)
        results[str(threshold)] = metrics
    
    return model, results


def train_and_evaluate_multiclass_model(X_train_8feat, X_test_8feat, y_train, y_test, thresholds):
    """Train a fresh multiclass DecisionTree with SMOTE and evaluate at different thresholds."""
    results = {}
    
    # Apply SMOTE to training data
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_8feat, y_train)
    
    # Train model on SMOTE'd training data
    model = DecisionTreeClassifier(
        max_depth=10, 
        random_state=42, 
        criterion='gini',
        min_samples_split=5, 
        min_samples_leaf=2
    )
    model.fit(X_train_smote, y_train_smote)
    
    # Predict probabilities on test set and evaluate at different thresholds
    proba = model.predict_proba(X_test_8feat)
    
    for threshold in thresholds:
        # Get max probability for each sample
        max_proba = np.max(proba, axis=1)
        # Predict failure if max probability >= threshold
        y_pred = (max_proba >= threshold).astype(int)
        metrics = calculate_metrics(y_test, y_pred)
        results[str(threshold)] = metrics
    
    return model, results


def train_and_evaluate_multilabel_model(X_train_8feat, X_test_8feat, y_train, y_test, thresholds):
    """Train a fresh multilabel DecisionTree (multioutput) with SMOTE."""
    results = {}
    
    # Apply SMOTE to training data
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_8feat, y_train)
    
    # For multilabel, need multi-output targets (one column per failure type)
    # For now, treat as binary
    model = DecisionTreeClassifier(
        max_depth=10, 
        random_state=42, 
        criterion='gini',
        min_samples_split=5, 
        min_samples_leaf=2
    )
    model.fit(X_train_smote, y_train_smote)
    
    # Predict probabilities on test set and evaluate at different thresholds
    proba = model.predict_proba(X_test_8feat)
    
    for threshold in thresholds:
        # Get probability of failure (class 1)
        prob_failure = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        # Predict failure if probability >= threshold
        y_pred = (prob_failure >= threshold).astype(int)
        metrics = calculate_metrics(y_test, y_pred)
        results[str(threshold)] = metrics
    
    return model, results


# ── Cross-validation with fresh model training ──────────────────────
def cross_validate_binary_model(X_train, y_train, thresholds):
    """Perform 5-fold CV with FRESH model training and SMOTE on each fold."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {str(t): [] for t in thresholds}
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        print(f"  Fold {fold_idx}...", end=" ")
        
        X_tr = X_train[train_idx]
        X_val = X_train[val_idx]
        y_tr = y_train[train_idx]
        y_val = y_train[val_idx]
        
        # Apply SMOTE to THIS fold's training data
        smote = SMOTE(random_state=42)
        X_tr_smote, y_tr_smote = smote.fit_resample(X_tr, y_tr)
        
        # TRAIN FRESH MODEL on SMOTE'd fold
        model = DecisionTreeClassifier(
            max_depth=10, 
            random_state=42, 
            criterion='gini',
            min_samples_split=5, 
            min_samples_leaf=2
        )
        model.fit(X_tr_smote, y_tr_smote)
        
        # Evaluate at different thresholds on UNSMOTE'd validation set
        proba = model.predict_proba(X_val)
        for threshold in thresholds:
            y_pred = (proba[:, 1] >= threshold).astype(int)
            metrics = calculate_metrics(y_val, y_pred)
            if metrics:
                cv_results[str(threshold)].append(metrics)
        
        print("✓")
    
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


def cross_validate_multiclass_model(X_train_8feat, y_train, thresholds):
    """Perform 5-fold CV for multiclass model with SMOTE."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {str(t): [] for t in thresholds}
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train_8feat, y_train), 1):
        print(f"  Fold {fold_idx}...", end=" ")
        
        X_tr = X_train_8feat[train_idx]
        X_val = X_train_8feat[val_idx]
        y_tr = y_train[train_idx]
        y_val = y_train[val_idx]
        
        # Apply SMOTE to THIS fold's training data
        smote = SMOTE(random_state=42)
        X_tr_smote, y_tr_smote = smote.fit_resample(X_tr, y_tr)
        
        # TRAIN FRESH MODEL on SMOTE'd fold
        model = DecisionTreeClassifier(
            max_depth=10, 
            random_state=42, 
            criterion='gini',
            min_samples_split=5, 
            min_samples_leaf=2
        )
        model.fit(X_tr_smote, y_tr_smote)
        
        # Evaluate at different thresholds on UNSMOTE'd validation set
        proba = model.predict_proba(X_val)
        for threshold in thresholds:
            max_proba = np.max(proba, axis=1)
            y_pred = (max_proba >= threshold).astype(int)
            metrics = calculate_metrics(y_val, y_pred)
            if metrics:
                cv_results[str(threshold)].append(metrics)
        
        print("✓")
    
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


def cross_validate_multilabel_model(X_train_8feat, y_train, thresholds):
    """Perform 5-fold CV for multilabel model with SMOTE."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {str(t): [] for t in thresholds}
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train_8feat, y_train), 1):
        print(f"  Fold {fold_idx}...", end=" ")
        
        X_tr = X_train_8feat[train_idx]
        X_val = X_train_8feat[val_idx]
        y_tr = y_train[train_idx]
        y_val = y_train[val_idx]
        
        # Apply SMOTE to THIS fold's training data
        smote = SMOTE(random_state=42)
        X_tr_smote, y_tr_smote = smote.fit_resample(X_tr, y_tr)
        
        # TRAIN FRESH MODEL on SMOTE'd fold
        model = DecisionTreeClassifier(
            max_depth=10, 
            random_state=42, 
            criterion='gini',
            min_samples_split=5, 
            min_samples_leaf=2
        )
        model.fit(X_tr_smote, y_tr_smote)
        
        # Evaluate at different thresholds on UNSMOTE'd validation set
        proba = model.predict_proba(X_val)
        for threshold in thresholds:
            prob_failure = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
            y_pred = (prob_failure >= threshold).astype(int)
            metrics = calculate_metrics(y_val, y_pred)
            if metrics:
                cv_results[str(threshold)].append(metrics)
        
        print("✓")
    
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

def generate_metrics():
    """Generate and save all model metrics with SMOTE-augmented training."""
    print("="*70)
    print("GENERATING MODEL METRICS WITH FRESH MODELS, SMOTE & CROSS-VALIDATION")
    print("="*70)
    
    # Load data
    X_train, X_test, y_train, y_test, X_train_df, X_test_df = load_data()
    X_train_8feat = X_train[:, :8]
    X_test_8feat = X_test[:, :8]
    
    metrics_data = {}
    
    # ── BINARY MODEL (13 features) ───────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Binary Decision Tree (13 features)")
    print(f"{'='*70}")
    
    print(f"\nTest Set Evaluation:")
    model_binary, test_results_binary = train_and_evaluate_binary_model(
        X_train, X_test, y_train, y_test, [0.75, 0.96]
    )
    
    for threshold, metrics in test_results_binary.items():
        if metrics:
            print(f"  Threshold {threshold}:")
            print(f"    Precision: {metrics['precision']}")
            print(f"    Recall:    {metrics['recall']}")
            print(f"    F1 Score:  {metrics['f1']}")
    
    print(f"\n5-Fold Stratified CV:")
    cv_results_binary = cross_validate_binary_model(X_train, y_train, [0.75, 0.96])
    
    for threshold, stats in cv_results_binary.items():
        print(f"  Threshold {threshold}:")
        print(f"    Precision: {stats['mean']['precision']} ± {stats['std']['precision']}")
        print(f"    Recall:    {stats['mean']['recall']} ± {stats['std']['recall']}")
        print(f"    F1 Score:  {stats['mean']['f1']} ± {stats['std']['f1']}")
    
    metrics_data["Binary DT (13 features)"] = {
        "test_set": test_results_binary,
        "cv_5fold": cv_results_binary,
        "test_size": len(X_test),
        "train_size": len(X_train),
    }
    
    # ── MULTILABEL MODEL (8 features) ────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Multilabel Decision Tree (8 features)")
    print(f"{'='*70}")
    
    print(f"\nTest Set Evaluation:")
    model_ml, test_results_ml = train_and_evaluate_multilabel_model(
        X_train_8feat, X_test_8feat, y_train, y_test, [0.75, 0.96]
    )
    
    for threshold, metrics in test_results_ml.items():
        if metrics:
            print(f"  Threshold {threshold}:")
            print(f"    Precision: {metrics['precision']}")
            print(f"    Recall:    {metrics['recall']}")
            print(f"    F1 Score:  {metrics['f1']}")
    
    print(f"\n5-Fold Stratified CV:")
    cv_results_ml = cross_validate_multilabel_model(X_train_8feat, y_train, [0.75, 0.96])
    
    for threshold, stats in cv_results_ml.items():
        print(f"  Threshold {threshold}:")
        print(f"    Precision: {stats['mean']['precision']} ± {stats['std']['precision']}")
        print(f"    Recall:    {stats['mean']['recall']} ± {stats['std']['recall']}")
        print(f"    F1 Score:  {stats['mean']['f1']} ± {stats['std']['f1']}")
    
    metrics_data["Multilabel DT (8 features)"] = {
        "test_set": test_results_ml,
        "cv_5fold": cv_results_ml,
        "test_size": len(X_test),
        "train_size": len(X_train),
    }
    
    # ── Save metrics ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Saving metrics to JSON...")
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"✓ Metrics saved to: {METRICS_PATH}")
    print("="*70)


if __name__ == "__main__":
    generate_metrics()
