"""
Predictive Maintenance Module
Trains Naive Bayes and Decision Tree models to predict machine failures

Author: Predictive Maintenance Project
Date: April 2026

Key preprocessing order: Scaler → SMOTE → Model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix, classification_report, hamming_loss
)
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# PHASE 1: DATA LOADING & EDA
# ============================================================================

def load_data(filepath):
    """
    Load the predictive maintenance dataset.
    
    Args:
        filepath (str): Path to Predictive_M.csv
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    df = pd.read_csv(filepath)
    print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def exploratory_analysis(df):
    """
    Perform exploratory data analysis on the dataset.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        dict: Summary statistics
    """
    print("\n" + "="*70)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    # Basic info
    print(f"\nDataset shape: {df.shape}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # Machine failure distribution
    failure_counts = df['Machine failure'].value_counts()
    failure_pcts = df['Machine failure'].value_counts(normalize=True) * 100
    print(f"\n{'Machine Failure Distribution:':^70}")
    print(f"  No Failure (0): {failure_counts[0]:>6} ({failure_pcts[0]:.2f}%)")
    print(f"  Failure (1):    {failure_counts[1]:>6} ({failure_pcts[1]:.2f}%)")
    
    # Failure types
    failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    print(f"\n{'Failure Types Distribution:':^70}")
    for ft in failure_types:
        count = df[ft].sum()
        pct = (count / len(df)) * 100
        print(f"  {ft}: {count:>6} ({pct:.2f}%)")
    
    # Product types
    print(f"\n{'Product Types:':^70}")
    print(df['Type'].value_counts())
    
    # Feature statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"\n{'Feature Statistics (numeric):':^70}")
    print(df[numeric_cols].describe().round(2))
    
    stats = {
        'total_samples': len(df),
        'failure_rate': failure_pcts[1],
        'failure_count': failure_counts[1],
        'product_types': df['Type'].unique(),
    }
    
    return stats


def plot_eda_visualizations(df, output_path=None):
    """
    Create EDA visualizations.
    
    Args:
        df (pd.DataFrame): Input dataset
        output_path (str): Optional path to save figures
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Exploratory Data Analysis - Predictive Maintenance', fontsize=14, fontweight='bold')
    
    # 1. Machine Failure Distribution
    ax = axes[0, 0]
    failure_counts = df['Machine failure'].value_counts()
    ax.bar(['No Failure', 'Failure'], failure_counts.values, color=['green', 'red'], alpha=0.7)
    ax.set_title('Machine Failure Distribution')
    ax.set_ylabel('Count')
    for i, v in enumerate(failure_counts.values):
        ax.text(i, v + 50, str(v), ha='center', fontweight='bold')
    
    # 2. Failure Types
    ax = axes[0, 1]
    failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    ft_counts = [df[ft].sum() for ft in failure_types]
    ax.bar(failure_types, ft_counts, color='skyblue', alpha=0.7)
    ax.set_title('Failure Types Distribution')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)
    
    # 3. Product Type vs Machine Failure
    ax = axes[0, 2]
    product_failure = pd.crosstab(df['Type'], df['Machine failure'])
    product_failure.plot(kind='bar', ax=ax, color=['green', 'red'], alpha=0.7)
    ax.set_title('Failure Rate by Product Type')
    ax.set_ylabel('Count')
    ax.set_xlabel('Product Type')
    ax.legend(['No Failure', 'Failure'])
    ax.tick_params(axis='x', rotation=0)
    
    # 4. Rotational Speed Distribution
    ax = axes[1, 0]
    ax.hist(df['Rotational speed [rpm]'], bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax.set_title('Rotational Speed Distribution')
    ax.set_xlabel('Speed (rpm)')
    ax.set_ylabel('Frequency')
    
    # 5. Tool Wear Distribution
    ax = axes[1, 1]
    ax.hist(df['Tool wear [min]'], bins=50, color='orange', alpha=0.7, edgecolor='black')
    ax.set_title('Tool Wear Distribution')
    ax.set_xlabel('Tool Wear (min)')
    ax.set_ylabel('Frequency')
    
    # 6. Process Temperature vs Machine Failure
    ax = axes[1, 2]
    for failure_val in [0, 1]:
        label = 'No Failure' if failure_val == 0 else 'Failure'
        color = 'green' if failure_val == 0 else 'red'
        data = df[df['Machine failure'] == failure_val]['Process temperature [K]']
        ax.hist(data, bins=30, alpha=0.6, label=label, color=color)
    ax.set_title('Process Temperature by Failure Status')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Frequency')
    ax.legend()
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    return fig


def plot_correlation_heatmap(df, output_path=None):
    """
    Plot correlation heatmap of numeric features and targets.
    
    Args:
        df (pd.DataFrame): Input dataset
        output_path (str): Optional path to save figure
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('Feature Correlation Matrix - Predictive Maintenance', fontweight='bold', fontsize=12)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    return fig


# ============================================================================
# PHASE 2: DATA PREPROCESSING
# ============================================================================

def preprocess_data(df, target_col='Machine failure'):
    """
    Prepare data for modeling: Feature engineering, split, scale, SMOTE.
    
    IMPORTANT ORDER: Split → Scale → SMOTE
    
    Args:
        df (pd.DataFrame): Input dataset
        target_col (str): Target column name
        
    Returns:
        dict: Preprocessed data with scaler, SMOTE, and split datasets
    """
    print("\n" + "="*70)
    print("DATA PREPROCESSING")
    print("="*70)
    
    # Feature engineering: One-hot encode 'Type'
    df_processed = df.copy()
    df_processed = pd.get_dummies(df_processed, columns=['Type'], prefix='Type', drop_first=False)
    
    # Define feature columns (exclude UDI, Product ID, Machine failure, and individual failure types)
    feature_cols = ['Air temperature [K]', 'Process temperature [K]', 
                    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
                    'Type_H', 'Type_L', 'Type_M']
    
    X = df_processed[feature_cols]
    y = df_processed[target_col]
    
    print(f"\nFeatures selected: {len(feature_cols)}")
    print(f"Target variable: {target_col}")
    print(f"Original dataset: {X.shape}")
    
    # STEP 1: Train/Test Split with stratification
    print(f"\n[STEP 1] Train/Test Split (80/20) with stratification...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Training set: {X_train.shape} | Failure rate: {y_train.mean()*100:.2f}%")
    print(f"  Test set:     {X_test.shape}  | Failure rate: {y_test.mean()*100:.2f}%")
    
    # STEP 2: Fit scaler on training data, transform both train & test
    print(f"\n[STEP 2] Fit StandardScaler on training data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Transform test using fitted scaler
    print(f"  ✓ Scaler fitted on training data")
    print(f"  ✓ Training data scaled: mean={X_train_scaled.mean():.4f}, std={X_train_scaled.std():.4f}")
    print(f"  ✓ Test data scaled using fitted scaler")
    
    # STEP 3: Apply SMOTE to scaled training data
    print(f"\n[STEP 3] Apply SMOTE to scaled training data...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    print(f"  Training set after SMOTE: {X_train_smote.shape}")
    print(f"  Failure rate after SMOTE: {y_train_smote.mean()*100:.2f}%")
    print(f"  Original: {len(y_train)}, After SMOTE: {len(y_train_smote)}")
    
    # Create DataFrames for easier handling
    X_train_smote_df = pd.DataFrame(X_train_smote, columns=feature_cols)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_cols)
    
    result = {
        'X_train': X_train_smote_df,
        'X_test': X_test_scaled_df,
        'y_train': y_train_smote,
        'y_test': y_test,
        'scaler': scaler,
        'smote': smote,
        'feature_columns': feature_cols,
        'X_train_original': X_train,
        'X_test_original': X_test,
        'y_train_original': y_train,
    }
    
    print(f"\n✓ Preprocessing complete!")
    return result


# ============================================================================
# PHASE 3: BINARY CLASSIFICATION - TASK 2a & 2b
# ============================================================================

def train_naive_bayes_binary(X_train, y_train):
    """
    Train Gaussian Naive Bayes for binary classification.
    
    Args:
        X_train (pd.DataFrame or np.ndarray): Training features
        y_train (pd.Series or np.ndarray): Training labels
        
    Returns:
        GaussianNB: Trained model
    """
    print(f"\n{'Training Gaussian Naive Bayes (Binary)':^70}")
    model = GaussianNB()
    model.fit(X_train, y_train)
    print(f"✓ Model trained successfully")
    return model


def train_decision_tree_binary(X_train, y_train, max_depth=10, random_state=42):
    """
    Train Decision Tree for binary classification.
    
    Args:
        X_train (pd.DataFrame or np.ndarray): Training features
        y_train (pd.Series or np.ndarray): Training labels
        max_depth (int): Maximum depth of tree
        random_state (int): Random state for reproducibility
        
    Returns:
        DecisionTreeClassifier: Trained model
    """
    print(f"\n{'Training Decision Tree (Binary)':^70}")
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state, 
                                   criterion='gini', min_samples_split=5, min_samples_leaf=2)
    model.fit(X_train, y_train)
    print(f"✓ Model trained successfully")
    return model


def evaluate_binary_model(model, X_train, y_train, X_test, y_test, model_name):
    """
    Evaluate binary classification model.
    
    Args:
        model: Trained model
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_name (str): Name of the model (for display)
        
    Returns:
        dict: Evaluation metrics
    """
    print(f"\n{'Evaluating ' + model_name:^70}")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Probabilities for ROC-AUC
    if hasattr(model, 'predict_proba'):
        y_test_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_test_proba)
    else:
        y_test_proba = None
        roc_auc = None
    
    # Metrics
    metrics = {
        'model_name': model_name,
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred, zero_division=0),
        'recall': recall_score(y_test, y_test_pred, zero_division=0),
        'f1': f1_score(y_test, y_test_pred, zero_division=0),
        'roc_auc': roc_auc,
        'y_test_pred': y_test_pred,
        'y_test_proba': y_test_proba,
        'confusion_matrix': confusion_matrix(y_test, y_test_pred),
    }
    
    print(f"  Train Accuracy:  {metrics['train_accuracy']:.4f}")
    print(f"  Test Accuracy:   {metrics['test_accuracy']:.4f}")
    print(f"  Precision:       {metrics['precision']:.4f}")
    print(f"  Recall:          {metrics['recall']:.4f}")
    print(f"  F1-Score:        {metrics['f1']:.4f}")
    if roc_auc is not None:
        print(f"  ROC-AUC:         {roc_auc:.4f}")
    
    return metrics


def plot_confusion_matrices(metrics_nb, metrics_dt, output_path=None):
    """
    Plot confusion matrices for both models.
    
    Args:
        metrics_nb (dict): Naive Bayes evaluation metrics
        metrics_dt (dict): Decision Tree evaluation metrics
        output_path (str): Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Binary Classification - Confusion Matrices', fontweight='bold', fontsize=12)
    
    for idx, metrics in enumerate([metrics_nb, metrics_dt]):
        ax = axes[idx]
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                   xticklabels=['No Failure', 'Failure'],
                   yticklabels=['No Failure', 'Failure'])
        ax.set_title(f"{metrics['model_name']}")
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    return fig


def plot_roc_curves(metrics_nb, metrics_dt, y_test, output_path=None):
    """
    Plot ROC curves for both models.
    
    Args:
        metrics_nb (dict): Naive Bayes evaluation metrics
        metrics_dt (dict): Decision Tree evaluation metrics
        y_test (pd.Series): Test labels
        output_path (str): Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot for Naive Bayes
    if metrics_nb['y_test_proba'] is not None:
        fpr_nb, tpr_nb, _ = roc_curve(y_test, metrics_nb['y_test_proba'])
        auc_nb = metrics_nb['roc_auc']
        ax.plot(fpr_nb, tpr_nb, label=f"Naive Bayes (AUC={auc_nb:.4f})", linewidth=2, marker='o', markersize=4)
    
    # Plot for Decision Tree
    if metrics_dt['y_test_proba'] is not None:
        fpr_dt, tpr_dt, _ = roc_curve(y_test, metrics_dt['y_test_proba'])
        auc_dt = metrics_dt['roc_auc']
        ax.plot(fpr_dt, tpr_dt, label=f"Decision Tree (AUC={auc_dt:.4f})", linewidth=2, marker='s', markersize=4)
    
    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curves - Binary Classification', fontweight='bold', fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    return fig


def plot_feature_importance_dt(model, feature_columns, output_path=None):
    """
    Plot feature importance from Decision Tree.
    
    Args:
        model (DecisionTreeClassifier): Trained Decision Tree
        feature_columns (list): Feature column names
        output_path (str): Optional path to save figure
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(importances)), importances[indices], alpha=0.7, color='steelblue')
    ax.set_xlabel('Feature', fontweight='bold')
    ax.set_ylabel('Importance', fontweight='bold')
    ax.set_title('Feature Importance - Decision Tree (Binary)', fontweight='bold', fontsize=12)
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_columns[i] for i in indices], rotation=45, ha='right')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    return fig


# ============================================================================
# PHASE 4: MULTI-CLASS / MULTI-LABEL - TASKS 3a & 3b
# ============================================================================

def prepare_multiclass_target(df, failure_types=['TWF', 'HDF', 'PWF', 'OSF', 'RNF']):
    """
    Create multi-class target: 0=No failure, 1-5 = Individual failure types.
    Priority: TWF > HDF > PWF > OSF > RNF
    
    Args:
        df (pd.DataFrame): Dataset with failure type columns
        failure_types (list): List of failure type column names
        
    Returns:
        np.ndarray: Multi-class target (0-5)
    """
    target = np.zeros(len(df), dtype=int)
    
    for i, ft in enumerate(failure_types, start=1):
        target[df[ft] == 1] = i
    
    return target


def train_multilabel_models(X_train, y_train, failure_types):
    """
    Train multi-label classifiers (Naive Bayes & Decision Tree).
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.DataFrame): Training labels (one column per failure type)
        failure_types (list): List of failure type column names
        
    Returns:
        tuple: (model_nb, model_dt) - Trained multi-output models
    """
    print(f"\n{'Training Multi-Label Models':^70}")
    
    # Naive Bayes
    print(f"\n  [Multi-label] Training Naive Bayes...")
    nb_model = MultiOutputClassifier(GaussianNB())
    nb_model.fit(X_train, y_train)
    print(f"  ✓ Naive Bayes trained")
    
    # Decision Tree
    print(f"  [Multi-label] Training Decision Tree...")
    dt_model = MultiOutputClassifier(DecisionTreeClassifier(max_depth=10, random_state=42))
    dt_model.fit(X_train, y_train)
    print(f"  ✓ Decision Tree trained")
    
    return nb_model, dt_model


def train_multiclass_models(X_train, y_train_multiclass):
    """
    Train multi-class classifiers (Naive Bayes & Decision Tree).
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train_multiclass (np.ndarray): Training labels (6 classes: 0-5)
        
    Returns:
        tuple: (model_nb, model_dt) - Trained models
    """
    print(f"\n{'Training Multi-Class Models (6 classes)':^70}")
    
    # Naive Bayes
    print(f"\n  [Multi-class] Training Naive Bayes...")
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train_multiclass)
    print(f"  ✓ Naive Bayes trained")
    
    # Decision Tree
    print(f"  [Multi-class] Training Decision Tree...")
    dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt_model.fit(X_train, y_train_multiclass)
    print(f"  ✓ Decision Tree trained")
    
    return nb_model, dt_model


def evaluate_multilabel_models(model_nb, model_dt, X_test, y_test):
    """
    Evaluate multi-label models.
    
    Args:
        model_nb: Multi-output Naive Bayes
        model_dt: Multi-output Decision Tree
        X_test (pd.DataFrame): Test features
        y_test (pd.DataFrame): Test labels (one column per failure type)
        
    Returns:
        dict: Evaluation metrics
    """
    print(f"\n{'Evaluating Multi-Label Models':^70}")
    
    y_pred_nb = model_nb.predict(X_test)
    y_pred_dt = model_dt.predict(X_test)
    
    metrics = {
        'Naive Bayes': {
            'hamming_loss': hamming_loss(y_test, y_pred_nb),
            'subset_accuracy': (y_pred_nb == y_test.values).all(axis=1).mean(),
        },
        'Decision Tree': {
            'hamming_loss': hamming_loss(y_test, y_pred_dt),
            'subset_accuracy': (y_pred_dt == y_test.values).all(axis=1).mean(),
        }
    }
    
    print(f"\n  Naive Bayes:")
    print(f"    Hamming Loss: {metrics['Naive Bayes']['hamming_loss']:.4f}")
    print(f"    Subset Accuracy: {metrics['Naive Bayes']['subset_accuracy']:.4f}")
    print(f"\n  Decision Tree:")
    print(f"    Hamming Loss: {metrics['Decision Tree']['hamming_loss']:.4f}")
    print(f"    Subset Accuracy: {metrics['Decision Tree']['subset_accuracy']:.4f}")
    
    return metrics, y_pred_nb, y_pred_dt


def evaluate_multiclass_models(model_nb, model_dt, X_test, y_test_multiclass):
    """
    Evaluate multi-class models.
    
    Args:
        model_nb: Naive Bayes model
        model_dt: Decision Tree model
        X_test (pd.DataFrame): Test features
        y_test_multiclass (np.ndarray): Test labels (0-5)
        
    Returns:
        tuple: (metrics, y_pred_nb, y_pred_dt)
    """
    print(f"\n{'Evaluating Multi-Class Models (6 classes)':^70}")
    
    y_pred_nb = model_nb.predict(X_test)
    y_pred_dt = model_dt.predict(X_test)
    
    metrics = {
        'Naive Bayes': {
            'accuracy': accuracy_score(y_test_multiclass, y_pred_nb),
            'f1_weighted': f1_score(y_test_multiclass, y_pred_nb, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_test_multiclass, y_pred_nb, average='macro', zero_division=0),
        },
        'Decision Tree': {
            'accuracy': accuracy_score(y_test_multiclass, y_pred_dt),
            'f1_weighted': f1_score(y_test_multiclass, y_pred_dt, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_test_multiclass, y_pred_dt, average='macro', zero_division=0),
        }
    }
    
    print(f"\n  Naive Bayes:")
    print(f"    Accuracy: {metrics['Naive Bayes']['accuracy']:.4f}")
    print(f"    F1-Score (weighted): {metrics['Naive Bayes']['f1_weighted']:.4f}")
    print(f"    F1-Score (macro): {metrics['Naive Bayes']['f1_macro']:.4f}")
    print(f"\n  Decision Tree:")
    print(f"    Accuracy: {metrics['Decision Tree']['accuracy']:.4f}")
    print(f"    F1-Score (weighted): {metrics['Decision Tree']['f1_weighted']:.4f}")
    print(f"    F1-Score (macro): {metrics['Decision Tree']['f1_macro']:.4f}")
    
    return metrics, y_pred_nb, y_pred_dt


# ============================================================================
# MODEL PERSISTENCE
# ============================================================================

def save_model(model, filepath):
    """Save trained model to disk."""
    joblib.dump(model, filepath)
    print(f"✓ Model saved to {filepath}")


def load_model(filepath):
    """Load trained model from disk."""
    model = joblib.load(filepath)
    print(f"✓ Model loaded from {filepath}")
    return model


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_comparison_table(metrics_nb, metrics_dt):
    """Create comparison table for binary classification models."""
    comparison = pd.DataFrame({
        'Metric': ['Train Accuracy', 'Test Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        'Naive Bayes': [
            f"{metrics_nb['train_accuracy']:.4f}",
            f"{metrics_nb['test_accuracy']:.4f}",
            f"{metrics_nb['precision']:.4f}",
            f"{metrics_nb['recall']:.4f}",
            f"{metrics_nb['f1']:.4f}",
            f"{metrics_nb['roc_auc']:.4f}" if metrics_nb['roc_auc'] else "N/A"
        ],
        'Decision Tree': [
            f"{metrics_dt['train_accuracy']:.4f}",
            f"{metrics_dt['test_accuracy']:.4f}",
            f"{metrics_dt['precision']:.4f}",
            f"{metrics_dt['recall']:.4f}",
            f"{metrics_dt['f1']:.4f}",
            f"{metrics_dt['roc_auc']:.4f}" if metrics_dt['roc_auc'] else "N/A"
        ]
    })
    return comparison


if __name__ == "__main__":
    print("Predictive Maintenance Module - Ready to import")
