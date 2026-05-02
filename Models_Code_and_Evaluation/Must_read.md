# Foundational Files: Building the Predictive Maintenance Gradio Project

This document explains how `Predictive_Maintenance_Project.ipynb` and `predictive_maintenance.py` form the **core foundation** of the Gradio-based predictive maintenance system.

---

## Overview

These two files represent the **research, experimentation, and model development phase** of the project. They were essential in:

1. **Understanding the data** through exploratory data analysis (EDA)
2. **Engineering features** from raw sensor data
3. **Training and evaluating multiple models** (Naive Bayes vs Decision Tree)
4. **Generating trained model artifacts** (`.joblib` files) for deployment
5. **Creating visualizations and metrics** that justify model selection
6. **Establishing best practices** for preprocessing and evaluation

---

## File Roles

### 1. `Predictive_Maintenance_Project.ipynb` (Jupyter Notebook)

**Purpose:** Interactive development and experimentation environment

**Key Contributions:**

- **Phase 1: Data Exploration & Analysis**
  - Loads `Predictive_M.csv` (10,000 equipment samples)
  - Performs EDA: class distribution, failure rates, feature statistics
  - Visualizes patterns: sensor readings, product types, failure correlations
  - Identifies class imbalance (3.4% failure rate)

- **Phase 2: Data Preprocessing**
  - Implements feature engineering pipeline:
    - One-hot encoding for product types (L, M, H)
    - Selection of 8 base features (air temp, process temp, RPM, torque, tool wear, type flags)
  - Creates train/test split (80/20, stratified)
  - Fits StandardScaler on training data
  - Applies SMOTE to balance training classes (1:1 ratio)
  - Validates preprocessing order: Split → Scale → SMOTE

- **Phase 3: Binary Classification (Task 2)**
  - Trains **Gaussian Naive Bayes** model
  - Trains **Decision Tree** model (max_depth=10, gini criterion)
  - Evaluates both on test set:
    - Accuracy, Precision, Recall, F1-Score, ROC-AUC
    - Confusion matrices, ROC curves
  - **Finding:** Decision Tree outperforms Naive Bayes (F1: 0.74 vs ~0.55)

- **Phase 4: Multi-Label & Multi-Class Classification (Task 3)**
  - Trains multi-output classifiers for concurrent failures
  - Evaluates on 6 failure types: TWF, HDF, PWF, OSF, RNF
  - Generates per-failure-type precision/recall metrics
  - Creates comparison visualizations

- **Phase 5: Model Persistence**
  - Saves trained models to `Trained_models/` directory as `.joblib` files
  - Exports scaler and SMOTE objects for inference consistency
  - Generates performance metrics JSON for documentation

- **Phase 6: Visualization & Documentation**
  - Plots: correlation heatmap, feature distributions, failure patterns
  - ROC curves comparing models
  - Feature importance rankings
  - Decision tree visualizations showing split logic

**Outputs Generated:**
- `Trained_models/binary_decision_tree_feature_engineered_13features_threshold_0p96.joblib`
- `Trained_models/multilabel_decision_tree_multioutput_scaled_original_features.joblib`
- `scaler.joblib` (fitted StandardScaler)
- Performance metrics and visualizations
- Decisions: threshold tuning (0.75 vs 0.96), model architecture

---

### 2. `predictive_maintenance.py` (Python Module)

**Purpose:** Reusable, production-ready functions for the ML pipeline

**Key Contributions:**

- **Data Loading & EDA Functions**
  - `load_data()`: Loads CSV and validates structure
  - `exploratory_analysis()`: Generates summary statistics
  - `plot_eda_visualizations()`: Creates multi-panel EDA plots
  - `plot_correlation_heatmap()`: Visualizes feature relationships

- **Preprocessing Pipeline**
  - `preprocess_data()`: Complete pipeline orchestration
    - Feature engineering (one-hot encoding)
    - Train/test stratified split
    - StandardScaler fitting and transformation
    - SMOTE augmentation (training data only)
    - Returns: preprocessed data, scaler, and SMOTE object
  - **Critical:** Ensures consistent preprocessing across training and inference

- **Binary Classification**
  - `train_naive_bayes_binary()`: Trains Gaussian NB
  - `train_decision_tree_binary()`: Trains Decision Tree with tuned hyperparameters
  - `evaluate_binary_model()`: Computes all metrics (accuracy, precision, recall, F1, ROC-AUC)
  - `plot_confusion_matrices()`: Side-by-side model comparison
  - `plot_roc_curves()`: ROC curve visualization
  - `plot_feature_importance_dt()`: Feature importance ranking
  - `create_comparison_table()`: Summary metrics table

- **Multi-Label & Multi-Class Training**
  - `prepare_multiclass_target()`: Creates multi-class labels with priority encoding
  - `train_multilabel_models()`: Multi-output wrapper for both models
  - `train_multiclass_models()`: Multi-class training (6 failure types)
  - `evaluate_multilabel_models()`: Hamming loss, subset accuracy
  - `evaluate_multiclass_models()`: Weighted and macro F1 scores

- **Model Persistence**
  - `save_model()`: Saves models using joblib
  - `load_model()`: Loads saved models for inference
  - Enables reproducibility and model versioning

- **Utility Functions**
  - `create_comparison_table()`: Pandas DataFrame for metrics comparison

**Why This Module Matters:**
- **Modular design:** Each function has a single responsibility
- **Reusability:** Functions can be imported and used in other scripts (e.g., `generate_metrics_fixed.py`)
- **Documentation:** Every function has docstrings explaining inputs, outputs, and purpose
- **Best practices:** Follows scikit-learn conventions and proper ML workflow

---

## How These Files Support the Gradio Project

### 1. **Model Training & Evaluation** ✓
- `predictive_maintenance.py` provides the foundation for training and evaluation
- `Predictive_Maintenance_Project.ipynb` demonstrates the full workflow
- **Result:** Production-ready `.joblib` files used in `router.py`

### 2. **Data Preprocessing Pipeline** ✓
- Establishes StandardScaler fitting on training data
- Implements SMOTE for class imbalance
- **Used by:** `pipeline.py` in the Gradio app (replicates preprocessing steps)

### 3. **Feature Engineering** ✓
- Defines 8 core features and 5 engineered features (13 total)
- One-hot encoding for machine types
- **Used by:** `app.py` input sliders (sensor ranges, types)

### 4. **Model Selection & Threshold Tuning** ✓
- Justifies Decision Tree over Naive Bayes (interpretability, performance)
- Establishes thresholds: 0.75 (safety) vs 0.96 (cost-optimized)
- **Used by:** `router.py` for model routing and configuration

### 5. **Performance Metrics** ✓
- Calculates precision, recall, F1 at multiple thresholds
- Cross-validation for generalization estimates
- **Used by:** `app.py` Model Statistics tab (displays metrics)

### 6. **Visualization & Validation** ✓
- EDA plots justify feature selection
- ROC curves and confusion matrices validate model choices
- Feature importance shows interpretability
- **Purpose:** Support decision-making and documentation

---

## Workflow: From Notebook to Production

```
┌─ Predictive_Maintenance_Project.ipynb ─┐
│                                         │
├─ Load Predictive_M.csv                │
├─ Exploratory Data Analysis             │
├─ Feature Engineering & Preprocessing   │
├─ Train Binary Models (NB vs DT)       │
├─ Train Multi-Label/Multi-Class Models │
├─ Evaluate & Visualize                 │
│                                         │
└─ Save: Trained_models/*.joblib        │
        scaler.joblib                    │
        metrics.json                     │
        visualizations.png               │
└──────────────────────────────────────┘
            ↓
      [predictive_maintenance.py]
      Reusable module functions
            ↓
┌──────────────────────────────────┐
│ generate_metrics_fixed.py         │
│ (Reproduces metrics at scale)     │
└──────────────────────────────────┘
            ↓
┌──────────────────────────────────┐
│ Gradio App (app.py)              │
│ + pipeline.py (preprocessing)    │
│ + router.py (prediction logic)   │
│                                   │
│ Uses:                            │
│ - Trained models (.joblib)       │
│ - Scaler.joblib                  │
│ - Metrics.json                   │
│ - Feature engineering logic      │
└──────────────────────────────────┘
```

---

## Key Functions Used in Gradio

### From `predictive_maintenance.py`:

| Function | Gradio Usage | Location |
|----------|--------------|----------|
| `preprocess_data()` | Feature engineering logic | `pipeline.py` |
| `evaluate_binary_model()` | Metrics calculation framework | `generate_metrics_fixed.py` |
| Feature engineering logic | One-hot encoding, derived features | `pipeline.py` preprocessing |
| Scaler/SMOTE patterns | Training/inference consistency | `router.py` model loading |

### From `Predictive_Maintenance_Project.ipynb`:

| Task | Gradio Usage | Location |
|------|--------------|----------|
| Model selection (DT > NB) | Justifies `MODEL_MAP` config | `router.py` |
| Threshold tuning (0.75, 0.96) | Supports business priorities | `app.py` radio button options |
| Feature importance | Explains model decisions | `app.py` prediction output |
| Performance metrics | Displays in Model Statistics tab | `app.py` metrics rendering |
| Data validation | Input range checks | `app.py` input sliders |

---

## How to Use These Files

### Run the Notebook:
```bash
jupyter notebook Predictive_Maintenance_Project.ipynb
```
- Execute cells sequentially to understand the full pipeline
- Modify hyperparameters or feature engineering to experiment
- Regenerate models if needed

### Use the Module:
```python
from predictive_maintenance import (
    load_data, 
    exploratory_analysis, 
    preprocess_data,
    train_decision_tree_binary,
    evaluate_binary_model
)

# Load and explore
df = load_data("Predictive_M.csv")
stats = exploratory_analysis(df)

# Preprocess
prep_data = preprocess_data(df)

# Train
model = train_decision_tree_binary(prep_data['X_train'], prep_data['y_train'])

# Evaluate
metrics = evaluate_binary_model(model, prep_data['X_train'], prep_data['y_train'], 
                                prep_data['X_test'], prep_data['y_test'], 
                                model_name="Decision Tree")
```

### Regenerate Models:
```bash
python generate_metrics_fixed.py
```
- Retrains models with fresh train/test splits
- Performs 5-fold cross-validation
- Saves updated metrics to `model_metrics_fixed.json`

---

## Key Learnings & Decisions

### 1. **Why Decision Tree over Naive Bayes?**
- **Interpretability:** DT provides clear decision rules; NB outputs only probabilities
- **Non-linear relationships:** Sensor data exhibits complex patterns DT captures naturally
- **Feature interactions:** DT models interactions (e.g., temperature × speed); NB assumes independence
- **Class imbalance:** DT + SMOTE handles 3.4% failure rate better than NB

### 2. **Why SMOTE Before Training?**
- Balances minority class without information loss
- Applied ONLY to training data (validation/test remain original)
- Prevents data leakage and realistic performance estimates

### 3. **Why Multiple Thresholds (0.75 vs 0.96)?**
- **0.75 threshold:** Safety-first mode (high recall, catch failures)
- **0.96 threshold:** Cost-optimized mode (high precision, minimize false alarms)
- Allows operational flexibility based on business priority

### 4. **Why Feature Engineering?**
- Original 8 features insufficient for strong signal
- Engineered 5 derived features (Temp_Diff, Power_Estimate, etc.)
- Captures domain knowledge: physical relationships in equipment

---

## Outputs Generated

| File | Purpose | Used In |
|------|---------|---------|
| `binary_decision_tree_*.joblib` | Main failure prediction model | `router.py` |
| `multilabel_decision_tree_*.joblib` | Concurrent failure detection | `router.py` (optional) |
| `scaler.joblib` | Fitted StandardScaler | `pipeline.py` preprocessing |
| `model_metrics_fixed.json` | Performance metrics at thresholds | `app.py` Model Statistics tab |
| `.png` visualizations | EDA and model analysis plots | Documentation, presentations |

---

## Conclusion

These two files represent the **research and development phase** that established the solid foundation for the production Gradio application:

- **`Predictive_Maintenance_Project.ipynb`** is the **experimental lab** where all hypotheses are tested and decisions are validated
- **`predictive_maintenance.py`** is the **reusable toolkit** that ensures consistency across training, evaluation, and inference

Together, they ensure that the Gradio app is built on evidence-based decisions, well-validated models, and reproducible pipelines.

---

**Author:** Najam iqbal and M. Fawaz Asif
