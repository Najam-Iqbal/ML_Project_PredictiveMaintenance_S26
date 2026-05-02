"""
pipeline.py — Data Preprocessing Pipeline
==========================================
Handles all data preparation: loading, feature engineering, scaling.
Independent of Gradio (pure sklearn/pandas).
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# ── Load scaler ──────────────────────────────────────────────────────
_BASE = Path(__file__).parent
SCALER_PATH = _BASE / "scaler.joblib"

try:
    SCALER = joblib.load(str(SCALER_PATH))
except FileNotFoundError:
    print(f"WARNING: scaler.joblib not found at {SCALER_PATH}")
    SCALER = None


# ── Feature engineering ──────────────────────────────────────────────
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


# ── Main preprocessing function ──────────────────────────────────────
def preprocess(df):
    """
    Full preprocessing pipeline:
      1. One-hot encode Type
      2. Engineer 5 derived features
      3. Select 13 feature columns
      4. Scale
    
    Returns: scaled feature matrix (numpy array)
    """
    df = df.copy()
    
    # Ensure Type column exists
    if 'Type' not in df.columns:
        raise ValueError("Missing 'Type' column (expected L, M, or H)")
    
    # One-hot encode Type
    df = pd.get_dummies(df, columns=['Type'], prefix='Type', drop_first=False)
    
    # Ensure all three Type columns exist (even if not present in data)
    for type_col in ['Type_H', 'Type_L', 'Type_M']:
        if type_col not in df.columns:
            df[type_col] = 0
    
    # Engineer features
    df = engineer_features(df)
    
    # Select all 13 features in correct order
    feature_cols = [
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
    
    X = df[feature_cols]
    
    # Scale
    if SCALER is not None:
        X_scaled = SCALER.transform(X)
    else:
        # Fallback: use raw features (not recommended)
        X_scaled = X.values
    
    return X_scaled


# ── Helper: build single-row input ──────────────────────────────────
def build_single_input(
    air_temp, process_temp, rpm, torque, tool_wear, machine_type
):
    """
    Create a 1-row DataFrame from manual input values.
    """
    return pd.DataFrame({
        'Air temperature [K]': [float(air_temp)],
        'Process temperature [K]': [float(process_temp)],
        'Rotational speed [rpm]': [float(rpm)],
        'Torque [Nm]': [float(torque)],
        'Tool wear [min]': [float(tool_wear)],
        'Type': [str(machine_type)],
    })
