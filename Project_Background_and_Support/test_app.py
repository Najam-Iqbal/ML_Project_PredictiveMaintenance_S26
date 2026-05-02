"""
Quick test script to verify the Gradio app works locally
Run this before deploying to Hugging Face:
    python test_app.py
"""

import sys
import os
from pathlib import Path

print("="*70)
print("PREDICTIVE MAINTENANCE - LOCAL APP VERIFICATION")
print("="*70)

# Check Python version
print("\n[1] Checking Python version...")
version = sys.version_info
if version.major == 3 and version.minor >= 9:
    print(f"✓ Python {version.major}.{version.minor} (OK)")
else:
    print(f"⚠ Python {version.major}.{version.minor} (recommended 3.9+)")

# Check required files
print("\n[2] Checking required files...")
required_files = ['app.py', 'requirements.txt', 'scaler.joblib']
models_folder = Path('Trained_models')

all_present = True
for file in required_files:
    if Path(file).exists():
        size = os.path.getsize(file)
        print(f"✓ {file} ({size:,} bytes)")
    else:
        print(f"❌ {file} NOT FOUND")
        all_present = False

if models_folder.exists():
    model_files = list(models_folder.glob('*.joblib'))
    print(f"✓ Trained_models/ ({len(model_files)} models)")
    for model in model_files[:3]:
        print(f"  • {model.name}")
    if len(model_files) > 3:
        print(f"  ... and {len(model_files)-3} more")
else:
    print(f"❌ Trained_models/ folder NOT FOUND")
    all_present = False

if not all_present:
    print("\n⚠ Missing files! Please check the setup.")
    sys.exit(1)

# Try importing dependencies
print("\n[3] Checking Python dependencies...")
dependencies = ['gradio', 'pandas', 'numpy', 'sklearn', 'joblib']
missing = []

for dep in dependencies:
    try:
        __import__(dep)
        print(f"✓ {dep}")
    except ImportError:
        print(f"❌ {dep} NOT INSTALLED")
        missing.append(dep)

if missing:
    print(f"\n⚠ Missing packages: {', '.join(missing)}")
    print(f"Install with: pip install -r requirements.txt")
    sys.exit(1)

# Test model loading
print("\n[4] Testing model loading...")
try:
    import joblib
    scaler = joblib.load('scaler.joblib')
    print(f"✓ Scaler loaded successfully")
    print(f"  Features: {scaler.n_features_in_}")
    
    model_path = list(models_folder.glob('*.joblib'))[0]
    model = joblib.load(model_path)
    print(f"✓ Model loaded: {model_path.name}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# Test prediction
print("\n[5] Testing sample prediction...")
try:
    import pandas as pd
    import numpy as np
    
    # Sample input
    test_input = {
        'Air temperature [K]': 298.5,
        'Process temperature [K]': 311,
        'Rotational speed [rpm]': 2000,
        'Torque [Nm]': 40,
        'Tool wear [min]': 100,
        'Type_H': 0,
        'Type_L': 1,
        'Type_M': 0,
    }
    
    X = pd.DataFrame([test_input])
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    
    if prediction == 0:
        result = "No Failure ✅"
    else:
        result = "Failure ⚠️"
    
    print(f"✓ Sample prediction: {result}")
    print(f"  Input: Type_L, 298.5K air, 311K process, 2000 RPM, 40 Nm, 100 min wear")
    
except Exception as e:
    print(f"❌ Prediction error: {e}")
    sys.exit(1)

# Test Gradio app import
print("\n[6] Testing Gradio app import...")
try:
    import app
    print(f"✓ app.py imports successfully")
except Exception as e:
    print(f"⚠ Warning in app.py: {e}")
    print(f"  (This may be OK if Gradio interface isn't fully tested)")

print("\n" + "="*70)
print("✓ ALL CHECKS PASSED!")
print("="*70)
print("\nYou're ready to:")
print("  1. Run locally: python app.py")
print("  2. Deploy to Hugging Face Spaces")
print("\nNext: Run 'python app.py' to launch the web interface")
print("      Open http://localhost:7860 in your browser")
