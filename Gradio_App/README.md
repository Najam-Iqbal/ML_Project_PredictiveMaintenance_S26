# 🔧 Predictive Maintenance - ML Lab Project

A **Decision Tree-based machine failure prediction system** deployed as an interactive web application using Gradio and Hugging Face Spaces.

## Gradio App: Detail and Working Mechanism

The Gradio app is the interactive front end of this project. It is built in [app.py](app.py) using `gr.Blocks` and is designed to let users test the predictive maintenance model without running any ML code manually.

### What the app provides

- A **Manual Input** tab for single-machine prediction.
- A **Batch CSV Upload** tab for multiple rows at once.
- A **Model Statistics** tab that displays saved precision, recall, and F1 metrics.
- A polished UI with custom styling, clear status cards, and downloadable outputs.

### How it works

1. The app loads the trained model pipeline and saved metric JSON at startup.
2. In **Manual Input**, the user enters sensor values such as air temperature, process temperature, RPM, torque, tool wear, and machine type.
3. The input values are converted into a feature-ready row using the preprocessing pipeline.
4. The selected business priority is passed to the routing logic, which chooses the appropriate failure-detection model and threshold.
5. The app returns a formatted result card showing whether failure is predicted, the confidence score, the selected model, and the reason.
6. In **Batch CSV Upload**, the user uploads a CSV file, the app preprocesses every row, runs predictions for each sample, and adds output columns such as predicted failure, reason, and confidence.
7. The processed batch results are shown as a preview table and also saved as a downloadable CSV file.
8. The **Model Statistics** tab reads `model_metrics.json` and displays test-set and 5-fold cross-validation results for comparison.

### Working mechanism in simple terms

The app does not train models during prediction. It only receives user input, preprocesses it, sends it to the routing and prediction functions, and then formats the output for display. This keeps the interface fast, interactive, and suitable for deployment on Hugging Face Spaces.

## 📋 Project Overview

This project predicts machine equipment failures based on sensor readings from a manufacturing dataset containing:
- **10,000 equipment samples**
- **5 sensor features** (Air temperature, Process temperature, Rotational speed, Torque, Tool wear)
- **Product Type** classification (Low, Medium, High quality)
- **Target:** Machine failure prediction (Binary classification: Failure/No Failure)

**Failure Rate:** 3.4% (highly imbalanced dataset - solved using SMOTE)

## 🎯 Key Achievements

✅ **Binary Classification Model:** Decision Tree with feature engineering
✅ **Class Imbalance Handling:** SMOTE oversampling technique  
✅ **Performance Optimization:** Threshold tuning (0.96 for F1-score maximization)  
✅ **Interactive Web Interface:** Gradio app with real-time predictions  
✅ **Cloud Deployment:** Ready for Hugging Face Spaces

## 🚀 Quick Start

### Local Deployment

1. **Clone or download the project**
   ```bash
   cd Project_ML
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   python app.py
   ```

4. **Open in browser**
   - Navigate to `http://localhost:7860`

### Hugging Face Spaces Deployment

1. **Create a new Space on Hugging Face**
   - Go to https://huggingface.co/spaces/new
   - Select "Gradio" as the Space SDK
   - Choose your namespace and space name

2. **Upload project files**
   - Upload `app.py`
   - Upload `requirements.txt`
   - Upload the `Trained_models/` folder
   - Optional: Upload `scaler.joblib` and `SMOTE object` for preprocessing

3. **Configure the app**
   - The space should automatically detect `app.py` as the Gradio interface
   - Click "Run" to deploy

4. **Access your app**
   - Your Gradio app will be available at `https://huggingface.co/spaces/[username]/[space-name]`

## 📊 Input Features

| Feature | Range | Unit | Description |
|---------|-------|------|-------------|
| Air Temperature | 295-305 | Kelvin | Ambient temperature in manufacturing environment |
| Process Temperature | 305-317.5 | Kelvin | Operating temperature of the equipment |
| Rotational Speed | 1168-2886 | RPM | Machine spindle rotation speed |
| Torque | 3.8-76 | Newton-meter | Mechanical force applied |
| Tool Wear | 0-253 | Minutes | Cumulative wear on cutting tool |
| Product Type | L, M, H | Category | Quality grade (Low, Medium, High) |

## 🧠 Model Information

**Algorithm:** Decision Tree Classifier
- **Max Depth:** 10
- **Features:** 8 (5 numeric + 3 one-hot encoded)
- **Training Data:** 8,000 samples (after 80/20 split)
- **Preprocessing Pipeline:**
  1. StandardScaler normalization
  2. SMOTE for class balance (1:1 ratio)
  3. Model training with stratified cross-validation

**Best Model Selected:** Feature Engineered Decision Tree (13 features, threshold=0.96)

## 📁 Project Structure

```
Root/
├── app.py                          # Gradio web application
├── requirements.txt                # Python dependencies
├── Predictive_Maintenance_Project.ipynb  # Full analysis notebook
├── predictive_maintenance.py       # Model utilities & functions
├── Predictive_M.csv               # Dataset (10,000 samples)
├── Trained_models/
│   ├── binary_decision_tree_feature_engineered_13features_threshold_0p96.joblib
│   └── multilabel_decision_tree_multioutput_scaled_original_features.joblib
└── Documentation/
    └── For_Submission/
```

## 🔧 Configuration

### To use a different model:
Edit `app.py` line ~40:
```python
MODEL_PATH = model_dir / "your_model_name.joblib"
```

### To adjust input ranges:
Modify the Slider components in `create_gradio_interface()` function (lines ~180-220)

## 📊 Example Predictions

### Scenario 1: Normal Operation ✅
- Air Temp: 298 K
- Process Temp: 310 K
- RPM: 1500
- Torque: 20 Nm
- Tool Wear: 50 min
- Type: L
→ **Result:** No failure predicted

### Scenario 2: High Stress ⚠️
- Air Temp: 304 K
- Process Temp: 316 K
- RPM: 2500
- Torque: 70 Nm
- Tool Wear: 200 min
- Type: H
→ **Result:** Failure predicted (Schedule maintenance)

## 📈 Model Performance

The trained models achieve:
- **Accuracy:** ~96-98%
- **Precision:** ~85-90%
- **Recall:** ~85-95%
- **F1-Score:** ~0.85-0.92
- **ROC-AUC:** ~0.95

(Metrics vary by model variant - see notebook for detailed comparison)

## ⚙️ Technical Stack

| Technology | Purpose |
|-----------|---------|
| Python 3.9+ | Programming language |
| Gradio 4.32+ | Web interface framework |
| Scikit-learn | Machine learning models |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Imbalanced-learn | SMOTE implementation |
| Joblib | Model serialization |

## 🎓 Educational Context

**Course:** CS058 - Machine Learning Laboratory  
**Semester:** #06  
**University:** [Your University]  
**Project Type:** Supervised Learning - Binary Classification

This project demonstrates:
- End-to-end ML pipeline development
- Class imbalance handling techniques
- Model evaluation & optimization
- Web application deployment
- Cloud infrastructure utilization

## 🚨 Disclaimer

⚠️ **Educational Use Only**

This model is developed for educational purposes as part of an ML laboratory course. Real-world maintenance decisions should:
- Consider multiple data sources
- Involve domain experts
- Include additional safety margins
- Be validated against actual failure data
- Follow industry safety standards

**Use at your own risk.** The developers assume no liability for decisions made based on this model's predictions.

## 📝 License

This project is part of an academic course. Distribution and modification are allowed with appropriate attribution.

## 👨‍💻 Author
Najam Iqbal
Muhammad Fawaz Asif

## 🤝 Contributing

For improvements or suggestions:
1. Check the current model performance
2. Propose changes with data backing
3. Test thoroughly before deployment

**Happy Predictions! 🎯**
