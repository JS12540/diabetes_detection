# 🩺 Diabetes Detection — End-to-End ML Pipeline with MLflow

An end-to-end machine learning workflow for diabetes prediction using feature engineering, RFE-based feature selection, KMeans-SMOTE oversampling, XGBoost modeling, optimal thresholding with Youden's J statistic, and comprehensive experiment tracking via MLflow with Dagshub backend.

**📍 MLflow Tracking URI:**  
[https://dagshub.com/jayshah0726/diabetes_detection.mlflow](https://dagshub.com/jayshah0726/diabetes_detection.mlflow)

---

## 🚀 Project Overview

This project implements a complete, production-ready ML experimentation pipeline featuring:

- **Pima Indians Diabetes Dataset**
- **Advanced Feature Engineering**
- **Feature Selection with RFE** (Logistic Regression)
- **KMeans-SMOTE** for balanced training
- **XGBoost** model training
- **Optimal Threshold Calculation** (Youden's J statistic)
- **Automated Cross-Validation**
- **Full MLflow Tracking** (parameters, metrics, artifacts, models)

The objective is to create a robust, reproducible, and transparent workflow for binary classification in healthcare settings.

---

## 📂 Project Structure
```
.
├── Disease Detection.ipynb   # Main end-to-end notebook
├── README.md                                 # Project documentation
└── mlruns / dagshub                           # (Remote) MLflow tracking
```

---

## 📊 Dataset

We use the **Pima Indians Diabetes Dataset**, loaded directly from the public repository.

**Dataset Variables:**
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Age
- Diabetes Pedigree Function
- **Target:** Outcome (0 = Non-diabetic, 1 = Diabetic)

---

## 🔧 Feature Engineering

Several domain-driven interaction and ratio features are created to enhance model performance:

### ✨ Interaction Features
- `BMI_SkinThickness`
- `Age_DPF`
- `BMI_Age`
- `Glucose_BMI`

### ✨ Ratio Features
- `Preg_Age_Ratio`
- `SkinThickness_BMI_Ratio`
- `Glucose_Age_Ratio`

These features help uncover non-linear relationships missed by base features.

---

## 🧬 Feature Selection (RFE with Logistic Regression)

To eliminate irrelevant or redundant features, **Recursive Feature Elimination (RFE)** is applied using a Logistic Regression estimator.

**Steps:**
1. Scale all features
2. Apply RFE to select the top N features
3. Rescale only selected features
4. Feed into oversampling + modeling pipeline

This reduces noise and improves generalization.

---

## ⚖️ Handling Class Imbalance — KMeans SMOTE

The dataset exhibits class imbalance (more non-diabetic cases). We use **KMeans-SMOTE** (from `imblearn`) to oversample the minority class intelligently by clustering first, then applying SMOTE.

**Benefits:**
- ✔ Prevents overfitting
- ✔ Handles multi-modal minority distributions
- ✔ Works well with RFE-selected features

---

## 🔥 Model Training — XGBoost

We train an **XGBClassifier** with carefully tuned hyperparameters:

### Hyperparameter Configuration
```python
xgb_params = { 
    "objective": "binary:logistic", 
    "eval_metric": ["aucpr", "auc", "logloss"], 
    "use_label_encoder": False, 
    "tree_method": "hist", 
    "max_depth": 16, 
    "max_leaves": 2048, 
    "grow_policy": "lossguide", 
    "n_estimators": 6000, 
    "learning_rate": 0.035, 
    "subsample": 1.0, 
    "colsample_bytree": 0.7, 
    "colsample_bylevel": 0.7, 
    "colsample_bynode": 0.7, 
    "min_child_weight": 10, 
    "gamma": 1.0, 
    "reg_alpha": 0.0, 
    "reg_lambda": 20.0, 
    "random_state": 42, 
    "n_jobs": -1, 
}
```

### Key Parameters Explained
- **Tree Method:** `hist` - Histogram-based algorithm for faster training
- **Growth Policy:** `lossguide` - Splits nodes with highest loss reduction
- **Max Depth:** 16 - Maximum tree depth
- **Max Leaves:** 2048 - Maximum number of leaf nodes
- **Learning Rate:** 0.035 - Step size shrinkage to prevent overfitting
- **Regularization:** L2 (lambda=20.0) for model generalization
- **Feature Sampling:** 70% column sampling at tree, level, and node

### Cross-Validation Results

The model is trained across **5-fold cross-validation**, with each fold logging:
- Metrics
- ROC Curve
- Optimal threshold
- Feature set
- Model artifact

**Performance Metrics:**
- **Mean CV ROC-AUC:** 0.8303
- **Best Fold:** Fold 1
- **Best ROC-AUC:** 0.8659
- **Selected Features:** 12 features (via RFE)
- **Model Version:** 9 (registered in MLflow)

All runs are captured in Dagshub's MLflow server.

---

## 🎯 Optimal Threshold — Youden's J Statistic

Instead of using the default 0.5 threshold, we compute the **optimal decision threshold** for each fold:

**J = Sensitivity + Specificity - 1**

**Benefits:**
- ✔ Increases sensitivity for medical applications
- ✔ Reduces false negatives
- ✔ Balances precision & recall effectively

---

## 📈 MLflow Tracking (Dagshub Integration)

All experiments automatically log the following:

### Parameters
- All XGBoost hyperparameters
- RFE feature count
- Random seed values
- Selected features per fold
- Threshold values

### Metrics
- AUC
- AUC-PR
- Accuracy
- Precision
- Recall
- F1
- Optimal J statistic

### Artifacts
- ROC curve plots
- Feature importance plots
- Final trained models
- Preprocessing scalers

### Model Registry
- **Model Name:** `best_model_xgb_diabetes`
- **Latest Version:** 9
- **Registered Features:** 12 selected features with inferred signature
- **Best Model Selection:** Based on highest ROC-AUC across CV folds

**MLflow Backend:**  
Tracking server: [https://dagshub.com/jayshah0726/diabetes_detection.mlflow](https://dagshub.com/jayshah0726/diabetes_detection.mlflow)

**Dagshub Features:**
- Remote experiment tracking
- Versioning of datasets/models
- Cloud-hosted MLflow UI
- Model registry for production deployment

---

## 🔄 Pipeline Flow Diagram
```
      Raw Diabetes Dataset
                │
                ▼
      Feature Engineering
                │
                ▼
     Train/Test Split → (Test stored)
                │
                ▼
  For Each CV Fold (5-fold)
      ├─ Scale Features
      ├─ RFE Feature Selection
      ├─ KMeans-SMOTE Oversampling
      ├─ Train XGBoost
      ├─ Compute Optimal Threshold (Youden J)
      ├─ Log Everything to MLflow
                ▼
        Collect Metrics & Results
                ▼
      Select Best Model (ROC-AUC)
                ▼
      Register to MLflow Model Registry
```

---

## 🏆 Model Performance Summary

| Metric | Value |
|--------|-------|
| Mean CV ROC-AUC | 0.8303 |
| Best Fold ROC-AUC | 0.8659 |
| Best Fold | Fold 1 |
| Selected Features | 12 |
| Model Version | 9 |

---

## 🏁 Final Deliverables

- ✔ Cleaned, engineered dataset
- ✔ Selected feature subsets (12 optimal features)
- ✔ Multiple XGBoost models across CV folds
- ✔ Best model registered in MLflow (v9)
- ✔ Optimal decision thresholds
- ✔ All experiments logged in MLflow
- ✔ Ready-to-extend experimentation framework

---

## 🧠 Future Improvements

Planned enhancements for this project:

- **SHAP explainability** for model interpretability
- **Hyperparameter tuning with Optuna** for automated optimization
- **FastAPI model inference server** for production deployment
- **Dockerized MLflow experiment runner** for reproducibility
- **Feature store integration** for feature management
- **A/B testing framework** for model comparison
- **Real-time prediction API** with model monitoring

---

## 💡 Author

**Jay Shah**  
ML Engineer | AI Practitioner  

**MLflow Repo:** [https://dagshub.com/jayshah0726/diabetes_detection.mlflow](https://dagshub.com/jayshah0726/diabetes_detection.mlflow)

---

## ⭐ Show your support

Give a ⭐️ if this project helped you!
