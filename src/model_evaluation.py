import numpy as np
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

DATA_DIR = "processed_data"
MODEL_DIR = "models"

print("="*60)
print("Model Evaluation")
print("="*60)

X_test = np.load(f"{DATA_DIR}/X_test.npy")
y_test = np.load(f"{DATA_DIR}/y_test.npy")
X_train = np.load(f"{DATA_DIR}/X_train.npy")
y_train = np.load(f"{DATA_DIR}/y_train.npy")

print(f"\nTest set size: {len(y_test)}")
print(f"Class distribution - Real: {np.sum(y_test==0)}, Fraudulent: {np.sum(y_test==1)}")

models = {}
try:
    models["Logistic Regression"] = joblib.load(f"{MODEL_DIR}/logistic_regression_v1.pkl")
except FileNotFoundError:
    print(f"Warning: {MODEL_DIR}/logistic_regression_v1.pkl not found")

try:
    models["Random Forest"] = joblib.load(f"{MODEL_DIR}/random_forest_v1.pkl")
except FileNotFoundError:
    print(f"Warning: {MODEL_DIR}/random_forest_v1.pkl not found")

if not models:
    print("No models found to evaluate!")
    exit(1)

results = []

print("\n" + "="*60)
print("Test Set Performance")
print("="*60)

for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate all metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        roc_auc = 0.0

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc
    })

    print(f"\nMetrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")

    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print("                 Predicted")
    print("              Real  Fraudulent")
    print(f"Actual Real      {cm[0,0]:4d}      {cm[0,1]:4d}")
    print(f"       Fraudulent {cm[1,0]:4d}      {cm[1,1]:4d}")
    
    print(f"\nClassification Report:")
    print(classification_report(
        y_test, 
        y_pred, 
        target_names=['Real', 'Fraudulent'],
        zero_division=0
    ))

df_results = pd.DataFrame(results)
print("\n" + "="*60)
print("Model Comparison Summary")
print("="*60)
print(df_results.to_string(index=False))

# Cross-validation for more robust evaluation
print("\n" + "="*60)
print("Cross-Validation Performance")
print("="*60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n{name}:")
    
    # Cross-validate with multiple metrics
    cv_f1 = cross_val_score(model, X_train, y_train, cv=skf, scoring="f1")
    cv_precision = cross_val_score(model, X_train, y_train, cv=skf, scoring="precision")
    cv_recall = cross_val_score(model, X_train, y_train, cv=skf, scoring="recall")
    cv_accuracy = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")
    
    print(f"  F1 Score:     {cv_f1.mean():.4f} (+/- {cv_f1.std()*2:.4f})")
    print(f"  Precision:    {cv_precision.mean():.4f} (+/- {cv_precision.std()*2:.4f})")
    print(f"  Recall:       {cv_recall.mean():.4f} (+/- {cv_recall.std()*2:.4f})")
    print(f"  Accuracy:     {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std()*2:.4f})")

print("\n" + "="*60)
print("Evaluation Complete!")
print("="*60)

