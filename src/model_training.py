import numpy as np
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)

DATA_DIR = "processed_data"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

print("="*60)
print("Loading Data")
print("="*60)

X_train = np.load(f"{DATA_DIR}/X_train.npy")
X_test = np.load(f"{DATA_DIR}/X_test.npy")
y_train = np.load(f"{DATA_DIR}/y_train.npy")
y_test = np.load(f"{DATA_DIR}/y_test.npy")

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Training - Real: {np.sum(y_train==0)}, Fraudulent: {np.sum(y_train==1)}")
print(f"Test - Real: {np.sum(y_test==0)}, Fraudulent: {np.sum(y_test==1)}")

print("\n" + "="*60)
print("Training Logistic Regression")
print("="*60)

# Use StratifiedKFold for better cross-validation with imbalanced data
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["liblinear", "lbfgs"],
    "class_weight": ["balanced"]  # Add class_weight to handle imbalance
}

lr = LogisticRegression(max_iter=2000, random_state=42)

grid_lr = GridSearchCV(
    lr,
    param_grid,
    scoring="f1",  # Use F1 score for imbalanced data
    cv=skf,
    n_jobs=-1,
    verbose=1
)

grid_lr.fit(X_train, y_train)
best_lr = grid_lr.best_estimator_

joblib.dump(best_lr, f"{MODEL_DIR}/logistic_regression_v1.pkl")

print(f"Best Logistic Regression Params: {grid_lr.best_params_}")
print(f"Best CV F1 Score: {grid_lr.best_score_:.4f}")

# Evaluate on test set
y_pred_lr = best_lr.predict(X_test)
y_prob_lr = best_lr.predict_proba(X_test)[:, 1]

print("\nLogistic Regression Test Performance:")
print(f"  Accuracy:  {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_lr, zero_division=0):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred_lr, zero_division=0):.4f}")
print(f"  F1 Score:  {f1_score(y_test, y_pred_lr, zero_division=0):.4f}")
try:
    print(f"  ROC-AUC:   {roc_auc_score(y_test, y_prob_lr):.4f}")
except ValueError:
    pass
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_lr)}")

print("\n" + "="*60)
print("Training Random Forest")
print("="*60)

# Improved Random Forest with better hyperparameters
rf = RandomForestClassifier(
    n_estimators=300,  # More trees
    max_depth=15,  # Reduced to prevent overfitting
    min_samples_split=5,  # Increased to handle small dataset
    min_samples_leaf=2,  # Increased to handle small dataset
    max_features='sqrt',  # Better for text features
    random_state=42,
    class_weight="balanced",  # Handle class imbalance
    n_jobs=-1
)

rf.fit(X_train, y_train)

joblib.dump(rf, f"{MODEL_DIR}/random_forest_v1.pkl")

print("Random Forest trained successfully")

# Evaluate on test set
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

print("\nRandom Forest Test Performance:")
print(f"  Accuracy:  {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_rf, zero_division=0):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred_rf, zero_division=0):.4f}")
print(f"  F1 Score:  {f1_score(y_test, y_pred_rf, zero_division=0):.4f}")
try:
    print(f"  ROC-AUC:   {roc_auc_score(y_test, y_prob_rf):.4f}")
except ValueError:
    pass
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_rf)}")

print("\n" + "="*60)
print("Model Training Complete!")
print("="*60)
