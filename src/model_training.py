"""
model_training.py — Phase 3/4: Advanced Model Training & Evaluation

Trains two models as required by Track B:
  1. Baseline: Logistic Regression
  2. Main Model: CatBoost with Optuna Bayesian Hyperparameter Optimization
Compares both, saves the best to 'models/best_model.pkl' and 'models/best_params.json'.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import optuna
import logging

# Mute optuna logs slightly
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Add project root to path so we can import from src
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from src.data_loader import preprocess_for_modeling, CATEGORICAL_COLS

class ManualVotingClassifier(BaseEstimator, ClassifierMixin):
    """Custom voting classifier to bypass sklearn cloning issues with CatBoost."""
    def __init__(self, cat_model, xgb_model, lgb_model, cat_features):
        self.cat_model = cat_model
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        self.cat_features = cat_features
        self.classes_ = np.array([0, 1])
        
    def fit(self, X, y):
        self.cat_model.fit(X, y, cat_features=self.cat_features)
        self.xgb_model.fit(X, y)
        self.lgb_model.fit(X, y)
        return self
        
    def predict_proba(self, X):
        p1 = self.cat_model.predict_proba(X)
        p2 = self.xgb_model.predict_proba(X)
        p3 = self.lgb_model.predict_proba(X)
        return (p1 + p2 + p3) / 3.0
        
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
        
    def get_all_params(self):
        return "Manual Soft Voting Ensemble"

def print_metrics(model_name: str, y_true, y_pred) -> dict:
    """Print accuracy, precision, recall, F1, confusion matrix, and classification report."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"\n--- {model_name} Metrics ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}\n")
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def run_pipeline():
    print("=" * 60)
    print("BankMind Model Training (Logistic Regression vs CatBoost)")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Feature Selection & Preparation
    # ---------------------------------------------------------
    data_label = preprocess_for_modeling(encoding="label", scale=False)
    X_full_train, X_test = data_label["X_train"], data_label["X_test"]
    y_full_train, y_test = data_label["y_train"], data_label["y_test"]
    
    # Drop noisy features
    noisy_features = ["day"]
    cols_to_drop = [c for c in noisy_features if c in X_full_train.columns]
    if cols_to_drop:
        print(f"\n[1] Dropping Noisy Features: {cols_to_drop}")
        X_full_train = X_full_train.drop(columns=cols_to_drop)
        X_test = X_test.drop(columns=cols_to_drop)

    # ---------------------------------------------------------
    # 2. BASELINE: Logistic Regression (required by Track B)
    # ---------------------------------------------------------
    # Logistic Regression needs scaled features to converge properly
    print("\n[2] Training Baseline: Logistic Regression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_full_train)
    X_test_scaled = scaler.transform(X_test)

    lr_model = LogisticRegression(
        max_iter=1000,           # ensure convergence
        class_weight='balanced', # handle class imbalance
        random_state=42,
        solver='lbfgs'
    )
    lr_model.fit(X_train_scaled, y_full_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    lr_metrics = print_metrics("Logistic Regression (Baseline)", y_test, y_pred_lr)

    # ---------------------------------------------------------
    # 3. MAIN MODEL: CatBoost with Optuna tuning
    # ---------------------------------------------------------
    # Prepare categorical features for CatBoost
    cat_features = [i for i, col in enumerate(X_full_train.columns) if col in CATEGORICAL_COLS]
    
    for col in CATEGORICAL_COLS:
        if col in X_full_train.columns:
            X_full_train[col] = X_full_train[col].astype(int)
            X_test[col] = X_test[col].astype(int)

    # Split train further into train/val for honest Optuna tuning without touching X_test
    X_train, X_val, y_train, y_val = train_test_split(
        X_full_train, y_full_train, test_size=0.2, random_state=42, stratify=y_full_train
    )

    print("\n[3] Running Optuna Optimization for CatBoost (15 trials)...")
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'depth': trial.suggest_int('depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'auto_class_weights': 'Balanced',
            'random_seed': 42,
            'verbose': 0
        }
        
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_val, y_val), early_stopping_rounds=50)
        
        y_val_pred = model.predict(X_val)
        return f1_score(y_val, y_val_pred)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15) # 15 trials to keep runtime reasonable
    
    print("\n[4] Optuna Best Parameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    # ---------------------------------------------------------
    # 4. Train Final Champion Model (Voting Ensemble)
    # ---------------------------------------------------------
    print("\n[5] Training Final Ensemble Champion on full training data...")
    final_params = study.best_params.copy()
    final_params['auto_class_weights'] = 'Balanced'
    final_params['random_seed'] = 42
    final_params['verbose'] = 0

    cat_model = CatBoostClassifier(**final_params)
    
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=7.5, # approx 88/12 ratio
        random_state=42,
        eval_metric='logloss'
    )
    
    lgb_model = LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )

    champion_model = ManualVotingClassifier(cat_model, xgb_model, lgb_model, cat_features)
    champion_model.fit(X_full_train, y_full_train)
    
    # Evaluate on untouched test set
    y_pred_test = champion_model.predict(X_test)
    metrics = print_metrics("Voting Ensemble (CatBoost+XGB+LGBM)", y_test, y_pred_test)

    # ---------------------------------------------------------
    # 5. Model Comparison & Save Logic
    # ---------------------------------------------------------
    print("\n[6] Comparing Models...")
    print(f"\n{'='*60}")
    print(f"{'MODEL COMPARISON':^60}")
    print(f"{'='*60}")
    print(f"{'Metric':<15} {'Logistic Reg':<20} {'Voting Ensemble':<20}")
    print(f"{'-'*55}")
    for m in ['accuracy', 'precision', 'recall', 'f1']:
        print(f"{m.capitalize():<15} {lr_metrics[m]:<20.4f} {metrics[m]:<20.4f}")
    print(f"{'='*60}\n")
    
    best_f1 = 0.0
    params_path = os.path.join(_BASE_DIR, "models", "best_params.json")
    if os.path.exists(params_path):
        try:
            with open(params_path, "r") as f:
                old_params = json.load(f)
                best_f1 = old_params.get("f1_score", 0.0)
                print(f"Loaded previous best model ({old_params.get('model_name')}) with F1: {best_f1:.4f}")
        except Exception:
            pass

    if metrics["f1"] > best_f1:
        print("*** New ALL-TIME BEST found! Overwriting previous model... ***")
        model_path = os.path.join(_BASE_DIR, "models", "best_model.pkl")
        joblib.dump(champion_model, model_path)
        
        current_best_params = {
            "model_name": "Voting Ensemble (CatBoost+XGB+LGBM)",
            "f1_score": metrics["f1"],
            "parameters": "Ensemble parameters fixed, CatBoost tuned via Optuna"
        }
        with open(params_path, "w") as f:
            json.dump(current_best_params, f, indent=4)
        
        print(f"Saved model to: {model_path}")
        print(f"Saved best parameters to: {params_path}")
    else:
        print(f"No new best found. Previous F1 ({best_f1:.4f}) is still the champion.")

    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
