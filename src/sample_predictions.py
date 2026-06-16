"""
sample_predictions.py — Phase 4: Sample Predictions & SHAP Analysis

Loads the trained XGBoost model and:
1. Picks 5 test customers and prints a readable summary of their predictions.
2. Generates SHAP feature importance plots for interpretability (Bonus).
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

# Add project root to path so we can import from src
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from src.data_loader import preprocess_for_modeling, CATEGORICAL_COLS
from src.model_training import ManualVotingClassifier

MODELS_DIR = os.path.join(_BASE_DIR, "models")
ASSETS_DIR = os.path.join(_BASE_DIR, "assets")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")

def decode_features(row_series: pd.Series, encoders: dict) -> dict:
    """Helper to convert encoded integers back to readable strings for printing."""
    readable = {}
    for col, val in row_series.items():
        if col in encoders and col != "y":
            # Inverse transform the label encoder
            try:
                readable[col] = encoders[col].inverse_transform([int(val)])[0]
            except Exception:
                readable[col] = val
        else:
            readable[col] = val
    return readable

def run_predictions_and_shap():
    print("=" * 60)
    print("BankMind — Sample Predictions & SHAP Analysis")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please run 'python src/model_training.py' first.")
        return

    # Load model and data
    print("[1] Loading saved model and test data...")
    saved_artifacts = joblib.load(MODEL_PATH)
    
    if isinstance(saved_artifacts, dict) and "model" in saved_artifacts:
        model = saved_artifacts["model"]
        encoders = saved_artifacts.get("encoders", {})
    else:
        model = saved_artifacts
        encoders = {}
    
    data = preprocess_for_modeling(encoding="label", scale=False)
    X_test = data["X_test"]
    y_test = data["y_test"]
    
    noisy_features = ["day"]
    cols_to_drop = [c for c in noisy_features if c in X_test.columns]
    if cols_to_drop:
        X_test = X_test.drop(columns=cols_to_drop)

    # Generate predictions and probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability of class 1 ('yes')

    X_test_with_preds = X_test.copy()
    X_test_with_preds["Actual"] = y_test
    X_test_with_preds["Predicted"] = y_pred
    X_test_with_preds["Probability"] = y_prob

    # ---------------------------------------------------------
    # 1. Pick 5 Sample Customers
    # ---------------------------------------------------------
    print("\n[2] Extracting 5 Sample Customers...")
    
    # We want: 2 strong YES, 2 strong NO, 1 borderline
    strong_yes = X_test_with_preds[(X_test_with_preds["Predicted"] == 1) & (X_test_with_preds["Actual"] == 1)].sort_values(by="Probability", ascending=False).head(2)
    strong_no = X_test_with_preds[(X_test_with_preds["Predicted"] == 0) & (X_test_with_preds["Actual"] == 0)].sort_values(by="Probability", ascending=True).head(2)
    borderline = X_test_with_preds[(X_test_with_preds["Probability"] > 0.45) & (X_test_with_preds["Probability"] < 0.55)].head(1)

    samples = pd.concat([strong_yes, strong_no, borderline])

    print("\n--- 5 Sample Predictions ---")
    for idx, row in samples.iterrows():
        # Get readable features
        features = row.drop(["Actual", "Predicted", "Probability"])
        readable_features = decode_features(features, encoders)
        
        prob_pct = row["Probability"] * 100
        actual_str = "Yes" if row["Actual"] == 1 else "No"
        pred_str = "Yes" if row["Predicted"] == 1 else "No"
        
        print(f"\nCustomer ID: {idx}")
        print(f"Profile: Age {readable_features.get('age')}, {readable_features.get('job')} ({readable_features.get('education')})")
        print(f"Finances: Balance €{readable_features.get('balance')}, Housing Loan: {readable_features.get('housing')}, Personal Loan: {readable_features.get('loan')}")
        print(f"Campaign: Duration {readable_features.get('duration')}s, Previous outcome: {readable_features.get('poutcome')}")
        print(f"--> Model Prediction: {pred_str} ({prob_pct:.1f}% probability)")
        print(f"--> Actual Truth:     {actual_str}")

    # ---------------------------------------------------------
    # 2. SHAP Analysis (Bonus)
    # ---------------------------------------------------------
    print("\n[3] Generating SHAP Feature Importance Plots...")
    os.makedirs(ASSETS_DIR, exist_ok=True)

    # Use a random sample of 500 test instances to speed up SHAP calculation
    X_test_sample = X_test.sample(500, random_state=42)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_sample)

    # Plot 1: SHAP Summary Plot (Beeswarm)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_sample, show=False)
    plt.tight_layout()
    summary_path = os.path.join(ASSETS_DIR, "shap_summary_plot.png")
    plt.savefig(summary_path)
    plt.close()
    print(f"Saved SHAP summary plot: {summary_path}")

    # Plot 2: SHAP Bar Plot (Global Importance)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False)
    plt.tight_layout()
    bar_path = os.path.join(ASSETS_DIR, "shap_bar_plot.png")
    plt.savefig(bar_path)
    plt.close()
    print(f"Saved SHAP bar plot: {bar_path}")

    print("=" * 60)

if __name__ == "__main__":
    run_predictions_and_shap()
