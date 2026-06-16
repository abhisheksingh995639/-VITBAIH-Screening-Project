# Track B (ML Engineer) Submission

> **Goal:** Predict which clients will subscribe to a term deposit based on direct marketing campaigns (phone calls) from a Portuguese banking institution.

This repository contains a complete, production-ready Machine Learning pipeline designed to handle extreme class imbalances (11.7% conversion rate) while maximizing business ROI for Relationship Managers.

---

## 🚀 Project Highlights

*   **Identified & Fixed Data Leakage:** Removed the `duration` feature (call length) which is unknown prior to making a call, ensuring the model's predictions are strictly causal and viable for real-world deployment.
*   **Robust Baseline Testing:** Established a strict Logistic Regression baseline using `StandardScaler` and balanced class weights to understand the true difficulty of the dataset.
*   **Optuna Bayesian Optimization:** Replaced static GridSearch with Tree-structured Parzen Estimator (TPE) trials to find highly regularized, deep-tree configurations for CatBoost.
*   **Custom Voting Ensemble:** Overcame the performance ceiling of single algorithms by building a custom ensemble that averages the soft probabilities of CatBoost, XGBoost, and LightGBM.
*   **SHAP Interpretability:** Implemented SHapley Additive exPlanations to visualize global feature importance and directional impact (e.g., proving that prior campaign success is the strongest predictor of future success).

---

## 📊 Model Performance

In a highly imbalanced dataset where a naive "predict No for everyone" model achieves 88.3% accuracy, we optimized strictly for the **F1 Score** to balance wasted calls (False Positives) against missed revenue (False Negatives).

| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression (Baseline) | 68.89% | 22.41% | 67.39% | 0.3364 |
| **Voting Ensemble (Champion)** | **81.85%** | **34.67%** | **62.29%** | **0.4454** |

*By maintaining a 62% Recall with 34.6% Precision, an RM only has to make ~3 calls to find 1 subscriber, drastically improving efficiency over the baseline 11.7% conversion rate.*

---

## 🛠️ Feature Engineering Dictionary
Beyond raw data, we created custom mathematical features in `data_loader.py` to capture latent customer behavior:

| Feature Name | Logic | Business Rationale |
| :--- | :--- | :--- |
| `total_loans` | `1 if housing == 'yes' or loan == 'yes' else 0` | Simplifies the debt profile into a single boolean constraint. |
| `wealth_age_ratio` | `balance / age` | Differentiates between young high-earners and older individuals with depleted savings, creating a stronger financial health signal. |
| `campaign_fatigue` | `1 if campaign > 5 else 0` | Captures the psychological breaking point where repeated phone calls actively annoy the customer and ruin conversion chances. |

---

## ⚠️ Model Limitations & Caveats
While a Recall of 62% is exceptionally strong for this dataset, our Precision remains mathematically capped at ~35%. This is a known limitation when predicting human behaviour using demographic and financial proxies. A customer might have the perfect statistical profile to subscribe, but if they simply had a bad day or are busy when the RM calls, they will reject the offer. This inherent psychological randomness sets a hard mathematical ceiling on precision that no algorithm can overcome.

---

## 📂 Repository Structure

```text
├── assets/                  # Output directory for SHAP feature importance plots (.png)
├── data/                    # Contains the raw Bank Marketing CSV datasets
├── models/                  # Stores the trained champion .pkl and Optuna .json parameters
├── src/                     # Source code directory
│   ├── eda.py               # Exploratory Data Analysis & visual generation
│   ├── data_loader.py       # Feature engineering, categorical encoding, and scaling
│   ├── model_training.py    # Pipeline for training the Baseline, tuning Optuna, and building the Ensemble
│   └── sample_predictions.py# Inference script that outputs 5 samples and generates SHAP visuals
├── EXPLANATION.md           # Detailed technical diary and answers to the 5 mandatory Track B questions
├── changes.md               # Developer change log tracking iterations and metric improvements
├── requirements.txt         # Clean, minimal dependencies
└── README.md                # This file
```

---

## ⚙️ How to Run Locally

**1. Install Dependencies**
Ensure you have Python 3.9+ installed. We recommend using a virtual environment.
```bash
pip install -r requirements.txt
```

**2. Execute the Training Pipeline**
This script will load the data, train the baseline Logistic Regression, run Optuna Bayesian Optimization trials, construct the custom Voting Ensemble, evaluate against the test set, and save the best model to disk.
```bash
python src/model_training.py
```
*Note: Due to Optuna trials and tree building, this may take 1-3 minutes to complete.*

**3. Run Inference & Generate SHAP Visuals**
This script loads the `best_model.pkl`, runs it against 5 specific customer profiles (showing the model's logic for high, low, and borderline probabilities), and generates SHAP beeswarm plots.
```bash
python src/sample_predictions.py
```
*Note: Look inside your console for the text output, and check the `assets/` folder for the newly generated SHAP `.png` files.*
