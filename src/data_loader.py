"""
data_loader.py — Shared data loading & preprocessing for BankMind project.

This module provides functions to:
    1. Load the UCI Bank Marketing dataset (bank-full.csv)
    2. Add engineered features (age bins, contacted_before flag)
    3. Encode categorical variables (label encoding + one-hot encoding)
    4. Create a stratified train/test split ready for modeling

Usage:
    from src.data_loader import load_raw_data, preprocess_for_modeling

Other scripts (eda.py, model_training.py, dashboard.py) all import from here
so the preprocessing logic stays in one place.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Path to the dataset — resolved relative to this file's location
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_BASE_DIR, "bank", "bank-full.csv")

# Age bins matching the task requirements
AGE_BINS = [17, 30, 45, 60, 100]
AGE_LABELS = ["18-30", "31-45", "46-60", "60+"]

# Columns grouped by type
CATEGORICAL_COLS = [
    "job", "marital", "education", "default",
    "housing", "loan", "contact", "month", "poutcome",
]
BINARY_TARGET = "y"
NUMERIC_COLS = [
    "age", "balance", "day", "duration",
    "campaign", "pdays", "previous",
    "total_loans", "wealth_age_ratio", "campaign_fatigue"
]

# Random seed for reproducibility
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------

def load_raw_data(filepath: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the bank-full.csv dataset as-is.

    Parameters
    ----------
    filepath : str
        Path to the CSV file (default: bank/bank-full.csv).

    Returns
    -------
    pd.DataFrame
        Raw dataframe with 45,211 rows and 17 columns.
    """
    df = pd.read_csv(filepath, sep=";")
    return df


# ---------------------------------------------------------------------------
# 2. Feature engineering (non-destructive — adds columns)
# ---------------------------------------------------------------------------

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features used by both the EDA and the models.

    New columns:
        - age_group : categorical bin (18-30, 31-45, 46-60, 60+)
        - contacted_before : 1 if pdays != -1 (customer was contacted in a
          prior campaign), else 0
        - y_numeric : 1 for 'yes', 0 for 'no' — convenient for groupby means

    Returns a copy so the original is never mutated.
    """
    df = df.copy()

    # Age bins as specified in the task
    df["age_group"] = pd.cut(
        df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=True
    )

    # Flag: was this customer contacted in a previous campaign?
    # pdays == -1 means "never contacted before"
    df["contacted_before"] = (df["pdays"] != -1).astype(int)

    # Feature: y_numeric
    if "y" in df.columns:
        df["y_numeric"] = (df["y"] == "yes").astype(int)
        
    # --- ADVANCED FEATURE ENGINEERING ---
    # 2. Total Debt Indicator (0, 1, or 2 loans)
    if "housing" in df.columns and "loan" in df.columns:
        df["total_loans"] = (df["housing"] == "yes").astype(int) + (df["loan"] == "yes").astype(int)

    # 3. Wealth Proxy (Raw Balance / Age)
    if "balance" in df.columns and "age" in df.columns:
        df["wealth_age_ratio"] = df["balance"] / df["age"]

    # 4. Campaign Fatigue (Current contacts / Previous contacts)
    if "campaign" in df.columns and "previous" in df.columns:
        df["campaign_fatigue"] = df["campaign"] / (df["previous"] + 1.0)
        
    # Feature: log_balance (Symmetric log transform to handle outliers)
    # Applied last so wealth_age_ratio gets the raw untransformed balance
    if "balance" in df.columns:
        # np.sign(x) * np.log1p(abs(x)) preserves negative balances while squishing extreme values
        df["balance"] = np.sign(df["balance"]) * np.log1p(np.abs(df["balance"]))

    return df


# ---------------------------------------------------------------------------
# 3. Encode categoricals
# ---------------------------------------------------------------------------

def label_encode(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Label-encode all categorical columns in-place style (returns a copy).

    Useful for tree-based models (XGBoost, Random Forest) which work fine
    with ordinal integers.

    Returns
    -------
    df_encoded : pd.DataFrame
        DataFrame with categoricals replaced by integer codes.
    encoders : dict[str, LabelEncoder]
        Mapping of column name → fitted LabelEncoder (for inverse transforms).
    """
    df = df.copy()
    encoders: dict[str, LabelEncoder] = {}

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Encode the target column as well
    le_target = LabelEncoder()
    df[BINARY_TARGET] = le_target.fit_transform(df[BINARY_TARGET].astype(str))
    encoders[BINARY_TARGET] = le_target

    return df, encoders


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode all categorical columns.

    Useful for linear models (Logistic Regression) that cannot handle
    ordinal integer encoding meaningfully.

    The target column 'y' is converted to 0/1 but NOT one-hot encoded
    (it's binary, not multi-class).
    """
    df = df.copy()

    # Convert target to numeric first
    df[BINARY_TARGET] = (df[BINARY_TARGET] == "yes").astype(int)

    # One-hot encode the categorical features
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    return df


# ---------------------------------------------------------------------------
# 4. Train / test split
# ---------------------------------------------------------------------------

def get_train_test_split(
    df: pd.DataFrame,
    target_col: str = BINARY_TARGET,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified 80/20 train-test split.

    Stratification ensures the ~88/12 class ratio is preserved in both
    the training and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe (must already have target encoded as int).
    target_col : str
        Name of the target column.
    test_size : float
        Fraction of data to hold out for testing.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    # Drop non-feature columns
    cols_to_drop = [target_col]
    # Also drop helper columns that shouldn't be features
    for helper_col in ["y_numeric", "age_group"]:
        if helper_col in df.columns:
            cols_to_drop.append(helper_col)
            
    # CRITICAL: Drop 'duration' to prevent data leakage.
    # Duration is highly predictive but only known AFTER the call is made,
    # which makes it invalid for predicting WHO to call beforehand.
    if "duration" in df.columns:
        cols_to_drop.append("duration")

    X = df.drop(columns=cols_to_drop, errors="ignore")
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Standardize numeric features (zero mean, unit variance).

    Needed for Logistic Regression; tree models don't require this
    but it doesn't hurt them either.

    Fits only on X_train to avoid data leakage into X_test.
    """
    scaler = StandardScaler()

    # Only scale numeric columns that exist in the dataframe
    num_cols = [c for c in NUMERIC_COLS if c in X_train.columns]

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_test, scaler


# ---------------------------------------------------------------------------
# 5. High-level convenience function
# ---------------------------------------------------------------------------

def preprocess_for_modeling(
    encoding: str = "label",
    scale: bool = False,
) -> dict:
    """
    End-to-end pipeline: load → engineer → encode → split → (optional) scale.

    Parameters
    ----------
    encoding : str
        'label' for tree-based models, 'onehot' for linear models.
    scale : bool
        Whether to standardize numeric features (recommended for LR).

    Returns
    -------
    dict with keys:
        'X_train', 'X_test', 'y_train', 'y_test',
        'encoders' (if label encoding),
        'scaler' (if scale=True),
        'raw_df' (original unprocessed dataframe),
        'feature_names' (list of feature column names)
    """
    result = {}

    # Load raw data
    raw_df = load_raw_data()
    result["raw_df"] = raw_df

    # Engineer features
    df = add_engineered_features(raw_df)

    # Encode
    if encoding == "label":
        df_encoded, encoders = label_encode(df)
        result["encoders"] = encoders
    elif encoding == "onehot":
        df_encoded = one_hot_encode(df)
    else:
        raise ValueError(f"encoding must be 'label' or 'onehot', got '{encoding}'")

    # Split
    X_train, X_test, y_train, y_test = get_train_test_split(df_encoded)
    result["feature_names"] = list(X_train.columns)

    # Optionally scale
    if scale:
        X_train, X_test, scaler = scale_features(X_train, X_test)
        result["scaler"] = scaler

    result["X_train"] = X_train
    result["X_test"] = X_test
    result["y_train"] = y_train
    result["y_test"] = y_test

    return result


# ---------------------------------------------------------------------------
# Self-test: run this file directly to verify everything works
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("BankMind Data Loader — Self-Test")
    print("=" * 60)

    # 1. Raw data
    raw = load_raw_data()
    print(f"\n[1] Raw data loaded: {raw.shape[0]:,} rows × {raw.shape[1]} cols")
    print(f"    Columns: {list(raw.columns)}")

    # 2. Engineered features
    df = add_engineered_features(raw)
    print(f"\n[2] After feature engineering: {df.shape[1]} cols (+3 new)")
    print(f"    Age group distribution:\n{df['age_group'].value_counts().sort_index().to_string()}")
    print(f"    Contacted before: {df['contacted_before'].sum():,} / {len(df):,}")
    print(f"    Target (y_numeric): {df['y_numeric'].sum():,} yes / {len(df):,} total "
          f"({df['y_numeric'].mean()*100:.1f}%)")

    # 3. Label encoding pipeline
    print("\n[3] Label-encoded pipeline (for tree models):")
    data_label = preprocess_for_modeling(encoding="label", scale=False)
    print(f"    X_train: {data_label['X_train'].shape}")
    print(f"    X_test:  {data_label['X_test'].shape}")
    print(f"    y_train class dist: {data_label['y_train'].value_counts().to_dict()}")
    print(f"    y_test  class dist: {data_label['y_test'].value_counts().to_dict()}")
    print(f"    Features: {data_label['feature_names']}")

    # 4. One-hot encoding pipeline
    print("\n[4] One-hot encoded pipeline (for linear models):")
    data_ohe = preprocess_for_modeling(encoding="onehot", scale=True)
    print(f"    X_train: {data_ohe['X_train'].shape}")
    print(f"    X_test:  {data_ohe['X_test'].shape}")
    print(f"    Scaled: Yes")

    print("\n" + "=" * 60)
    print("All checks passed. data_loader.py is ready.")
    print("=" * 60)
