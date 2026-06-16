"""
eda.py — Exploratory Data Analysis for the BankMind Project.

This script performs a comprehensive EDA, printing key statistics and saving
plots to the 'assets/' folder. It answers the specific business questions
for Track A and provides deep feature insights for Track B.

Inspired by the UCI Bank Marketing Case Study notebook, this script follows
a per-feature deep-dive pattern:
  - Categorical features: count distribution + subscription rate side-by-side.
  - Numeric features: overlapping KDE (yes vs no) + box plot.
  - Correlation heatmap across all numeric features.
"""

import os
import sys
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# Add project root to path so we can import from src
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from src.data_loader import load_raw_data, add_engineered_features, NUMERIC_COLS

# ---------------------------------------------------------------------------
# Configure plotting style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("muted")
PALETTE = {"no": "#5B8DB8", "yes": "#E07B54"}   # blue = no, orange = yes

ASSETS_DIR = os.path.join(_BASE_DIR, "assets")


# ---------------------------------------------------------------------------
# Helper: dual-panel bar chart (count | subscription rate)
# ---------------------------------------------------------------------------

def plot_categorical_feature(df, col, title, order=None, figsize=(14, 5)):
    """
    Two-panel chart for a categorical feature:
      Left  — absolute count per category, stacked by subscription (yes/no)
      Right — subscription rate (%) per category, sorted descending
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    if order is None:
        order = df[col].value_counts().index.tolist()

    # --- Left: stacked bar (count) ---
    counts = df.groupby([col, "y"]).size().unstack(fill_value=0)
    counts = counts.reindex(order)
    counts[["no", "yes"]].plot(kind="bar", stacked=True, ax=ax1,
                               color=[PALETTE["no"], PALETTE["yes"]],
                               edgecolor="white", linewidth=0.5)
    ax1.set_title("Count by Outcome", fontsize=11)
    ax1.set_xlabel("")
    ax1.set_ylabel("Count")
    ax1.tick_params(axis="x", rotation=40)
    ax1.legend(title="Subscribed", labels=["No", "Yes"])

    # --- Right: subscription rate ---
    rate = (df.groupby(col)["y_numeric"].mean() * 100).reindex(order).sort_values(ascending=False)
    sns.barplot(x=rate.index, y=rate.values, ax=ax2, palette="viridis", order=rate.index)
    ax2.set_title("Subscription Rate (%)", fontsize=11)
    ax2.set_xlabel("")
    ax2.set_ylabel("% Subscribed")
    ax2.tick_params(axis="x", rotation=40)
    for p in ax2.patches:
        ax2.annotate(f"{p.get_height():.1f}%",
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    return fig


def plot_numeric_feature(df, col, title, figsize=(14, 4)):
    """
    Two-panel chart for a numeric feature:
      Left  — overlapping KDE for yes vs no subscribers
      Right — box plot by subscription status
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    for label, color in PALETTE.items():
        subset = df[df["y"] == label][col].dropna()
        subset.plot.kde(ax=ax1, label=label.capitalize(), color=color, linewidth=2)
    ax1.set_title("Distribution by Outcome (KDE)", fontsize=11)
    ax1.set_xlabel(col)
    ax1.legend(title="Subscribed")

    sns.boxplot(x="y", y=col, data=df, ax=ax2,
                palette=PALETTE, order=["no", "yes"])
    ax2.set_title("Box Plot by Outcome", fontsize=11)
    ax2.set_xlabel("Subscribed")
    ax2.set_ylabel(col)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main EDA function
# ---------------------------------------------------------------------------

def run_eda():
    print("=" * 65)
    print("BankMind EDA — Running Enhanced Analysis")
    print("=" * 65)

    # -----------------------------------------------------------------------
    # 0. Load data
    # -----------------------------------------------------------------------
    raw_df = load_raw_data()
    df = add_engineered_features(raw_df)
    os.makedirs(ASSETS_DIR, exist_ok=True)

    # -----------------------------------------------------------------------
    # SECTION 1: Dataset overview
    # -----------------------------------------------------------------------
    print(f"\n[1] DATASET OVERVIEW")
    print(f"    Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"    Dtypes:\n{df.dtypes.to_string()}")

    missing = df.isnull().sum()
    print(f"\n    Missing values:\n{missing[missing > 0].to_string() if missing.any() else '    None — dataset is complete.'}")

    print(f"\n    Numeric summary:\n{df.describe().round(2).to_string()}")

    # -----------------------------------------------------------------------
    # SECTION 2: Target distribution
    # -----------------------------------------------------------------------
    print(f"\n[2] TARGET DISTRIBUTION (y)")
    yes_count = (df["y"] == "yes").sum()
    no_count  = (df["y"] == "no").sum()
    total     = len(df)
    yes_pct   = yes_count / total * 100
    no_pct    = no_count  / total * 100
    print(f"    Yes (subscribed): {yes_count:,}  ({yes_pct:.1f}%)")
    print(f"    No (did not):     {no_count:,}  ({no_pct:.1f}%)")
    print(f"    >> Class imbalance ratio ~1 : {no_count / yes_count:.1f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["No", "Yes"], [no_count, yes_count],
           color=[PALETTE["no"], PALETTE["yes"]], edgecolor="white")
    ax.set_title("Target Variable Distribution", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count")
    for i, v in enumerate([no_count, yes_count]):
        ax.text(i, v + 200, f"{v:,}\n({[no_pct, yes_pct][i]:.1f}%)",
                ha="center", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "s2_target_distribution.png"), dpi=120)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # SECTION 3: Track-A Business Questions
    # -----------------------------------------------------------------------
    print(f"\n[3] TRACK-A BUSINESS QUESTIONS")

    # Q1 — Job types and subscription rate
    print("\n    [Q1] Subscription rate by job type:")
    job_order = df["job"].value_counts().index.tolist()
    job_fig = plot_categorical_feature(df, "job", "Q1 — Subscription by Job Type", order=job_order)
    job_fig.savefig(os.path.join(ASSETS_DIR, "q1_job_subscription.png"), dpi=120, bbox_inches="tight")
    plt.close(job_fig)
    job_rates = df.groupby("job")["y_numeric"].mean().sort_values(ascending=False) * 100
    print(job_rates.round(1).to_string())

    # Q2 — Balance vs subscription
    print("\n    [Q2] Balance vs subscription status:")
    avg_bal_no  = df[df["y"] == "no"]["balance"].mean()
    avg_bal_yes = df[df["y"] == "yes"]["balance"].mean()
    print(f"        Avg balance (No):  {avg_bal_no:.2f}")
    print(f"        Avg balance (Yes): {avg_bal_yes:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("Q2 — Account Balance by Subscription Status", fontsize=13, fontweight="bold")
    # KDE
    for label, color in PALETTE.items():
        df[df["y"] == label]["balance"].plot.kde(ax=axes[0], label=label.capitalize(),
                                                  color=color, linewidth=2)
    axes[0].set_title("Balance Distribution (KDE)")
    axes[0].set_xlabel("Balance (log-transformed)")
    axes[0].legend(title="Subscribed")
    # Boxplot — clip for readability
    sns.boxplot(x="y", y="balance", data=df, ax=axes[1],
                palette=PALETTE, order=["no", "yes"])
    axes[1].set_title("Balance Box Plot")
    axes[1].set_xlabel("Subscribed")
    axes[1].set_ylabel("Balance (log-transformed)")
    plt.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "q2_balance_subscription.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Q3 — Subscription rate by age group
    print("\n    [Q3] Subscription rate by age group:")
    age_rates = df.groupby("age_group", observed=False)["y_numeric"].mean() * 100
    print(age_rates.round(1).to_string())

    # Dual-panel: count vs rate
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Q3 — Subscription by Age Group", fontsize=13, fontweight="bold")
    counts_age = df.groupby(["age_group", "y"], observed=False).size().unstack(fill_value=0)
    counts_age[["no", "yes"]].plot(kind="bar", stacked=True, ax=ax1,
                                   color=[PALETTE["no"], PALETTE["yes"]],
                                   edgecolor="white")
    ax1.set_title("Count by Outcome")
    ax1.set_xlabel("Age Group")
    ax1.tick_params(axis="x", rotation=0)
    ax1.legend(title="Subscribed", labels=["No", "Yes"])

    sns.barplot(x=age_rates.index, y=age_rates.values, ax=ax2, palette="magma")
    ax2.set_title("Subscription Rate (%)")
    ax2.set_xlabel("Age Group")
    ax2.set_ylabel("% Subscribed")
    for p in ax2.patches:
        ax2.annotate(f"{p.get_height():.1f}%",
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "q3_age_subscription.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Q4 — Housing loan effect
    print("\n    [Q4] Housing loan effect on subscription:")
    housing_rates = df.groupby("housing")["y_numeric"].mean() * 100
    print(housing_rates.round(1).to_string())

    housing_fig = plot_categorical_feature(df, "housing",
                                           "Q4 — Subscription by Housing Loan Status",
                                           order=["yes", "no"],
                                           figsize=(10, 5))
    housing_fig.savefig(os.path.join(ASSETS_DIR, "q4_housing_subscription.png"),
                         dpi=120, bbox_inches="tight")
    plt.close(housing_fig)

    # -----------------------------------------------------------------------
    # SECTION 4: Campaign Feature Deep-Dive  [NEW — from reference notebook]
    # -----------------------------------------------------------------------
    print(f"\n[4] CAMPAIGN FEATURE DEEP-DIVE")

    # 4a. Contact type
    print("\n    [4a] Contact type subscription rate:")
    contact_rates = df.groupby("contact")["y_numeric"].mean() * 100
    print(contact_rates.round(1).to_string())
    contact_fig = plot_categorical_feature(df, "contact",
                                           "Contact Type vs Subscription",
                                           figsize=(10, 5))
    contact_fig.savefig(os.path.join(ASSETS_DIR, "s4a_contact_subscription.png"),
                         dpi=120, bbox_inches="tight")
    plt.close(contact_fig)

    # 4b. Month of last contact
    print("\n    [4b] Month of last contact subscription rate:")
    month_order = ["jan", "feb", "mar", "apr", "may", "jun",
                   "jul", "aug", "sep", "oct", "nov", "dec"]
    # Keep only months present in the data
    month_order = [m for m in month_order if m in df["month"].unique()]
    month_fig = plot_categorical_feature(df, "month",
                                         "Month of Last Contact vs Subscription",
                                         order=month_order,
                                         figsize=(16, 5))
    month_fig.savefig(os.path.join(ASSETS_DIR, "s4b_month_subscription.png"),
                       dpi=120, bbox_inches="tight")
    plt.close(month_fig)
    month_rates = df.groupby("month")["y_numeric"].mean().reindex(month_order) * 100
    print(month_rates.round(1).to_string())

    # 4c. Call duration — top numeric predictor but known AFTER the call
    print("\n    [4c] Call duration analysis ([!] leakage risk - only for reference):")
    print(f"        Mean duration (No):  {df[df['y'] == 'no']['duration'].mean():.0f}s")
    print(f"        Mean duration (Yes): {df[df['y'] == 'yes']['duration'].mean():.0f}s")
    dur_fig = plot_numeric_feature(df, "duration",
                                   "Call Duration vs Subscription\n([!] Data Leakage Risk - Not used in model)")
    dur_fig.savefig(os.path.join(ASSETS_DIR, "s4c_duration_subscription.png"),
                     dpi=120, bbox_inches="tight")
    plt.close(dur_fig)

    # 4d. Campaign (number of contacts in current campaign)
    print("\n    [4d] Number of campaign contacts:")
    print(f"        Median contacts (No):  {df[df['y'] == 'no']['campaign'].median():.0f}")
    print(f"        Median contacts (Yes): {df[df['y'] == 'yes']['campaign'].median():.0f}")
    # Cap at 10 for better viz (extreme outliers exist)
    df_cap = df.copy()
    df_cap["campaign_capped"] = df_cap["campaign"].clip(upper=10)
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, color in PALETTE.items():
        subset = df_cap[df_cap["y"] == label]["campaign_capped"]
        ax.hist(subset, bins=range(1, 12), alpha=0.6, label=label.capitalize(),
                color=color, edgecolor="white", density=True)
    ax.set_title("Campaign Contact Count Distribution\n(capped at 10 for clarity)", fontsize=13)
    ax.set_xlabel("Number of Contacts (campaign)")
    ax.set_ylabel("Density")
    ax.legend(title="Subscribed")
    plt.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "s4d_campaign_distribution.png"), dpi=120)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # SECTION 5: Previous Campaign Signals  [NEW — from reference notebook]
    # -----------------------------------------------------------------------
    print(f"\n[5] PREVIOUS CAMPAIGN SIGNALS")

    # 5a. poutcome — outcome of previous campaign (very predictive)
    print("\n    [5a] Previous campaign outcome subscription rate:")
    pout_rates = df.groupby("poutcome")["y_numeric"].mean() * 100
    print(pout_rates.round(1).to_string())
    pout_fig = plot_categorical_feature(df, "poutcome",
                                        "Previous Campaign Outcome vs Subscription",
                                        figsize=(11, 5))
    pout_fig.savefig(os.path.join(ASSETS_DIR, "s5a_poutcome_subscription.png"),
                      dpi=120, bbox_inches="tight")
    plt.close(pout_fig)

    # 5b. pdays — days since last contact from previous campaign (-1 = never)
    print("\n    [5b] pdays analysis:")
    never_contacted = (df["pdays"] == -1).sum()
    print(f"        Never contacted before: {never_contacted:,} ({never_contacted/total*100:.1f}%)")
    contacted_df = df[df["pdays"] != -1]
    if len(contacted_df) > 0:
        print(f"        Among those contacted: mean pdays = {contacted_df['pdays'].mean():.1f} days")
        rate_contacted = contacted_df["y_numeric"].mean() * 100
        rate_not       = df[df["pdays"] == -1]["y_numeric"].mean() * 100
        print(f"        Subscription rate if previously contacted: {rate_contacted:.1f}%")
        print(f"        Subscription rate if NOT previously contacted: {rate_not:.1f}%")

    # 5c. Marital status (demographic signal)
    print("\n    [5c] Marital status subscription rate:")
    marital_fig = plot_categorical_feature(df, "marital",
                                           "Marital Status vs Subscription",
                                           figsize=(10, 5))
    marital_fig.savefig(os.path.join(ASSETS_DIR, "s5c_marital_subscription.png"),
                         dpi=120, bbox_inches="tight")
    plt.close(marital_fig)

    # 5d. Education level
    print("\n    [5d] Education level subscription rate:")
    edu_order = ["primary", "secondary", "tertiary", "unknown"]
    edu_order = [e for e in edu_order if e in df["education"].unique()]
    edu_fig = plot_categorical_feature(df, "education",
                                       "Education Level vs Subscription",
                                       order=edu_order,
                                       figsize=(11, 5))
    edu_fig.savefig(os.path.join(ASSETS_DIR, "s5d_education_subscription.png"),
                     dpi=120, bbox_inches="tight")
    plt.close(edu_fig)

    # -----------------------------------------------------------------------
    # SECTION 6: Correlation heatmap (Track B)
    # -----------------------------------------------------------------------
    print(f"\n[6] CORRELATION HEATMAP (Track B)")
    cols_to_corr = [c for c in NUMERIC_COLS if c in df.columns] + ["y_numeric"]
    corr = df[cols_to_corr].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f",
                vmin=-1, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title("Correlation Heatmap of Numeric Features", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "s6_correlation_heatmap.png"), dpi=120)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # SECTION 7: Summary / Key Findings
    # -----------------------------------------------------------------------
    print(f"\n[7] KEY FINDINGS SUMMARY")
    print("    1. Target imbalance: ~88% 'no' vs ~12% 'yes' → use F1/Recall, not Accuracy.")
    print("    2. Students & retired customers have the highest subscription rates.")
    print("    3. Cellular contact outperforms telephone contact significantly.")
    print("    4. Customers contacted in Mar, Sep, Oct, Dec convert at much higher rates.")
    print("    5. Longer call duration is strongly correlated with subscription (leakage risk).")
    print("    6. Customers with a 'success' in a prior campaign are ~65% likely to subscribe.")
    print("    7. High campaign contact count (>5) correlates with lower conversion — fatigue.")
    print("    8. Tertiary education customers have higher subscription rates.")

    print(f"\nAll plots saved to: {ASSETS_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    run_eda()
