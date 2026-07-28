import os
import pandas as pd

# ==========================
# Load Dataset
# ==========================
DATA_PATH = "../data/processed/creditcard_clean.csv"

df = pd.read_csv(DATA_PATH)

# ==========================
# Dataset Overview
# ==========================

rows = df.shape[0]
columns = df.shape[1]

target = "Class"

missing_values = df.isnull().sum().sum()

duplicate_records = df.duplicated().sum()

# ==========================
# Class Distribution
# ==========================

class_counts = df[target].value_counts()

normal = class_counts[0]
fraud = class_counts[1]

normal_percent = (normal / rows) * 100
fraud_percent = (fraud / rows) * 100

# ==========================
# Transaction Amount
# ==========================

amount_mean = df["Amount"].mean()
amount_median = df["Amount"].median()
amount_max = df["Amount"].max()

# ==========================
# Time Analysis
# ==========================

earliest = df["Time"].min()
latest = df["Time"].max()

# ==========================
# Correlation
# ==========================

corr = df.corr(numeric_only=True)["Class"].sort_values()

positive = corr.tail(6)
negative = corr.head(5)

# ==========================
# Write Markdown Report
# ==========================

report = f"""# Exploratory Data Analysis Report

---

## Dataset Overview

| Metric | Value |
|---------|-------|
| Rows | {rows:,} |
| Columns | {columns} |
| Target Column | {target} |
| Missing Values | {missing_values} |
| Duplicate Records | {duplicate_records} |

---

## Class Distribution

| Class | Count | Percentage |
|------|------:|-----------:|
| Normal (0) | {normal:,} | {normal_percent:.3f}% |
| Fraud (1) | {fraud:,} | {fraud_percent:.3f}% |

---

## Transaction Amount Statistics

| Metric | Value |
|---------|-------:|
| Mean | {amount_mean:.2f} |
| Median | {amount_median:.2f} |
| Maximum | {amount_max:.2f} |

---

## Time Analysis

| Metric | Value |
|---------|-------:|
| Earliest Transaction | {earliest} |
| Latest Transaction | {latest} |

---

## Most Positively Correlated Features

| Feature | Correlation |
|---------|------------:|
"""

for feature, value in positive.items():
    report += f"| {feature} | {value:.4f} |\n"

report += """

---

## Most Negatively Correlated Features

| Feature | Correlation |
|---------|------------:|
"""

for feature, value in negative.items():
    report += f"| {feature} | {value:.4f} |\n"

report += """

---

# Key Observations

- The dataset is highly imbalanced with less than **0.2%** fraudulent transactions.
- No missing values were detected.
- Duplicate records were removed during preprocessing.
- Most transactions have relatively small monetary values, while a few high-value transactions create a long-tailed distribution.
- Several PCA-transformed features (V1–V28) show meaningful correlation with the target class.
- The dataset is suitable for supervised binary classification after handling class imbalance using techniques such as **SMOTE**.

---

# Recommended Next Steps

- Feature Scaling
- Train-Test Split
- Handle Class Imbalance (SMOTE)
- Machine Learning Model Training
- Hyperparameter Tuning
- SHAP Explainability
- FastAPI Deployment
- Power BI Dashboard

"""

# ==========================
# Save Report
# ==========================

os.makedirs("../reports", exist_ok=True)

with open("../reports/eda_report.md", "w", encoding="utf-8") as f:
    f.write(report)

print("EDA Report Generated Successfully!")
print("Location: reports/eda_report.md")