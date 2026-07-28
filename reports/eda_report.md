# Exploratory Data Analysis Report

---

## Dataset Overview

| Metric | Value |
|---------|-------|
| Rows | 283,726 |
| Columns | 31 |
| Target Column | Class |
| Missing Values | 0 |
| Duplicate Records | 0 |

---

## Class Distribution

| Class | Count | Percentage |
|------|------:|-----------:|
| Normal (0) | 283,253 | 99.833% |
| Fraud (1) | 473 | 0.167% |

---

## Transaction Amount Statistics

| Metric | Value |
|---------|-------:|
| Mean | 88.47 |
| Median | 22.00 |
| Maximum | 25691.16 |

---

## Time Analysis

| Metric | Value |
|---------|-------:|
| Earliest Transaction | 0.0 |
| Latest Transaction | 172792.0 |

---

## Most Positively Correlated Features

| Feature | Correlation |
|---------|------------:|
| V8 | 0.0331 |
| V19 | 0.0336 |
| V2 | 0.0846 |
| V4 | 0.1293 |
| V11 | 0.1491 |
| Class | 1.0000 |


---

## Most Negatively Correlated Features

| Feature | Correlation |
|---------|------------:|
| V17 | -0.3135 |
| V14 | -0.2934 |
| V12 | -0.2507 |
| V10 | -0.2070 |
| V16 | -0.1872 |


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

