# Week 4 — Supervised Learning Model Implementation

## Project
Binary classification of the Wisconsin Diagnostic Breast Cancer (WDBC) dataset using a Scikit-learn Pipeline with `StandardScaler` and `LogisticRegression`.

> **Educational use only:** This project is not a medical device and must not be used for diagnosis or treatment decisions.

## Dataset
The dataset is bundled with scikit-learn and loads through `sklearn.datasets.load_breast_cancer`. It contains 569 samples and 30 numeric features. Target mapping in scikit-learn: `0 = malignant`, `1 = benign`. The script checks for missing values before training.

Sources:
- UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- scikit-learn dataset documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

## Repository structure
```text
Week4_Supervised_Learning_Project/
├── README.md
├── requirements.txt
├── .gitignore
├── Week4_Supervised_Learning_Report.docx
├── src/
│   └── train_model.py
└── outputs/
    └── (generated metrics and plots appear here after running)
```

## Run locally
Python 3.10+ is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/train_model.py
```

macOS/Linux activation:
```bash
source .venv/bin/activate
pip install -r requirements.txt
python src/train_model.py
```

## Method
1. Load data from scikit-learn.
2. Split data into 80% train / 20% test using stratification and `random_state=42`.
3. Use a pipeline with standardisation and logistic regression to avoid leakage from preprocessing.
4. Report accuracy, precision, recall, F1, ROC-AUC, and confusion matrix.
5. Run five-fold stratified cross-validation.
6. Save JSON metrics, ROC curve, confusion matrix, and standardised logistic coefficients to `outputs/`.

## Baseline result
With scikit-learn's WDBC dataset and the settings in the script, the expected held-out results are approximately 98.25% accuracy and 0.995 ROC-AUC. Exact values can vary slightly across library versions. Consult the generated `outputs/metrics.json` for the results from your own run.

## Important interpretation note
The positive class in this script is benign (`1`). In a clinical context, malignant-class sensitivity and error costs require separate attention. High scores on this small curated dataset do not prove generalisation or clinical safety. External validation, calibration, fairness assessment, and expert review would be necessary for any real-world application.
