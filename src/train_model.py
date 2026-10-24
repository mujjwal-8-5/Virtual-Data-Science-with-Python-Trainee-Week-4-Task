"""Train and evaluate a Logistic Regression classifier on the WDBC dataset."""
from pathlib import Path
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

def main():
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target  # 0 = malignant, 1 = benign
    if X.isna().any().any() or y.isna().any():
        raise ValueError("Missing values detected. Add justified imputation inside the pipeline.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000, random_state=42))
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "dataset": "Wisconsin Diagnostic Breast Cancer (scikit-learn bundled copy)",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "class_mapping": {"0": "malignant", "1": "benign"},
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "test_metrics_positive_class_benign": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
        },
        "confusion_matrix_labels_order_0_malignant_1_benign": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
        "versions": {"python_note": "See your active environment", "scikit_learn": sklearn.__version__, "pandas": pd.__version__},
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}
    cv_result = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)
    metrics["five_fold_stratified_cv"] = {}
    for key in scoring:
        vals = cv_result[f"test_{key}"]
        metrics["five_fold_stratified_cv"][key] = {
            "fold_scores": [float(v) for v in vals],
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=[0, 1], display_labels=["Malignant", "Benign"],
        cmap="Blues", values_format="d"
    )
    disp.ax_.set_title("Confusion Matrix — Held-out Test Set")
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix.png", dpi=160)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, y_prob, name="Logistic Regression (benign positive class)")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.title("ROC Curve — Held-out Test Set")
    plt.tight_layout()
    plt.savefig(OUT / "roc_curve.png", dpi=160)
    plt.close()

    # Coefficients are standardised-feature coefficients; interpret cautiously due to correlated features.
    clf = model.named_steps["classifier"]
    coef = pd.DataFrame({"feature": X.columns, "coefficient_for_benign_class": clf.coef_[0]})
    coef["absolute_coefficient"] = coef["coefficient_for_benign_class"].abs()
    coef.sort_values("absolute_coefficient", ascending=False).to_csv(OUT / "feature_coefficients.csv", index=False)

    print("Dataset:", data.DESCR.splitlines()[0])
    print(f"Samples: {len(X)} | Features: {X.shape[1]} | Train: {len(X_train)} | Test: {len(X_test)}")
    print("\nTest metrics (positive class = benign):")
    for key, value in metrics["test_metrics_positive_class_benign"].items():
        print(f"  {key}: {value:.4f}")
    print("\nConfusion matrix [malignant, benign]:")
    print(np.array(metrics["confusion_matrix_labels_order_0_malignant_1_benign"]))
    print("\nFive-fold cross-validation means:")
    for key, value in metrics["five_fold_stratified_cv"].items():
        print(f"  {key}: {value['mean']:.4f} (std {value['std']:.4f})")
    print(f"\nOutputs saved to: {OUT}")

if __name__ == "__main__":
    main()
