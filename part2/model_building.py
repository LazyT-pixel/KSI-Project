"""
KSI Predictive Model Building
Part 2 deliverable: train and compare baseline classifiers on top of the
Part 1 preprocessing pipeline (KSI.py), then score each on the held-out
test set.

Reuses KSI.py as-is (imports it as a module -- it already builds
X_train_processed / X_test_processed / y_train / y_test / full_pipeline
as top-level variables when run). This keeps the Part 1 pipeline as the
single source of truth for cleaning/encoding instead of duplicating it.

Models (baseline, default-ish hyperparameters -- tuning is a separate
pass once we agree these 5 are the right set for the assignment):
  1. Logistic Regression   (class_weight="balanced")
  2. Decision Tree         (class_weight="balanced")
  3. Random Forest         (class_weight="balanced")
  4. Linear SVM            (class_weight="balanced")
  5. Neural Network (MLP)  (no native class_weight -- see note below)

NOTE: check the Part 2 assignment sheet for the exact required algorithm
list/count before treating this as final -- swap any of these out if it
specifies different ones.

RUN THIS FROM THE REPO ROOT (not from inside part2/):
    python3 part2/model_building.py
"""

import os
import sys
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

# Lets this find KSI.py at the repo root no matter where this script is run
# from (this file lives in part2/, KSI.py lives one level up).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import KSI  # noqa: E402  (runs KSI.py's Part 1 pipeline, exposes the variables below)

X_train = KSI.X_train_processed
X_test = KSI.X_test_processed
y_train = KSI.y_train
y_test = KSI.y_test

# MLP needs dense input; the others are fine with either, so densify once
# and reuse everywhere for a fair/simple comparison.
if hasattr(X_train, "toarray"):
    X_train = X_train.toarray()
    X_test = X_test.toarray()

models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(class_weight="balanced", n_estimators=200, random_state=42, n_jobs=-1),
    "Linear SVM": LinearSVC(class_weight="balanced", max_iter=5000, random_state=42),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True, random_state=42),
}

results = []

print(f"\n{'='*70}\nTraining {len(models)} baseline models\n{'='*70}")

for name, model in models.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    elapsed = time.time() - t0

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    results.append({
        "model": name, "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1, "train_time_s": elapsed,
    })

    print(f"\n--- {name} ({elapsed:.1f}s) ---")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}  (of predicted-fatal, how many really were)")
    print(f"Recall:    {rec:.3f}  (of actual-fatal, how many we caught)")
    print(f"F1:        {f1:.3f}")
    print(f"Confusion matrix [[TN FP] [FN TP]]:\n{cm}")

print(f"\n{'='*70}\nSummary (sorted by recall -- catching fatal cases is the priority)\n{'='*70}")
results_sorted = sorted(results, key=lambda r: r["recall"], reverse=True)
print(f"{'Model':<22}{'Accuracy':<10}{'Precision':<11}{'Recall':<9}{'F1':<8}{'Time(s)'}")
for r in results_sorted:
    print(f"{r['model']:<22}{r['accuracy']:<10.3f}{r['precision']:<11.3f}{r['recall']:<9.3f}{r['f1']:<8.3f}{r['train_time_s']:.1f}")

print(
    "\nNote: accuracy alone is misleading here (~86% negative class baseline). "
    "Recall on the fatal class matters most for this problem -- a missed fatal "
    "prediction is a worse error than a false alarm. class_weight='balanced' is "
    "applied to every model that supports it to push toward higher recall; "
    "next step is hyperparameter tuning (GridSearchCV) on whichever models look "
    "most promising here, plus deciding on a final model for deployment."
)
