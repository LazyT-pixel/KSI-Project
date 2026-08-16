"""
Part 2 - Final model selection (Aidan).

Why this exists: the INJURY data leak (found by Aboud, Aug 17) invalidated
every model result reported before the fix, and Ibrahim hadn't submitted his
Decision Tree / Random Forest tuning as of the day before the deadline. Rather
than leave the submission with mismatched pre-fix and post-fix numbers, or an
incomplete comparison, this runs a single consistent tuning pass for all 5
models on the corrected pipeline and picks the final model for deployment.

This does NOT throw away Aboud's tuning work - his grid values, the choice to
add randomized search + ROC/AUC curves, and his conclusions about how C
affects the accuracy/recall trade-off are still credited and referenced in the
report. Only the specific numbers needed re-running because the leaked feature
changed what "best" means.

Scoring: recall is the primary selection metric (catching real fatal cases
matters more than raw accuracy for this problem - see PART2_PROGRESS.md for
the full reasoning). F1 is the tiebreaker.

Output: prints a full comparison table, then pickles the winning model
(preprocessing + classifier together, as one pipeline) to
part2/final_model.pkl for the Flask deployment.
"""

import os
import sys
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import KSI  # runs the corrected Part 1 pipeline (INJURY + FATAL_NO both dropped)

X_train_raw = KSI.X_train
X_test_raw = KSI.X_test
y_train = KSI.y_train
y_test = KSI.y_test
preprocessor = KSI.preprocessor

# Each entry: (name, model, param_grid). Grids are intentionally small - the
# goal tonight is one complete, honest, correct comparison before the
# deadline, not an exhaustive search. Ranges are informed by what Aboud/
# Ibrahim already explored in their own scripts.
candidates = [
    ("Logistic Regression",
     LogisticRegression(max_iter=1000, random_state=42),
     {"clf__C": [0.1, 1, 10], "clf__class_weight": ["balanced"]}),

    ("Linear SVM",
     LinearSVC(max_iter=5000, dual=False, random_state=42),
     {"clf__C": [0.1, 1, 10], "clf__class_weight": ["balanced"]}),

    ("Decision Tree",
     DecisionTreeClassifier(random_state=42),
     {"clf__max_depth": [10, 20, None], "clf__min_samples_leaf": [5],
      "clf__class_weight": ["balanced"]}),

    ("Random Forest",
     RandomForestClassifier(random_state=42),
     {"clf__n_estimators": [100], "clf__max_depth": [10, 20],
      "clf__class_weight": ["balanced"]}),

    ("Neural Network (MLP)",
     MLPClassifier(max_iter=300, early_stopping=True, random_state=42),
     {"clf__hidden_layer_sizes": [(64, 32)], "clf__alpha": [0.001]}),
]

results = []
best_overall = None  # (recall, f1, name, fitted_pipeline)

print(f"\n{'='*70}\nFinal tuning pass - all 5 models, corrected pipeline\n{'='*70}")

for name, model, param_grid in candidates:
    print(f"\n--- {name} ---")

    # Preprocessing + classifier as ONE pipeline, fit inside each CV fold.
    # This also fixes issue #2 Aboud flagged (preprocessing previously fit
    # once outside cross-validation) for whichever model wins here.
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("clf", model)])

    search = GridSearchCV(pipe, param_grid, cv=5, scoring="recall", n_jobs=1)
    search.fit(X_train_raw, y_train)

    best_pipe = search.best_estimator_
    y_pred = best_pipe.predict(X_test_raw)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"Best params: {search.best_params_}")
    print(f"Accuracy: {acc:.3f}  Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}")
    print(f"Confusion matrix [[TN FP] [FN TP]]:\n{confusion_matrix(y_test, y_pred)}")

    results.append({"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1})

    if best_overall is None or (rec, f1) > (best_overall[0], best_overall[1]):
        best_overall = (rec, f1, name, best_pipe)

print(f"\n{'='*70}\nFinal comparison (sorted by recall)\n{'='*70}")
for r in sorted(results, key=lambda r: r["recall"], reverse=True):
    print(f"{r['model']:<22} acc={r['accuracy']:.3f}  prec={r['precision']:.3f}  "
          f"recall={r['recall']:.3f}  f1={r['f1']:.3f}")

winner_recall, winner_f1, winner_name, winner_pipe = best_overall
print(f"\nSELECTED FOR DEPLOYMENT: {winner_name} (recall={winner_recall:.3f}, f1={winner_f1:.3f})")

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_model.pkl")
with open(out_path, "wb") as f:
    pickle.dump({"pipeline": winner_pipe, "model_name": winner_name}, f)
print(f"Saved winning pipeline to: {out_path}")
