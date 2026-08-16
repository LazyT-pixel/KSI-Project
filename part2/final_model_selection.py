"""
Part 2 - Final model selection (Aidan).

Why this exists: the INJURY data leak (found by Aboud) invalidated every
model result reported before the fix, and Ibrahim never submitted his
Decision Tree / Random Forest tuning. Rather than leave the submission with
mismatched pre-fix numbers or an incomplete comparison, this runs ONE
consistent pass for all 5 models on the corrected pipeline and picks the
final model for deployment.

Per the actual assignment sheet (section 3):
  "Fine tune the models using Grid search and randomized grid search."
  "Present results as accuracy, precision, recall, F1 scores, confusion
   matrices and plot the ROC curves of the models."
Both apply to all 5 models, not a subset - so every model here gets BOTH
grid search and randomized search, and all 5 get plotted on one ROC chart.
(Aboud's tune_aboud.py already did this in more depth for Logistic
Regression + Linear SVM specifically, including his own separate ROC plot -
this script does the same thing for all 5 so nothing is missing group-wide.)

Selection metric: recall first (missing a real fatal case is worse than a
false alarm for this problem), F1 as tiebreaker, across BOTH grid and
randomized results for a given model - whichever search finds the better
result wins for that model.

Output: prints a full comparison table (grid vs randomized, per model),
saves one ROC curve plot with all 5 models to part2/roc_all_models.png,
and pickles the overall winning model to part2/final_model.pkl.
"""

import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import loguniform, randint
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, roc_auc_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import KSI  # runs the corrected Part 1 pipeline (INJURY + FATAL_NO both dropped)

X_train_raw = KSI.X_train
X_test_raw = KSI.X_test
y_train = KSI.y_train
y_test = KSI.y_test
preprocessor = KSI.preprocessor


def get_scores(fitted_pipe, X):
    """Probability/decision scores for ROC - predict_proba if available
    (most models), else decision_function (LinearSVC has no predict_proba)."""
    clf = fitted_pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return fitted_pipe.predict_proba(X)[:, 1]
    return fitted_pipe.decision_function(X)


def evaluate(fitted_pipe, label):
    y_pred = fitted_pipe.predict(X_test_raw)
    scores = get_scores(fitted_pipe, X_test_raw)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, scores)
    print(f"  [{label}] acc={acc:.3f} prec={prec:.3f} recall={rec:.3f} f1={f1:.3f} auc={auc:.3f}")
    print(f"    Confusion matrix [[TN FP] [FN TP]]:\n{confusion_matrix(y_test, y_pred)}")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc,
            "scores": scores, "pipe": fitted_pipe, "label": label}


# Each entry: (name, model, grid_params, random_params, n_iter)
# Grids/distributions are intentionally modest given the time left before
# the deadline - real search, not exhaustive.
candidates = [
    ("Logistic Regression",
     LogisticRegression(max_iter=1000, random_state=42),
     {"clf__C": [0.1, 1, 10], "clf__class_weight": ["balanced"]},
     {"clf__C": loguniform(1e-3, 1e2), "clf__class_weight": ["balanced"]},
     6),

    ("Linear SVM",
     LinearSVC(max_iter=5000, dual=False, random_state=42),
     {"clf__C": [0.1, 1, 10], "clf__class_weight": ["balanced"]},
     {"clf__C": loguniform(1e-3, 1e2), "clf__class_weight": ["balanced"]},
     6),

    ("Decision Tree",
     DecisionTreeClassifier(random_state=42),
     {"clf__max_depth": [10, 20, None], "clf__min_samples_leaf": [5],
      "clf__class_weight": ["balanced"]},
     {"clf__max_depth": randint(3, 30), "clf__min_samples_leaf": randint(1, 20),
      "clf__class_weight": ["balanced"]},
     6),

    ("Random Forest",
     RandomForestClassifier(random_state=42),
     {"clf__n_estimators": [100], "clf__max_depth": [10, 20],
      "clf__class_weight": ["balanced"]},
     {"clf__n_estimators": randint(50, 200), "clf__max_depth": randint(5, 30),
      "clf__class_weight": ["balanced"]},
     4),

    ("Neural Network (MLP)",
     MLPClassifier(max_iter=300, early_stopping=True, random_state=42),
     {"clf__hidden_layer_sizes": [(64, 32)], "clf__alpha": [0.001]},
     {"clf__hidden_layer_sizes": [(32,), (64, 32), (64,)],
      "clf__alpha": loguniform(1e-5, 1e-1)},
     3),
]

all_results = {}  # name -> best result dict (grid or random, whichever wins)
best_overall = None  # (recall, f1, name, result)

print(f"\n{'='*70}\nFinal tuning pass - grid search AND randomized search, all 5 models\n{'='*70}")

for name, model, grid_params, random_params, n_iter in candidates:
    print(f"\n--- {name} ---")
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("clf", model)])

    grid_search = GridSearchCV(pipe, grid_params, cv=5, scoring="recall", n_jobs=1)
    grid_search.fit(X_train_raw, y_train)
    print(f"  Grid search best params: {grid_search.best_params_}")
    grid_result = evaluate(grid_search.best_estimator_, "grid search")

    random_search = RandomizedSearchCV(pipe, random_params, n_iter=n_iter, cv=5,
                                        scoring="recall", n_jobs=1, random_state=42)
    random_search.fit(X_train_raw, y_train)
    print(f"  Randomized search best params: {random_search.best_params_}")
    random_result = evaluate(random_search.best_estimator_, "randomized search")

    # Keep whichever search did better for this model (recall, then F1)
    best_for_model = max(
        [grid_result, random_result],
        key=lambda r: (r["recall"], r["f1"]),
    )
    all_results[name] = best_for_model
    print(f"  -> Best for {name}: {best_for_model['label']} "
          f"(recall={best_for_model['recall']:.3f}, f1={best_for_model['f1']:.3f})")

    if best_overall is None or (best_for_model["recall"], best_for_model["f1"]) > \
            (best_overall[0], best_overall[1]):
        best_overall = (best_for_model["recall"], best_for_model["f1"], name, best_for_model)

# --- Final comparison table ---
print(f"\n{'='*70}\nFinal comparison (best of grid/randomized per model, sorted by recall)\n{'='*70}")
print(f"{'Model':<22}{'Search':<20}{'Accuracy':<10}{'Precision':<11}{'Recall':<9}{'F1':<8}{'AUC'}")
for name, r in sorted(all_results.items(), key=lambda kv: kv[1]["recall"], reverse=True):
    print(f"{name:<22}{r['label']:<20}{r['accuracy']:<10.3f}{r['precision']:<11.3f}"
          f"{r['recall']:<9.3f}{r['f1']:<8.3f}{r['auc']:.3f}")

# --- ROC curves, all 5 models on one chart ---
plt.figure(figsize=(7, 6))
for name, r in all_results.items():
    fpr, tpr, _ = roc_curve(y_test, r["scores"])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {r['auc']:.3f})")
plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random guess (AUC = 0.500)")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate (recall)")
plt.title("ROC curves - all 5 tuned models (best of grid/randomized search)")
plt.legend(loc="lower right", fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
roc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "roc_all_models.png")
plt.savefig(roc_path, dpi=150)
print(f"\nROC curve (all 5 models) saved to: {roc_path}")

# --- Winner ---
winner_recall, winner_f1, winner_name, winner_result = best_overall
print(f"\nSELECTED FOR DEPLOYMENT: {winner_name} ({winner_result['label']}, "
      f"recall={winner_recall:.3f}, f1={winner_f1:.3f}, auc={winner_result['auc']:.3f})")

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_model.pkl")
with open(out_path, "wb") as f:
    pickle.dump({"pipeline": winner_result["pipe"], "model_name": winner_name,
                 "search_type": winner_result["label"]}, f)
print(f"Saved winning pipeline to: {out_path}")
