"""
Part 2 - Aboud's task: tune Logistic Regression and Linear SVM.

WHAT TO DO (please do all steps, don't skip to the end):

STEP 1 - Before you run anything, answer this for yourself (one sentence,
write it down, you'll send it with your results):
  "C" controls how closely the model fits the training data.
  A SMALL C (like 0.01) = simpler model, may underfit (miss real patterns).
  A LARGE C (like 100) = fits the training data very closely, may overfit
  (memorizes the training data instead of learning general patterns, and
  does worse on new/unseen data).
  Question: what do you think will happen to accuracy vs. recall if C is
  very small compared to very large? Just guess, that's the point.

STEP 2 - Fill in the "C_values" list below (search for "TODO"). Pick at
least 4 values: something small (0.001-0.01), something medium (0.1-1),
something large (10-100), and one more of your choice. You choose the
exact numbers.

STEP 3 - Open a terminal in the main project folder (the one with KSI.py
in it), then run: python3 part2/tune_aboud.py

STEP 4 - It prints the "best settings" it found for each model, and how
well each model did on the test data.

STEP 5 - Write 3-4 sentences about what you see:
   - Which of the two models (Logistic Regression or Linear SVM) did better?
   - Did tuning improve "recall" compared to the baseline numbers we already
     have? (Recall = how many of the real fatal accidents we correctly caught.)
   - Was your guess from Step 1 right or wrong? Why do you think that is?

STEP 6 - Send me: your Step 1 guess, your Step 5 write-up, and everything
the script printed.

This isn't a trick - there's no single "correct" list of C values. The
point is picking them yourself and seeing what happens, not just running
someone else's numbers.
"""

# ---------------------------------------------------------------------------
# Aboud - my notes on what I changed and why:
#
# 1. Picked 6 C values spanning 0.001 to 100 (five orders of magnitude) so the
#    trend is visible instead of just two or three points.
# 2. Added RandomizedSearchCV as a second, separate search. The assignment
#    (section 3) asks for grid search AND randomized grid search, and we only
#    had grid search. It samples C from a continuous log-uniform range, so it
#    can land on values between my fixed grid points. random_state=42 so it
#    gives the same answer every time it runs.
# 3. Added ROC curves + AUC, which the assignment asks for in section 4.
#    LinearSVC has no predict_proba, so I used decision_function() for both
#    models to keep the two curves comparable.
# 4. Fixed the LinearSVC convergence warning properly instead of hiding it -
#    see the comment above the LinearSVC model at the bottom.
# 5. Scoring is still recall (unchanged), but I now collect accuracy,
#    precision and F1 during the same cross-validation so I can show the
#    accuracy-vs-recall trade-off per C without touching the test set.
#
# IMPORTANT - treat every number below as preliminary. While working on this
# I found three data problems that will change these results once we fix them
# (INJURY leaks the target, the preprocessing is fitted outside the CV folds,
# and the same collision appears in both train and test). Written up
# separately - we need to decide as a group before these numbers are final.
# ---------------------------------------------------------------------------

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save the plot to a file instead of opening a window
import matplotlib.pyplot as plt
from scipy.stats import loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, roc_auc_score,
)

# lets Python find KSI.py in the main folder, one level up from part2/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import KSI  # this runs the Part 1 data pipeline and gives us clean train/test data

X_train = KSI.X_train_processed
X_test = KSI.X_test_processed
y_train = KSI.y_train
y_test = KSI.y_test

# Both searches only ever see X_train / y_train. X_test is untouched until the
# final scoring step, and it is never used to choose C.
print(f"\nTraining data: {X_train.shape[0]} samples x {X_train.shape[1]} features")
print(f"Test data:     {X_test.shape[0]} samples (only used for final scoring)")

# We ask the CV to record all four metrics, but selection is still done on
# recall alone (refit="recall") so the chosen model is the same as before.
SCORING = {
    "recall": "recall",
    "accuracy": "accuracy",
    "precision": "precision",
    "f1": "f1",
}


def convergence_status(model):
    """Report how many iterations the solver actually needed.

    If this comes back equal to max_iter, the model stopped early and did not
    really converge - that is what the sklearn warning is about.
    """
    n_iter = getattr(model, "n_iter_", None)
    if n_iter is None:
        return "n/a"
    used = int(np.max(n_iter))
    limit = getattr(model, "max_iter", None)
    if limit is not None and used >= limit:
        return f"{used} iterations - DID NOT CONVERGE (hit max_iter={limit})"
    return f"{used} iterations - converged (max_iter={limit})"


def print_c_sweep(search):
    """Show every C that was tried and how it scored in cross-validation.

    This is the table that answers STEP 1: what happens to accuracy vs recall
    as C goes from very small to very large. All of it comes from the training
    folds - the test set is not involved.
    """
    res = search.cv_results_
    cs = np.array([p["C"] for p in res["params"]], dtype=float)
    print("\n  Cross-validation scores for each C (training data only):")
    print(f"    {'C':>10} {'accuracy':>10} {'precision':>10} {'recall':>10} {'f1':>10}")
    for i in np.argsort(cs):
        print(f"    {cs[i]:>10.5g} {res['mean_test_accuracy'][i]:>10.3f} "
              f"{res['mean_test_precision'][i]:>10.3f} "
              f"{res['mean_test_recall'][i]:>10.3f} {res['mean_test_f1'][i]:>10.3f}")


def evaluate_on_test(name, best_model, search_label):
    """Score the winning model on the held-out test set."""
    y_pred = best_model.predict(X_test)
    scores = best_model.decision_function(X_test)  # works for LinearSVC too
    auc = roc_auc_score(y_test, scores)

    print(f"\n  Results on the held-out test set ({name}, {search_label}):")
    print(f"    Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"    Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"    Recall:    {recall_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"    F1 score:  {f1_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"    ROC AUC:   {auc:.3f}")
    print(f"    Confusion matrix [[TN FP] [FN TP]]:\n{confusion_matrix(y_test, y_pred)}")
    print(f"    Solver:    {convergence_status(best_model)}")
    return scores, auc


def tune_and_report(name, model, param_grid):
    """Grid search: try every C in the list I picked."""
    print(f"\n{'='*70}\nGRID SEARCH - {name}\n{'='*70}")

    # cv=5 means it tries every combination 5 times on different slices of
    # the training data, and averages the score - this avoids getting lucky
    # or unlucky with one split.
    # refit="recall" because for this project, catching real fatal cases
    # matters more than raw accuracy.
    search = GridSearchCV(model, param_grid, cv=5, scoring=SCORING,
                          refit="recall", n_jobs=-1)
    search.fit(X_train, y_train)

    print("Best settings found:", search.best_params_)
    print(f"Best cross-validation recall: {search.best_score_:.3f}")
    print_c_sweep(search)

    best_model = search.best_estimator_
    scores, auc = evaluate_on_test(name, best_model, "grid search")
    return best_model, scores, auc


def random_search_and_report(name, model, param_dist, n_iter=20, seed=42):
    """Randomized search: sample C from a continuous range instead of a fixed list.

    The grid search can only ever pick one of my 6 numbers. This one draws
    n_iter random values from a log-uniform range between 0.0001 and 1000, so
    it can find good values sitting between my grid points. random_state makes
    the draw identical every run.
    """
    print(f"\n{'='*70}\nRANDOMIZED SEARCH - {name} ({n_iter} random C values)\n{'='*70}")

    search = RandomizedSearchCV(model, param_dist, n_iter=n_iter, cv=5,
                                scoring=SCORING, refit="recall",
                                n_jobs=-1, random_state=seed)
    search.fit(X_train, y_train)

    best_c = search.best_params_["C"]
    print(f"Best C found: {best_c:.6g}")
    print(f"Best cross-validation recall: {search.best_score_:.3f}")

    best_model = search.best_estimator_
    evaluate_on_test(name, best_model, "randomized search")
    return search.best_score_, best_c


# TODO: pick at least 4 values yourself - one small, one medium, one large,
# and one more of your choice. Delete the placeholder and put your own numbers.
# Example of the FORMAT (not the values to use): C_values = [0.01, 0.1, 1, 10]
C_values = [0.001, 0.01, 0.1, 1, 10, 100]

if not C_values:
    raise ValueError(
        "C_values is empty! Go fill in the TODO above with your own numbers "
        "before running this (see STEP 1 and STEP 2 in the instructions at "
        "the top of this file)."
    )

# For the randomized search - a continuous range rather than a fixed list.
C_distribution = loguniform(1e-4, 1e3)

# --- Model 1: Logistic Regression ---
# max_iter=1000 is enough here: I checked the worst points before running -
# C=100 needs 432 iterations and C=1000 (top of the randomized range) needs
# 332, so nothing hits the limit. The script prints the real iteration count
# after each fit so this can be checked rather than assumed.
log_reg_grid = {
    "C": C_values,
    "class_weight": ["balanced"],      # keep this - it helps catch more fatal cases
}
log_reg_best, log_reg_scores, log_reg_auc = tune_and_report(
    "Logistic Regression", LogisticRegression(max_iter=1000, random_state=42), log_reg_grid
)
random_search_and_report(
    "Logistic Regression", LogisticRegression(max_iter=1000, random_state=42),
    {"C": C_distribution, "class_weight": ["balanced"]},
)

# --- Model 2: Linear SVM ---
# dual=False fixes the convergence warning this model was giving.
# LinearSVC can solve either the "dual" or the "primal" version of the same
# problem. The dual one is meant for data with more features than samples;
# we have the opposite (15,164 samples vs 373 features), so the dual solver
# was still not converged after 5,000 iterations. Switching to the primal
# solver with dual=False converges in about 16 iterations and is ~7x faster.
# Raising max_iter on its own does NOT fix this, it just runs longer.
svm_grid = {
    "C": C_values,
    "class_weight": ["balanced"],
}
svm_best, svm_scores, svm_auc = tune_and_report(
    "Linear SVM", LinearSVC(max_iter=5000, dual=False, random_state=42), svm_grid
)
random_search_and_report(
    "Linear SVM", LinearSVC(max_iter=5000, dual=False, random_state=42),
    {"C": C_distribution, "class_weight": ["balanced"]},
)

# --- ROC curves for both tuned models (assignment section 4) ---
plt.figure(figsize=(7, 6))
plt.plot(*roc_curve(y_test, log_reg_scores)[:2],
         label=f"Logistic Regression (AUC = {log_reg_auc:.3f})")
plt.plot(*roc_curve(y_test, svm_scores)[:2],
         label=f"Linear SVM (AUC = {svm_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random guess (AUC = 0.500)")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate (recall)")
plt.title("ROC curves - tuned Logistic Regression vs Linear SVM")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()

plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "roc_aboud.png")
plt.savefig(plot_path, dpi=150)
print(f"\nROC curve saved to: {plot_path}")

print("\nDone. Copy everything above and send it back with your 3-4 sentence write-up.")
