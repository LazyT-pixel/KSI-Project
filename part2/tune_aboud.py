"""
Part 2 - Aboud's task: tune Logistic Regression and Linear SVM.

WHAT TO DO:
1. Open a terminal in the main project folder (the one with KSI.py in it),
   then run: python3 part2/tune_aboud.py
2. It will print the "best settings" it found for each model, and how well
   each model did on the test data.
3. Copy the printed results.
4. Write 3-4 sentences about what you see. Example questions to answer:
   - Which of the two models (Logistic Regression or Linear SVM) did better?
   - Did tuning improve "recall" compared to the baseline numbers we already
     have? (Recall = how many of the real fatal accidents we correctly caught.)
5. Send me your 3-4 sentences + the printed results.

YOU CAN CHANGE THE NUMBERS in "param_grid" below if you want to try more
combinations, but you don't have to - the current lists already test a
reasonable range.
"""

import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

# lets Python find KSI.py in the main folder, one level up from part2/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import KSI  # this runs the Part 1 data pipeline and gives us clean train/test data

X_train = KSI.X_train_processed
X_test = KSI.X_test_processed
y_train = KSI.y_train
y_test = KSI.y_test


def tune_and_report(name, model, param_grid):
    print(f"\n{'='*60}\nTuning: {name}\n{'='*60}")

    # cv=5 means it tries every combination 5 times on different slices of
    # the training data, and averages the score - this avoids getting lucky
    # or unlucky with one split.
    # scoring="recall" because for this project, catching real fatal cases
    # matters more than raw accuracy.
    search = GridSearchCV(model, param_grid, cv=5, scoring="recall", n_jobs=-1)
    search.fit(X_train, y_train)

    print("Best settings found:", search.best_params_)
    print(f"Best cross-validation recall: {search.best_score_:.3f}")

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\nResults on the held-out test set (data the model never saw):")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"  Recall:    {recall_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"  F1 score:  {f1_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"  Confusion matrix [[TN FP] [FN TP]]:\n{confusion_matrix(y_test, y_pred)}")

    return best_model


# --- Model 1: Logistic Regression ---
log_reg_grid = {
    "C": [0.01, 0.1, 1, 10],          # smaller = simpler model, larger = fits data more closely
    "class_weight": ["balanced"],      # keep this - it helps catch more fatal cases
}
tune_and_report("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42), log_reg_grid)

# --- Model 2: Linear SVM ---
svm_grid = {
    "C": [0.01, 0.1, 1, 10],
    "class_weight": ["balanced"],
}
tune_and_report("Linear SVM", LinearSVC(max_iter=5000, random_state=42), svm_grid)

print("\nDone. Copy everything above and send it back with your 3-4 sentence write-up.")
