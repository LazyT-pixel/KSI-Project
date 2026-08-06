"""
Part 2 - Ibrahim's task: tune Decision Tree and Random Forest.

WHAT TO DO (please do all steps, don't skip to the end):

STEP 1 - Before running anything, write one or two sentences predicting
what you expect:
  "max_depth" limits how deep a tree can grow. A shallow tree (small number,
  like 5) is simpler and might miss patterns (underfit). A deep tree
  (large number, or None = unlimited) can fit the training data very
  closely, which sometimes means it does worse on new data (overfit).
  "n_estimators" (Random Forest only) is how many trees get combined -
  more trees is usually more stable but slower to train.
  Question: do you expect a deeper tree to have higher or lower recall
  than a shallow one? Just guess - you'll check yourself in Step 5.

STEP 2 - Fill in the TODO grids below yourself (search for "TODO"). Pick
your own values for max_depth, min_samples_leaf, and n_estimators - the
comments next to each explain what they control and give a reasonable
range to pick from.

STEP 3 - Open a terminal in the main project folder (the one with KSI.py
in it), then run: python3 part2/tune_ibrahim.py

STEP 4 - It prints the "best settings" found for each model, and how well
each model does on the test data.

STEP 5 - Write a short summary (a few sentences):
   - Which of the two (Decision Tree or Random Forest) did better?
   - Did tuning improve recall vs. the untuned baseline we already ran?
   - Random Forest had very high precision but low recall in the baseline
     run - does tuning change that trade-off at all?
   - Was your Step 1 guess right or wrong?

STEP 6 - Send back your Step 1 guess, your Step 5 write-up, and everything
the script printed.

There's no single "correct" grid here - pick values yourself and see what
actually happens, don't just copy someone else's numbers.
"""

import os
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

# lets Python find KSI.py in the main folder, one level up from part2/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import KSI  # runs the Part 1 data pipeline, gives us clean train/test data

X_train = KSI.X_train_processed
X_test = KSI.X_test_processed
y_train = KSI.y_train
y_test = KSI.y_test


def tune_and_report(name, model, param_grid):
    print(f"\n{'='*60}\nTuning: {name}\n{'='*60}")

    # scoring="recall" because catching real fatal cases matters more than
    # raw accuracy for this project.
    search = GridSearchCV(model, param_grid, cv=3, scoring="recall", n_jobs=1)
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


# TODO: pick your own values (delete the empty lists and fill in numbers).
# max_depth: try a few between 3 and 30, plus None (unlimited) if you want.
#   Example FORMAT (not the values to use): [5, 10, 20, None]
tree_max_depth = []          # <-- fill this in
# min_samples_leaf: try a few small numbers, e.g. between 1 and 20.
tree_min_samples_leaf = []   # <-- fill this in

if not tree_max_depth or not tree_min_samples_leaf:
    raise ValueError(
        "tree_max_depth / tree_min_samples_leaf is empty! Fill in the TODOs "
        "above with your own numbers first (see STEP 1 and STEP 2 at the top "
        "of this file)."
    )

# --- Model 1: Decision Tree ---
tree_grid = {
    "max_depth": tree_max_depth,
    "min_samples_leaf": tree_min_samples_leaf,
    "class_weight": ["balanced"],
}
tune_and_report("Decision Tree", DecisionTreeClassifier(random_state=42), tree_grid)

# TODO: pick your own values for the forest too.
# n_estimators: number of trees - try a couple values between 50 and 300.
forest_n_estimators = []     # <-- fill this in
# max_depth: same idea as the tree above.
forest_max_depth = []        # <-- fill this in

if not forest_n_estimators or not forest_max_depth:
    raise ValueError(
        "forest_n_estimators / forest_max_depth is empty! Fill in the TODOs "
        "above with your own numbers first."
    )

# --- Model 2: Random Forest ---
forest_grid = {
    "n_estimators": forest_n_estimators,
    "max_depth": forest_max_depth,
    "class_weight": ["balanced"],
}
# Note: RandomForestClassifier itself isn't set to n_jobs=-1 here (only the
# GridSearchCV around it is) - running both in parallel at once fights over
# CPU cores and actually makes this slower, not faster.
tune_and_report("Random Forest", RandomForestClassifier(random_state=42), forest_grid)

print("\nDone. Copy everything above and send it back with your write-up.")
