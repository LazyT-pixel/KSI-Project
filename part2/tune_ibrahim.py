"""
Part 2 - Ibrahim's task: tune Decision Tree and Random Forest.

WHAT TO DO:
1. Open a terminal in the main project folder (the one with KSI.py in it),
   then run: python3 part2/tune_ibrahim.py
2. It prints the "best settings" found for each model, and how well each
   model does on the test data.
3. Write a short summary (a few sentences):
   - Which of the two did better?
   - Did tuning improve recall vs. the untuned baseline we already ran?
   - Random Forest had very high precision but low recall in the baseline
     run - does tuning change that trade-off at all?
4. Send back your write-up + the printed results.

Feel free to expand the "param_grid" values below if you want to search a
wider range - not required, current ranges are reasonable.
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


# --- Model 1: Decision Tree ---
tree_grid = {
    "max_depth": [5, 10, 20, None],        # limits how deep the tree grows; None = no limit
    "min_samples_leaf": [1, 5, 10],        # minimum data points needed in a leaf
    "class_weight": ["balanced"],
}
tune_and_report("Decision Tree", DecisionTreeClassifier(random_state=42), tree_grid)

# --- Model 2: Random Forest ---
forest_grid = {
    "n_estimators": [100, 150],            # number of trees in the forest
    "max_depth": [10, 20],
    "class_weight": ["balanced"],
}
# Note: RandomForestClassifier itself isn't set to n_jobs=-1 here (only the
# GridSearchCV around it is) - running both in parallel at once fights over
# CPU cores and actually makes this slower, not faster.
tune_and_report("Random Forest", RandomForestClassifier(random_state=42), forest_grid)

print("\nDone. Copy everything above and send it back with your write-up.")
