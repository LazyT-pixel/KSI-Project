import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
import KSI
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline(steps=[
    ("preprocessor", KSI.preprocessor),
    ("clf", LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced", random_state=42)),
])
pipe.fit(KSI.X_train, KSI.y_train)

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_model.pkl")
with open(out_path, "wb") as f:
    pickle.dump({"pipeline": pipe, "model_name": "Logistic Regression", "search_type": "grid search"}, f)
print("Re-saved final_model.pkl with your local scikit-learn version")