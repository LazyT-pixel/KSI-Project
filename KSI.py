"""
KSI Data Modelling — Starter Pipeline Skeleton
Part 1 deliverable #2: cleaning, encoding, train/test split, imbalance handling, Pipeline.

Fill in the column lists / choices based on what Groups 1-3 found during exploration
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("data/TOTAL_KSI_4115794401574937330.csv", encoding="utf-8-sig")

# ---------------------------------------------------------------------------
# 2. Drop low-value / identifier columns (from glossary review)
# ---------------------------------------------------------------------------
drop_cols = [
    "INDEX", "ACCNUM", "OBJECTID", "OFFSET",
    "HOOD_140", "NEIGHBOURHOOD_140",  # older duplicates of the _158 versions
    "x", "y",  # projected copies of LATITUDE/LONGITUDE, redundant
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# ---------------------------------------------------------------------------
# 3. Define target
# ---------------------------------------------------------------------------
# ACCLASS actually has 4 values in the real data (confirmed from the downloaded CSV):
#   'Fatal', 'Non-Fatal Injury', 'Property Damage O' (truncated -> "Property Damage Only"), '' (blank)
#
# - Blank ACCLASS rows have no label -> drop them, can't train on unlabeled data.
# - "Property Damage Only" rows exist because this dataset is per-PERSON, not per-collision:
#   one KSI collision can still have a driver/passenger who personally only had property
#   damage / no injury. We're treating anything that isn't "Fatal" as the negative class.
#   (Flag this as an assumption in the report — worth double-checking with the group.)
target_col = "ACCLASS"

df = df[df[target_col].notna() & (df[target_col].str.strip() != "")]
df[target_col] = df[target_col].apply(lambda x: 1 if str(x).strip().lower() == "fatal" else 0)

y = df[target_col]
X = df.drop(columns=[target_col])

# ---------------------------------------------------------------------------
# 4. Column groups
#
# TODO (waiting on group findings, don't finalize yet):
#   - Group 1 (Aboud) and Group 2 (Ibrahim) haven't submitted their exploration
#     findings yet. The lists below are PLACEHOLDERS based on the raw column
#     list only — once they report missing-data % and cardinality per column,
#     come back and drop/adjust anything that's mostly empty or unusably high
#     cardinality (e.g. free-text-like fields).
#   - Group 3 (Aidan) is the one section here I've actually reviewed against
#     the real data — the Yes/No flags are clean (Yes/blank), low cardinality,
#     safe to one-hot as-is.
# ---------------------------------------------------------------------------
categorical_cols = [
    # Group 1 (location/conditions) — PLACEHOLDER, pending Aboud's findings
    "ROAD_CLASS", "DISTRICT", "ACCLOC", "TRAFFCTL", "VISIBILITY",
    "LIGHT", "RDSFCOND", "DIVISION", "NEIGHBOURHOOD_158",
    # Group 2 (people/vehicles) — PLACEHOLDER, pending Ibrahim's findings
    "INVTYPE", "INVAGE", "INJURY", "INITDIR", "VEHTYPE", "MANOEUVER",
    "DRIVACT", "DRIVCOND", "PEDTYPE", "PEDACT", "PEDCOND",
    "CYCLISTYPE", "CYCACT", "CYCCOND",
    # Group 3 (Yes/No flags) — reviewed against real data, values are "Yes"/blank
    "PEDESTRIAN", "CYCLIST", "AUTOMOBILE", "MOTORCYCLE", "TRUCK",
    "TRSN_CITY_VEH", "EMERG_VEH", "PASSENGER", "SPEEDING", "AG_DRIV",
    "REDLIGHT", "ALCOHOL", "DISABILITY",
]
numeric_cols = ["LATITUDE", "LONGITUDE"]  # add more if kept (e.g. engineered hour/month from DATE/TIME)

# keep only columns that actually exist in X (avoids KeyErrors if some got dropped earlier)
categorical_cols = [c for c in categorical_cols if c in X.columns]
numeric_cols = [c for c in numeric_cols if c in X.columns]

# ---------------------------------------------------------------------------
# 5. Preprocessing pipeline
# ---------------------------------------------------------------------------
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

preprocessor = ColumnTransformer(transformers=[
    ("cat", categorical_transformer, categorical_cols),
    ("num", numeric_transformer, numeric_cols),
])

# ---------------------------------------------------------------------------
# 6. Train/test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------------------
# 7. Handle class imbalance
#    Fatal collisions are almost certainly the minority class.
#    Option A: class_weight="balanced" in the eventual model (simplest, do at model-building stage)
#    Option B: SMOTE / oversampling here with imbalanced-learn (pip install imbalanced-learn)
# ---------------------------------------------------------------------------
# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline as ImbPipeline
# full_pipeline = ImbPipeline(steps=[
#     ("preprocessor", preprocessor),
#     ("smote", SMOTE(random_state=42)),
# ])

full_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
])

# Fit-transform on train, transform on test (no fitting on test data — avoids leakage)
X_train_processed = full_pipeline.fit_transform(X_train, y_train)
X_test_processed = full_pipeline.transform(X_test)

print("Train shape:", X_train_processed.shape)
print("Test shape:", X_test_processed.shape)
print("Class balance (train):", y_train.value_counts(normalize=True))

# Next step (Week 14 / Part 2): plug full_pipeline into a model-building script
# e.g. LogisticRegression(class_weight="balanced"), RandomForestClassifier, etc.