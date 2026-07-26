"""
KSI Data Modelling Pipeline
Part 1 deliverable #2: cleaning, encoding, train/test split, imbalance handling, Pipeline.

Column decisions finalized based on the exploration findings in notebooks 01-03
(Group 1/2 notebooks were completed as drafts when those group members did not
submit their own analysis in time; decisions below follow directly from what
that analysis found).
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
# 1b. Feature engineering: HOUR from TIME
# ---------------------------------------------------------------------------
# TIME is stored as HHMM without leading zeros (e.g. 236 = 2:36 AM), not usable
# directly. Group 1's exploration found a real pattern here (5 AM spikes to a
# 26.6% fatal rate vs ~14% baseline; congested 3-5 PM afternoon hours run lower
# despite heavier traffic) so it's worth keeping as an engineered feature
# rather than dropping TIME entirely.
df["HOUR"] = (df["TIME"] // 100).clip(0, 23)

# ---------------------------------------------------------------------------
# 2. Drop low-value / identifier columns (from glossary review)
# ---------------------------------------------------------------------------
drop_cols = [
    "INDEX", "ACCNUM", "OBJECTID", "OFFSET",
    "HOOD_140", "NEIGHBOURHOOD_140",  # older duplicates of the _158 versions
    "x", "y",  # projected copies of LATITUDE/LONGITUDE, redundant
    "FATAL_NO",  # DATA LEAKAGE: only populated when the person died — its
                 # presence directly reveals the target, must never be a feature
    "STREET1", "STREET2",  # too high-cardinality (1,942 / 2,821 unique) to
                            # encode sensibly; DISTRICT/NEIGHBOURHOOD_158 already
                            # capture location at a usable granularity
    "DATE", "TIME",  # replaced by the engineered HOUR feature below
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
# 4. Column groups — finalized from the exploration findings (notebooks 01-03)
#
#   - Group 1 (location/conditions): kept the low-cardinality, mostly-complete
#     columns. STREET1/STREET2/DATE/TIME dropped above (too high-cardinality,
#     or replaced by HOUR). LATITUDE/LONGITUDE kept as numeric.
#   - Group 2 (people/vehicles): PEDTYPE/PEDACT/PEDCOND and CYCLISTYPE/CYCACT/
#     CYCCOND are dropped here — they're 83-96% missing *by design* (only
#     apply to pedestrian/cyclist rows respectively, not randomly missing),
#     and Group 3's PEDESTRIAN/CYCLIST flags already capture whether those
#     involvement types were present. Including them would mean imputing
#     nonsense values onto rows they don't apply to.
#   - Group 3 (Yes/No flags): clean (Yes/blank), low cardinality, one-hot as-is.
# ---------------------------------------------------------------------------
categorical_cols = [
    # Group 1 — location/conditions
    "ROAD_CLASS", "DISTRICT", "ACCLOC", "TRAFFCTL", "VISIBILITY",
    "LIGHT", "RDSFCOND", "DIVISION", "NEIGHBOURHOOD_158",
    # Group 2 — people/vehicles (pedestrian/cyclist-only columns excluded, see above)
    "INVTYPE", "INVAGE", "INJURY", "INITDIR", "VEHTYPE", "MANOEUVER",
    "DRIVACT", "DRIVCOND",
    # Group 3 — Yes/No flags
    "PEDESTRIAN", "CYCLIST", "AUTOMOBILE", "MOTORCYCLE", "TRUCK",
    "TRSN_CITY_VEH", "EMERG_VEH", "PASSENGER", "SPEEDING", "AG_DRIV",
    "REDLIGHT", "ALCOHOL", "DISABILITY",
]
numeric_cols = ["LATITUDE", "LONGITUDE", "HOUR"]

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