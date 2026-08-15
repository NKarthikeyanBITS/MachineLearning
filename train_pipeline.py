import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# ==========================================
# 1. LOAD DATASET
# ==========================================
# REPLACE 'your_dataset.csv' with your actual downloaded Kaggle/UCI dataset filename
RAW_DATA_PATH = "credit_card_fraud_2026.csv" 
TARGET_COLUMN = "is_fraud"  # CHANGE THIS to the exact name of your target/label column

if not os.path.exists(RAW_DATA_PATH):
    raise FileNotFoundError(f"Please place your dataset file at '{RAW_DATA_PATH}' before running.")

df = pd.read_csv(RAW_DATA_PATH)

# ==========================================
# 2. DATA EXPLORATION (Console Summary)
# ==========================================
print("--- STARTING DATA EXPLORATION ---")
print(f"Initial Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\n--- Missing Value Counts ---")
print(df.isnull().sum()[df.isnull().sum() > 0])
print("\n--- Data Types Summary ---")
print(df.dtypes.value_counts())

# ==========================================
# 3. DATA CLEANUP & PREPROCESSING
# ==========================================
print("\n--- STARTING DATA CLEANUP ---")

# Separate target early to safeguard it during feature processing
if TARGET_COLUMN not in df.columns:
    # Fallback to the last column if target name is mismatching
    TARGET_COLUMN = df.columns[-1]
    print(f"Warning: Defined target not found. Defaulting to last column: '{TARGET_COLUMN}'")

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# Step A: Drop completely unique identifier columns (e.g., ID, PassengerId, Serial Numbers)
id_cols = [col for col in X.columns if X[col].nunique() == len(X)]
if id_cols:
    X = X.drop(columns=id_cols)
    print(f"Dropped ID-like columns: {id_cols}")

# Step B: Handle Missing Values
# Impute numerical columns with the median
num_cols = X.select_dtypes(include=[np.number]).columns
if len(num_cols) > 0:
    num_imputer = SimpleImputer(strategy="median")
    X[num_cols] = num_imputer.fit_transform(X[num_cols])

# Impute categorical columns with the most frequent value (mode)
cat_cols = X.select_dtypes(include=["object", "category"]).columns
if len(cat_cols) > 0:
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
print("Successfully handled missing features using imputation.")

# Step C: Encode Categorical Features to Numbers
# Uses text-based encoding (One-Hot Encoding)
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# Encode Target variable if it contains string labels
if y.dtype == "object" or y.dtype == "category":
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    print("Encoded categorical target labels into numerical tokens.")

print(f"Cleaned Feature Shape: {X.shape[0]} rows, {X.shape[1]} features")

# Step D: Final Validation Check for Assignment Constraints
if X.shape[1] < 12:
    print(f"🛑 WARNING: Final feature count ({X.shape[1]}) is lower than the mandatory minimum requirement of 12! [Page 1]")
if X.shape[0] < 500:
    print(f"🛑 WARNING: Final instance count ({X.shape[0]}) is lower than the mandatory minimum requirement of 500! [Page 1]")

# ==========================================
# 4. SPLIT & EXPORT CLEAN TEST DATA
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save clean test data for Streamlit UI submission requirements [Page 2, 3]
test_data = pd.concat([X_test.reset_index(drop=True), pd.Series(y_test, name=TARGET_COLUMN)], axis=1)
test_data.to_csv("test_data.csv", index=False)
print("Successfully saved clean data matrix locally as 'test_data.csv' for the application framework.")

# ==========================================
# 5. FEATURE SCALING
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Persist scaler object for production inference mapping
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# ==========================================
# 6. MODEL TRAINING
# ==========================================
models = {
    "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision_Tree": DecisionTreeClassifier(random_state=42),
    "kNN": KNeighborsClassifier(),
    "Naive_Bayes": GaussianNB(),
    "Random_Forest": RandomForestClassifier(random_state=42)
}

print("\n--- MODEL TRAINING ---")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    with open(f"model/{name}.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"Successfully optimized and saved model architecture to 'model/{name}.pkl'")

print("\nPipeline executed cleanly! Run your streamlit dashboard via: streamlit run app.py")
