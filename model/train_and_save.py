import os
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Ensure output directory exists
os.makedirs("model", exist_ok=True)

# 1. Load and process data
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save test data to project root for user testing
test_df = pd.DataFrame(X_test, columns=data.feature_names)
test_df['target'] = y_test
test_df.to_csv("test_data.csv", index=False)

# 2. Define models
models = {
    "Logistic_Regression": LogisticRegression(random_state=42),
    "Decision_Tree": DecisionTreeClassifier(random_state=42),
    "K-Nearest_Neighbor": KNeighborsClassifier(n_neighbors=5),
    "Gaussian_Naive_Bayes": GaussianNB(),
    "Random_Forest": RandomForestClassifier(random_state=42)
}

# 3. Train and save models
joblib.dump(scaler, "model/scaler.joblib")

for name, model in models.items():
    if name in ["Logistic_Regression", "K-Nearest_Neighbor"]:
        model.fit(X_train_scaled, y_train)
    else:
        model.fit(X_train, y_train)
    
    # Save serialized model file
    joblib.dump(model, f"model/{name}.joblib")

print("All models, scaler, and test_data.csv successfully saved!")
