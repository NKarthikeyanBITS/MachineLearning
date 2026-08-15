import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# 1. Load your chosen dataset (Replace 'your_dataset.csv' with your actual file)
# Ensure your dataset has at least 12 features and 500 instances
df = pd.read_csv("winequality-white.csv")

# Separate features and target (assuming target is the last column)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 2. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save the test data to a CSV file for the Streamlit app upload requirement
test_data = pd.concat([X_test, y_test], axis=1)
test_data.to_csv("test_data.csv", index=False)

# 3. Scale features (Highly recommended for Logistic Regression, kNN, etc.)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Save the scaler to apply the exact same transformation in Streamlit
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# 4. Initialize the 5 required classification models
models = {
    "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision_Tree": DecisionTreeClassifier(random_state=42),
    "kNN": KNeighborsClassifier(),
    "Naive_Bayes": GaussianNB(),
    "Random_Forest": RandomForestClassifier(random_state=42)
}

# 5. Train and save each model
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    with open(f"model/{name}.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"Successfully trained and saved {name}.pkl")

print("Training complete. 'test_data.csv' and model files are ready.")
