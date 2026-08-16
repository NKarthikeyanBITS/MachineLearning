import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Define configuration constants
DATASET_PATH = "credit_card_fraud_2026.csv"  # Replace with your actual file path
TARGET_COLUMN = "is_fraud"              # Replace with your target name
TEST_SIZE_PERCENT = 0.20                    # 20% dedicated to testing split

def generate_clean_test_file(file_path, target_col):
    # 2. Load original raw dataset
    print("🔄 Loading dataset...")
    df = pd.read_csv(file_path)
    
    # 3. Handle target NaNs immediately to prevent Streamlit metric errors
    initial_count = len(df)
    df = df.dropna(subset=[target_col])
    dropped_count = initial_count - len(df)
    if dropped_count > 0:
        print(f"⚠️ Dropped {dropped_count} rows containing NaN values in the target column.")

    # 4. Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 5. Extract the evaluation validation split 
    # stratify=y preserves categorical target proportions across splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE_PERCENT, 
        random_state=42,
        stratify=y if y.dtype == 'object' or len(y.unique()) < 10 else None
    )
    
    # 6. Reconstruct the precise test dataset layout required by your UI
    # Explicitly resetting both indices prevents alignment shifts / NaN injection
    test_data = pd.concat([
        X_test.reset_index(drop=True), 
        pd.Series(y_test, name=target_col).reset_index(drop=True)
    ], axis=1)
    
    # 7. Save output matrix locally
    output_filename = "test_data1.csv"
    test_data.to_csv(output_filename, index=False)
    print(f"✅ Successfully saved clean data matrix locally as '{output_filename}' ({len(test_data)} rows).")

# Execute generation
if __name__ == "__main__":
    generate_clean_test_file(DATASET_PATH, TARGET_COLUMN)
