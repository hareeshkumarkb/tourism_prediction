# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/hareeshkumarkb/tourism_prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop unique identifier and 'Unnamed: 0' if it exists
df.drop(columns=['CustomerID', 'Unnamed: 0'], errors='ignore', inplace=True)

# Define the target variable
target_col = 'ProdTaken'

# Identify categorical columns for encoding (excluding the target if it were categorical)
categorical_cols = df.select_dtypes(include='object').columns.tolist()

# Encoding categorical columns
label_encoder = LabelEncoder()
for col in categorical_cols:
    # Handle potential NaNs before encoding by converting to string
    df[col] = df[col].astype(str)
    df[col] = label_encoder.fit_transform(df[col])

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save prepared data to CSV files
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)


files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

# Upload files to Hugging Face Hub
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="hareeshkumarkb/tourism_prediction",
        repo_type="dataset",
    )
print("Data prepared, split, and uploaded to Hugging Face Hub.")
