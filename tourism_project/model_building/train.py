# for data manipulation
import pandas as pd
import numpy as np # Import numpy
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline # Import Pipeline for explicit naming
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score # Import all necessary metrics
# for model serialization
import joblib
# for hugging face space authentication to upload files
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism package")

api = HfApi(token=os.getenv("HF_TOKEN")) # Initialize API with token

Xtrain_path = "hf://datasets/hareeshkumarkb/tourism_prediction/Xtrain.csv"
Xtest_path = "hf://datasets/hareeshkumarkb/tourism_prediction/Xtest.csv"
ytrain_path = "hf://datasets/hareeshkumarkb/tourism_prediction/ytrain.csv"
ytest_path = "hf://datasets/hareeshkumarkb/tourism_prediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze() # .squeeze() to convert DataFrame to Series
ytest = pd.read_csv(ytest_path).squeeze() # .squeeze() to convert DataFrame to Series


# Define numeric features (all columns are now numerical after prep.py)
numeric_features = Xtrain.columns.tolist()
categorical_features = [] # All categorical features were label encoded in prep.py, so they are now numeric

# Preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    remainder='passthrough' # Keep other columns if any, or 'drop' them
)

# Define base XGBoost model (classification objective based on ProdTaken target)
xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')

# Calculate class weight for imbalanced data (for classification)
neg_count = ytrain.value_counts().get(0, 0)
pos_count = ytrain.value_counts().get(1, 0)
scale_pos_weight = neg_count / pos_count if pos_count != 0 else 1.0
xgb_model.set_params(scale_pos_weight=scale_pos_weight)

# Model pipeline with explicit step names
model_pipeline = Pipeline(steps=[
    ('preprocessor_step', preprocessor),
    ('classifier_step', xgb_model)
])

# Hyperparameter grid using explicit step names
param_grid = {
    'classifier_step__n_estimators': [50, 100, 150],
    'classifier_step__max_depth': [3, 5, 7],
    'classifier_step__learning_rate': [0.01, 0.05, 0.1],
    'classifier_step__subsample': [0.7, 0.8, 1.0],
    'classifier_step__colsample_bytree': [0.7, 0.8, 1.0],
    'classifier_step__reg_lambda': [0.1, 1, 10]
}

with mlflow.start_run():
    # Grid Search
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1, scoring='roc_auc') # Use roc_auc for classification
    grid_search.fit(Xtrain, ytrain)

    # Log parameter sets
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_roc_auc", mean_score)

    # Best model
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Predictions
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)
    y_proba_test = best_model.predict_proba(Xtest)[:, 1]

    # Metrics for classification
    train_accuracy = accuracy_score(ytrain, y_pred_train)
    test_accuracy = accuracy_score(ytest, y_pred_test)
    train_precision = precision_score(ytrain, y_pred_train)
    test_precision = precision_score(ytest, y_pred_test)
    train_recall = recall_score(ytrain, y_pred_train)
    test_recall = recall_score(ytest, y_pred_test)
    train_f1 = f1_score(ytrain, y_pred_train)
    test_f1 = f1_score(ytest, y_pred_test)
    test_roc_auc = roc_auc_score(ytest, y_proba_test)

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_precision": train_precision,
        "test_precision": test_precision,
        "train_recall": train_recall,
        "test_recall": test_recall,
        "train_f1": train_f1,
        "test_f1": test_f1,
        "test_roc_auc": test_roc_auc
    })

    # Save the model locally
    model_path = "best_tourism_prediction_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id_hf = "hareeshkumarkb/tourism_prediction_model" # Corrected repo_id
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id_hf, repo_type=repo_type)
        print(f"Space '{repo_id_hf}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id_hf}' not found. Creating new space...")
        create_repo(repo_id=repo_id_hf, repo_type=repo_type, private=False)
        print(f"Space '{repo_id_hf}' created.")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="best_tourism_prediction_model_v1.joblib",
        repo_id=repo_id_hf,
        repo_type=repo_type,
    )
print("Model uploaded to Hugging Face Hub.")
