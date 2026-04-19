import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os

def load_data():
    local_path = "loan_data.csv"
    remote_url = "https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv"
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
    else:
        df = pd.read_csv(remote_url)
    return df

def preprocess(df):
    if 'Loan_ID' in df.columns:
        df = df.drop('Loan_ID', axis=1)

    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    if 'Loan_Status' in numerical_cols:
        numerical_cols.remove('Loan_Status')
    if 'Loan_Status' in categorical_cols:
        categorical_cols.remove('Loan_Status')

    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']
    df['BalanceIncome'] = df['TotalIncome'] - df['EMI']

    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df_encoded, categorical_cols

def register_model(run_id, model_name="LoanApprovalModel"):
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, model_name)
    print(f"Model registered: {model_name} version {result.version}")
    
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Staging"
    )
    print(f"Model version {result.version} moved to Staging")
    
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Production"
    )
    print(f"Model version {result.version} promoted to Production")
    
    for mv in client.search_model_versions(f"name='{model_name}'"):
        if mv.version != result.version and mv.current_stage == "Production":
            client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage="Archived"
            )
            print(f"Model version {mv.version} archived")

    return result.version

def train_with_mlflow():
    mlflow.set_experiment("Loan-Approval-AutoML")

    df = load_data()
    df_encoded, categorical_cols = preprocess(df)

    X = df_encoded.drop('Loan_Status', axis=1)
    y = df_encoded['Loan_Status']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    best_accuracy = 0
    best_run_id = None
    best_model_name = None

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            mlflow.log_param("model_type", name)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)

            mlflow.sklearn.log_model(model, "model")

            print(f"{name:25s} -> Accuracy: {acc:.4f} | F1: {f1:.4f}")

            if acc > best_accuracy:
                best_accuracy = acc
                best_run_id = mlflow.active_run().info.run_id
                best_model_name = name

    print(f"\nBest Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
    print(f"Best Run ID: {best_run_id}")
    
    register_model(best_run_id)

    return best_run_id, best_model_name

if __name__ == "__main__":
    train_with_mlflow()
