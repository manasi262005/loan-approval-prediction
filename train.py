import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
import joblib
import json
import os

def load_data():
    if os.path.exists("loan_data.csv"):
        return pd.read_csv("loan_data.csv")
    remote_url = "https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv"
    df = pd.read_csv(remote_url)
    df.to_csv("loan_data.csv", index=False)
    return df

def preprocess(df):
    if 'Loan_ID' in df.columns:
        df = df.drop('Loan_ID', axis=1)
    
    # In case there's an index column
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

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

def train_models(df_encoded):
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

    results = {}
    trained_models = {}

    print("=" * 50)
    print("       AutoML Model Comparison Results")
    print("=" * 50)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results[name] = {'accuracy': acc, 'f1_score': f1}
        trained_models[name] = model
        print(f"{name:25s} -> Accuracy: {acc:.4f} | F1: {f1:.4f}")

    best_name = max(results, key=lambda k: results[k]['accuracy'])
    best_model = trained_models[best_name]
    best_acc = results[best_name]['accuracy']

    print("=" * 50)
    print(f"Best Model: {best_name} (Accuracy: {best_acc*100:.2f}%)")
    print("=" * 50)

    # Save model and metadata
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(list(X.columns), 'model_columns.pkl')

    metrics = {
        'best_model': best_name,
        'accuracy': best_acc,
        'f1_score': results[best_name]['f1_score'],
        'all_results': {k: v['accuracy'] for k, v in results.items()}
    }
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved: best_model.pkl")
    print(f"Columns saved: model_columns.pkl")
    print(f"Metrics saved: metrics.json")

    return best_model, best_name, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = load_data()
    df_encoded, categorical_cols = preprocess(df)
    joblib.dump(categorical_cols, 'categorical_cols.pkl')
    train_models(df_encoded)
