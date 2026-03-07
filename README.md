# Loan Approval Prediction System

A machine learning project that predicts whether a loan application will be **approved or rejected** based on applicant details like income, credit history, and loan amount. Built as a university FinTech project with a focus on transparency, automation, and production readiness.

---

## What This Project Does

When someone applies for a loan, banks need to decide whether to approve or reject it. This system automates that decision using machine learning. It takes in applicant data, processes it, compares multiple models, picks the best one, and gives a clear approve/reject decision with reasoning.

---

## How It Works

The project follows a step-by-step pipeline:

```
Raw CSV Data
     |
     v
Clean & Fill Missing Values (Imputation)
     |
     v
Create New Useful Features (Feature Engineering)
     |
     v
Convert Text to Numbers (One-Hot Encoding)
     |
     v
Train & Compare 4 Models (AutoML)
     |
     v
Pick the Best Model Automatically
     |
     v
Explain Predictions Using SHAP (Explainability)
     |
     v
Visualize Data in 3D (Plotly)
     |
     v
Save Model for Deployment (MLOps)
     |
     v
Run a Live Prediction Demo
```

---

## Dataset

We use the [Loan Prediction Dataset](https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv) which contains **491 records** of loan applicants.

**Columns in the dataset:**

| Column | Description |
|--------|-------------|
| Gender | Male / Female |
| Married | Yes / No |
| Dependents | Number of dependents (0, 1, 2, 3+) |
| Education | Graduate / Not Graduate |
| Self_Employed | Yes / No |
| ApplicantIncome | Applicant's monthly income |
| CoapplicantIncome | Co-applicant's monthly income |
| LoanAmount | Loan amount requested (in thousands) |
| Loan_Amount_Term | Loan repayment term (in months) |
| Credit_History | 1 = has credit history, 0 = no credit history |
| Property_Area | Urban / Semiurban / Rural |
| Loan_Status | 1 = Approved, 0 = Rejected (Target) |

---

## Features We Engineered

From the raw data, we created three new features that help the model make better predictions:

| Feature | Formula | Why It Matters |
|---------|---------|----------------|
| **TotalIncome** | ApplicantIncome + CoapplicantIncome | Combined household income gives a better picture of repayment ability |
| **EMI** | LoanAmount / Loan_Amount_Term | Monthly installment amount shows the actual burden on the applicant |
| **BalanceIncome** | TotalIncome - EMI | Income left after paying EMI indicates financial comfort |

---

## Data Preprocessing

Real-world data is messy. This dataset has missing values in multiple columns. Here is how we handled them:

- **Numerical columns** (like LoanAmount, Credit_History) -- filled missing values with the **median** (middle value), which is more robust to outliers than the mean
- **Categorical columns** (like Gender, Married) -- filled missing values with the **mode** (most frequent value)
- **One-Hot Encoding** -- converted all text categories into binary 0/1 columns so the model can process them

---

## Models Compared (AutoML)

We trained and compared four different machine learning models on the same data:

| Model | What It Does |
|-------|-------------|
| **Logistic Regression** | A simple linear model that estimates the probability of approval |
| **Decision Tree** | Makes decisions by splitting data based on feature thresholds |
| **Random Forest** | Combines 100 decision trees and takes a majority vote for better accuracy |
| **Gradient Boosting** | Builds trees one after another, each correcting the mistakes of the previous one |

The model with the **highest accuracy** on the test set is automatically selected as the best model.

---

## Explainability (SHAP)

In finance, it is not enough to just predict -- we need to explain **why** a loan was approved or rejected. We use **SHAP (SHapley Additive exPlanations)** to do this.

SHAP assigns an importance value to each feature for every prediction:

- **Credit_History** is usually the most important factor. Having a credit history strongly pushes the model toward approval.
- **TotalIncome** and **BalanceIncome** matter a lot -- higher income means higher approval chances.
- **LoanAmount** -- requesting very large loans can push toward rejection.
- **Property_Area** and **Married** status also contribute to the decision.

The notebook generates a **SHAP Summary Plot** that visually shows all of this.

---

## 3D Visualization

An interactive 3D scatter plot built with **Plotly** lets you explore the data visually:

- **X-axis:** Total Income
- **Y-axis:** EMI
- **Z-axis:** Loan Amount
- **Green dots:** Approved loans
- **Red dots:** Rejected loans

You can rotate, zoom, and hover over points to see individual applicant details.

---

## MLOps and CI/CD

### Model Export
The best model is saved as a `.pkl` file using `joblib`. This means it can be loaded and used for predictions without retraining.

### GitHub Actions Pipeline
A CI/CD pipeline is included at `.github/workflows/train.yml`. It automatically retrains the model whenever a new `data.csv` is pushed to the repository.

```yaml
Trigger: Push a new data.csv
    -> Installs Python and dependencies
    -> Runs the training script
    -> Uploads the trained model as an artifact
```

---

## Live Prediction Demo

The notebook includes a demo where a sample applicant's data is fed into the model:

```
Applicant Income : Rs 5,000
Co-applicant Inc.: Rs 2,000
Total Income     : Rs 7,000
Loan Amount      : Rs 150
EMI              : Rs 0.42
Credit History   : Yes

DECISION: *** APPROVED ***
REASON: The applicant has a positive credit history,
sufficient total income, and a manageable EMI-to-income
ratio, making them a low-risk borrower.
```

---

## Project Structure

```
loan-approval-prediction/
|-- loan_approval.ipynb              # Main notebook with all code
|-- README.md                        # Project documentation
|-- .github/
|   |-- workflows/
|       |-- train.yml                # CI/CD pipeline configuration
```

---

## How to Run

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `loan_approval.ipynb`
3. Click **Runtime -> Run all**
4. All outputs, plots, and predictions will appear in order

Alternatively, run locally with Jupyter:

```bash
pip install pandas numpy scikit-learn shap plotly joblib matplotlib
jupyter notebook loan_approval.ipynb
```

---

## Technologies Used

| Technology | Purpose |
|-----------|---------|
| Python | Programming language |
| Pandas & NumPy | Data manipulation and processing |
| Scikit-learn | Machine learning models and evaluation |
| SHAP | Model explainability |
| Plotly | Interactive 3D visualization |
| joblib | Model serialization and export |
| GitHub Actions | CI/CD automation |

---

## Key Concepts Covered

- **Data Imputation** -- handling missing values with median and mode
- **Feature Engineering** -- creating TotalIncome, EMI, and BalanceIncome
- **One-Hot Encoding** -- converting categorical data to numerical format
- **Ensemble Learning** -- Random Forest and Gradient Boosting techniques
- **AutoML** -- automatic model comparison and selection
- **SHAP Explainability** -- interpreting model decisions transparently
- **Model Persistence** -- saving and loading models with joblib
- **CI/CD Pipelines** -- automating retraining with GitHub Actions
