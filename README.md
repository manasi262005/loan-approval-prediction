# Loan Approval Prediction System - MLOps Edition 🚀

A comprehensive, production-ready machine learning project that predicts whether a loan application will be **approved or rejected** based on applicant details. This project has evolved from a simple Jupyter Notebook into a full-scale **MLOps Deployment** architecture.

---

## 🏗️ System Architecture

This project is fully containerized and features several interconnected microservices:

1. **FastAPI Backend (`api.py`)** - Very fast REST API that loads the machine learning models and serves predictions in real-time.
2. **Interactive Front-End (`static/`)** - A custom, beautiful web dashboard built with HTML/CSS/JS with interactive Plotly charts.
3. **Model Explainability (SHAP)** - Built directly into the backend to provide mathematical explanations for *why* a loan was approved or rejected.
4. **Monitoring Stack (Prometheus + Grafana)** - Real-time metrics collection tracking API latency, prediction distributions, and active model health.
5. **Dockerization** - The entire stack runs seamlessly across any OS via `docker-compose`.

---

## ⚡ How It Works

### The User Flow
1. Users visit the Web Dashboard on Port `8000`.
2. They input their standard financial details (Income, Loan Amount, Term length).
3. The front-end automatically calculates required variables and shoots a request to the FastAPI application.
4. The system validates the numbers, translates them into the scale the AI desires, and runs a prediction through the `Random Forest / Gradient Boosting` pipeline.
5. In split seconds, the user gets a bold Approved/Rejected screen complete with financial breakdown math to explain affordability.

### The Developer Flow
* **Data Processing:** `train.py` pulls the raw CSV data, imputes missing records with medians/modes, auto-creates engineered features (`TotalIncome`, `EMI`, `BalanceIncome`), formats categorical strings via One-Hot Encoding, and saves everything to `.pkl` artifacts.
* **AutoML Selection:** The training script automatically trains and competes four different classifiers (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting) and silently promotes the most accurate model to Production.
* **Model Degradation:** The deployment runs a `degradation_check.py` routine to detect data drifting or sudden crashes in approval odds.
* **CI/CD:** Through `.github/workflows/train.yml`, whenever data changes are pushed to GitHub, the server automatically executes `train.py` to continuously self-heal and improve the model.

---

## 💻 Running the Application

Because the entire application is containerized, starting it is incredibly easy.

**Step 1:** Force Docker to rebuild the image and bypass cache:
```bash
docker compose build --no-cache
```

**Step 2:** Start up the server and the monitoring stack:
```bash
docker compose up
```

**Step 3:** Open your browser and navigate to:
* **The Application:** [http://localhost:8000](http://localhost:8000)
* **API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)

*(Note: Ensure you hard-refresh your browser cache `Ctrl+Shift+R` to see the latest static UI updates).*

---

## 🛠️ Features Engineered

| Feature | Why It Matters |
|---------|----------------|
| **TotalIncome** | Combined household income gives a better picture of repayment ability |
| **EMI** | Monthly installment amount shows the actual burden on the applicant |
| **BalanceIncome** | Income left after paying EMI indicates financial comfort |

---

## 📊 Analytics and Monitoring

1. **Dashboard Data Explorer:** The web app features a page devoted to parsing and viewing the static summary dataset directly.
2. **Dashboard Model Performance:** Visualizes the backend training competition so users know which AI Model actually won and is currently serving their predictions.
3. **Prometheus Metrics:** Hooked to `port 8001`, silently scraping latency tags and prediction odds in the background.

---

## Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Python** | Core logic and training |
| **FastAPI** | High-performance Web server |
| **Scikit-learn** | Machine learning framework |
| **SHAP** | Explainable AI prediction mathematics |
| **Docker** | Cross-platform containerization |
| **Prometheus / Grafana** | MLOps operational monitoring |
| **GitHub Actions** | Automated CI/CD execution pipeline |
