import pandas as pd
import numpy as np
import joblib
import json
import os
import subprocess

def check_degradation():
    if not os.path.exists('best_model.pkl'):
        print("No model found. Exiting.")
        return False

    if not os.path.exists('metrics.json'):
        print("No baseline metrics found. Run train.py first.")
        return False

    with open('metrics.json', 'r') as f:
        baseline = json.load(f)

    baseline_accuracy = baseline['accuracy']

    log_file = 'logs/predictions.csv'
    if not os.path.exists(log_file):
        print("No prediction logs found. Not enough data to check degradation.")
        return False

    logs = pd.read_csv(log_file)
    total_predictions = len(logs)

    print("=" * 50)
    print("    Model Degradation Check")
    print("=" * 50)
    print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
    print(f"Total Predictions Logged: {total_predictions}")

    degraded = False
    reasons = []

    if total_predictions > 10:
        approval_rate = (logs['prediction'] == 'Approved').mean()
        if approval_rate > 0.95:
            degraded = True
            reasons.append(f"Approval rate too high: {approval_rate:.2f} (>0.95)")
        elif approval_rate < 0.30:
            degraded = True
            reasons.append(f"Approval rate too low: {approval_rate:.2f} (<0.30)")

    if total_predictions > 10:
        recent_confidence = logs.tail(10)['confidence'].mean()
        overall_confidence = logs['confidence'].mean()
        confidence_drop = overall_confidence - recent_confidence
        if confidence_drop > 0.05:
            degraded = True
            reasons.append(f"Confidence dropping: {confidence_drop:.4f} drop (>0.05)")

    if total_predictions > 20:
        recent_income = logs.tail(20)['applicant_income'].mean()
        overall_income = logs['applicant_income'].mean()
        income_drift = abs(recent_income - overall_income) / overall_income
        if income_drift > 0.15:
            degraded = True
            reasons.append(f"Data drift detected: income shifted by {income_drift:.2f} (>0.15)")

    if degraded:
        print("\nDEGRADATION DETECTED!")
        for r in reasons:
            print(f"  - {r}")
        print("\nTriggering retraining...")
        return True
    else:
        print("\nModel is healthy. No degradation detected.")
        return False

def trigger_retrain():
    print("\nRetraining model...")
    subprocess.run(["python", "train.py"], check=True)
    print("Retraining complete. New model saved.")

if __name__ == "__main__":
    if check_degradation():
        trigger_retrain()
