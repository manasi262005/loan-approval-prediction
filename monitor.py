from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

PREDICTION_COUNT = Counter(
    'loan_predictions_total',
    'Total number of loan predictions',
    ['result']
)

PREDICTION_LATENCY = Histogram(
    'loan_prediction_latency_seconds',
    'Time taken for each prediction',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
)

MODEL_ACCURACY = Gauge(
    'loan_model_accuracy',
    'Current model accuracy'
)

APPROVAL_RATE = Gauge(
    'loan_approval_rate',
    'Current approval rate'
)

PREDICTION_CONFIDENCE = Histogram(
    'loan_prediction_confidence',
    'Prediction confidence scores',
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
)

def record_prediction(result, confidence, latency):
    PREDICTION_COUNT.labels(result=result).inc()
    PREDICTION_LATENCY.observe(latency)
    PREDICTION_CONFIDENCE.observe(confidence)

def update_model_metrics(accuracy, approval_rate):
    MODEL_ACCURACY.set(accuracy)
    APPROVAL_RATE.set(approval_rate)

def start_metrics_server(port=8000):
    start_http_server(port)
    print(f"Prometheus metrics server running on http://localhost:{port}/metrics")
