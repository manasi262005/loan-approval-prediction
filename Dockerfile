FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run train.py to generate models if not available
RUN python train.py

EXPOSE 8000

HEALTHCHECK CMD curl --fail http://localhost:8000/ || exit 1

# Start FastAPI and mount static directory
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
