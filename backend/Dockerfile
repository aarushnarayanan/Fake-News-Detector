FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend ./backend
COPY models ./models
# Copy other necessary files (like clean_df.py if imported, but likely only models/predict.py logic uses transformers)
# models/predict.py uses tokenization, which relies on transformers.
# We ensure the models folder contains the model weights. The user has them locally.
# If they are large, Docker build context might be slow. 
# For now, we assume they are copied.

ENV PYTHONPATH=/app

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
