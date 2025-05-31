# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set default environment variables
ENV MONGODB_DATABASE=GoEmotion \
    MONGODB_COLLECTION=vectorizedText

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt .

# Install dependencies with pip cache
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app/ app/

# Expose the port that FastAPI will run on
EXPOSE 8000

# Start the FastAPI application with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 