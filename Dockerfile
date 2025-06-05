# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set default environment variables
ENV MONGODB_DATABASE=GoEmotion \
    MONGODB_COLLECTION=vectorizedText \
    PORT=8080

# Install build dependencies and clean up in the same layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app/ app/

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose the port that FastAPI will run on
EXPOSE ${PORT}

# Start the FastAPI application with uvicorn
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}