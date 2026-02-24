# Use Python 3.10 (TensorFlow compatible)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (important for opencv & tensorflow)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Run the application with Gunicorn
CMD gunicorn --bind 0.0.0.0:$PORT --timeout 120 app:app
