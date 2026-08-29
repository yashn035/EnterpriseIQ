# Use official Python 3.10 slim image
FROM python:3.10-slim

# Prevent Python from writing .pyc files & enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Make start.sh executable
RUN chmod +x /app/start.sh

# Expose default port
EXPOSE 8000 10000

# Default command to launch Streamlit application using $PORT
CMD ["/app/start.sh"]
