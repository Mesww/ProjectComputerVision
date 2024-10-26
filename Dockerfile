# Start with your base image (e.g., Python 3.10)
FROM python:3.10-slim

# Install required libraries for OpenCV, threading, and zbar
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libzbar0 \
    && rm -rf /var/lib/apt/lists/*

# Copy your application code
WORKDIR /app
COPY . /app

# Install Python dependencies including Gunicorn
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Flask application port
EXPOSE 5000

# Run the Flask application with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--worker-class", "gevent", "--workers", "1"]
