# Use a minimal and compatible Python base image
FROM python:3.10-slim

# Set environment variables for best practice
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies with increased timeout and clean cache
RUN pip install --no-cache-dir --default-timeout=100000 -r requirements.txt

# Copy application and model files
COPY app1.py .
COPY leafNetV3_model.tflite .
COPY converted_model.tflite .
COPY mobilevit_model17.pt .

# (Recommended) Use a non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Expose the port Flask will run on
EXPOSE 8080

# Run your Flask app
CMD ["python", "app1.py"]
