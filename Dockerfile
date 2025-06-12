FROM python:3.9-slim
# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100000 -r requirements.txt

# Copy application files
COPY app1.py .
COPY leafNetV3_model.tflite .
COPY converted_model.tflite .
COPY mobilevit_model17.pt .

# Set environment variable for Cloud Run port
ENV PORT=8080
EXPOSE 8080

# Run the Flask app using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app1:app"]
# CMD ["python", "app1.py"]
