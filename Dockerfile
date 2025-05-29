# Use a Python 3.10 base image
FROM python:3.11

# Set working dir
WORKDIR /app

# Copy only what’s needed
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app1.py .
COPY leafNetV3_model.tflite .
COPY converted_model.tflite .

# Expose the port Flask will run on
ENV PORT 8080
EXPOSE 8080

# Run your Flask app
CMD ["python", "app1.py"]
