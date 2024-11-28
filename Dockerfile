# Use the official Python image as the base
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy the contents of your project to the container
COPY . .

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    apt-get clean

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Flask app port
EXPOSE 5000

# Run the Flask app
CMD ["python", "model.py"]
