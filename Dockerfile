# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy application files
COPY . /app

# Install system-level dependencies for cv2 (if needed)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask runs on
EXPOSE 5000

# Define the entry point
CMD ["gunicorn", "-w", "4", "--threads", "2", "-b", "0.0.0.0:5000", "model:app"]
