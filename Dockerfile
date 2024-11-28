
FROM python:3.10-slim


WORKDIR /app


COPY . .


RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    apt-get clean


RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 5000


CMD ["python", "model.py"]
