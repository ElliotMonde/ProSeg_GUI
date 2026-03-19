FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV strictly required by ultralytics
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code and model weights (.pt files)
COPY . .

EXPOSE 8000

# Run FastAPI backend with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
