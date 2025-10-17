FROM python:3.10-slim

# Environment Setup 
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app

# Working Directory 
WORKDIR /app

# Install System Dependencies 
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        apt-transport-https \
        apt-utils \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        curl \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*


# Copy and Install Python Dependencies 
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

#Copy Source Code 
COPY . /app

#Create Persistent Directories 
RUN mkdir -p /app/chroma_db /app/data

# Expose FastAPI Port 
EXPOSE 8000

# Default Environment Variables 
ENV CHROMA_DIR=/app/chroma_db

# Start Command
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

