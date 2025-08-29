FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --timeout=1000 --retries=5 -r requirements.txt

# Copy rest of project
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit when container starts
CMD ["streamlit", "run", "/app/frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
