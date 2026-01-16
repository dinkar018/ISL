FROM python:3.10-slim

# Prevent Python buffering
ENV PYTHONUNBUFFERED=1

# System dependencies for MediaPipe + OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the project
COPY . .

# Expose Render port
EXPOSE 10000

# Start app
CMD ["bash", "start.sh"]
