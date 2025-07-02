# Use Python 3.11 slim base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies required for WeasyPrint and general builds
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libffi-dev \
    libssl-dev \
    libxml2 \
    libxslt1-dev \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libcairo2 \
    libgdk-pixbuf2.0-0 \
    fonts-liberation \
    fonts-freefont-ttf \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create an empty requirements.txt to satisfy Render's build process
RUN touch requirements.txt

# Copy only files needed for installing dependencies first (to cache better)
COPY pyproject.toml .
RUN pip install --upgrade pip setuptools wheel
RUN pip install .

# Copy the rest of the app
COPY . .

# Expose Streamlit default port (change if needed)
EXPOSE 8501

# Default command (can be changed in render.yaml)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]

