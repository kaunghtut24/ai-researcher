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
    xvfb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements.txt and install dependencies first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Make startup script executable
RUN chmod +x start.sh

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port 10000 for Render
EXPOSE 10000

# Health check for deployment monitoring
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:10000/?health=check || exit 1

# Command optimized for Render deployment
CMD ["./start.sh"]

