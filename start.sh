#!/bin/bash

# Startup script for Render deployment
set -e

echo "Starting Agentic Deep Researcher..."

# Set default environment variables if not provided
export STREAMLIT_SERVER_PORT=${STREAMLIT_SERVER_PORT:-10000}
export STREAMLIT_SERVER_ADDRESS=${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}
export STREAMLIT_SERVER_HEADLESS=${STREAMLIT_SERVER_HEADLESS:-true}
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=${STREAMLIT_BROWSER_GATHER_USAGE_STATS:-false}

# Create necessary directories
mkdir -p /app/logs

# Start the application
exec streamlit run app.py \
    --server.port=$STREAMLIT_SERVER_PORT \
    --server.address=$STREAMLIT_SERVER_ADDRESS \
    --server.headless=$STREAMLIT_SERVER_HEADLESS \
    --server.enableCORS=false \
    --browser.gatherUsageStats=$STREAMLIT_BROWSER_GATHER_USAGE_STATS
