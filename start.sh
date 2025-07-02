#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server in the background
python server.py &

# Start the Streamlit app
streamlit run app.py
