#!/bin/bash

echo "============================================"
echo "Urban Transit Zone Generator Dashboard"
echo "============================================"
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Please run setup_venv.sh first."
    exit 1
fi

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

# Install streamlit-folium if needed
echo "Checking dependencies..."
pip install -q streamlit-folium 2>/dev/null

# Launch dashboard
echo ""
echo "============================================"
echo "Starting dashboard..."
echo "Dashboard will open in your browser"
echo "Press Ctrl+C to stop the server"
echo "============================================"
echo ""

streamlit run app.py
