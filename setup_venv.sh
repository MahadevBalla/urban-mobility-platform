#!/bin/bash

# Setup script for virtual environment

echo "======================================"
echo "Urban Transit Tool - Environment Setup"
echo "======================================"

# Create virtual environment
echo ""
echo "[1/3] Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo ""
echo "[2/3] Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "[3/3] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "======================================"
echo "✅ Setup complete!"
echo "======================================"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the test:"
echo "  python test_zone_generation.py"
echo ""
echo "To deactivate when done:"
echo "  deactivate"
echo ""
