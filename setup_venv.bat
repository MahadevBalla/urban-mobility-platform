@echo off
REM Setup script for virtual environment (Windows)

echo ======================================
echo Urban Transit Tool - Environment Setup
echo ======================================

REM Create virtual environment
echo.
echo [1/3] Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo.
echo [2/3] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo.
echo [3/3] Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ======================================
echo Setup complete!
echo ======================================
echo.
echo To activate the environment:
echo   venv\Scripts\activate.bat
echo.
echo To run the test:
echo   python test_zone_generation.py
echo.
echo To deactivate when done:
echo   deactivate
echo.

pause
