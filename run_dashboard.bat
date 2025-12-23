@echo off
echo ============================================
echo Urban Transit Zone Generator Dashboard
echo ============================================
echo.

REM Check if venv exists
if not exist "venv\" (
    echo ERROR: Virtual environment not found!
    echo Please run setup_venv.bat first.
    pause
    exit /b 1
)

REM Activate venv
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install streamlit-folium if needed
echo Checking dependencies...
pip install -q streamlit-folium 2>nul

REM Launch dashboard
echo.
echo ============================================
echo Starting dashboard...
echo Dashboard will open in your browser
echo Press Ctrl+C to stop the server
echo ============================================
echo.

streamlit run app.py

pause
