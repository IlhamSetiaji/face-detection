@echo off
echo Setting up Face Detection Application...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.13.6 and try again
    pause
    exit /b 1
)

echo Python found. Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Setup complete! 
echo.
echo To run the application:
echo 1. Activate virtual environment: venv\Scripts\activate.bat
echo 2. Run tests: python test_system.py
echo 3. Start Flask app: python src\presentation\flask_app\app.py
echo.
pause
