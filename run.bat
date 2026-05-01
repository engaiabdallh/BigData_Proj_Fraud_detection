@echo off
echo ============================================
echo    Fraud Detection System
echo ============================================
echo.

REM Check if virtual environment exists
if not exist .venv (
    echo [ERROR] Virtual environment not found!
    echo Please run setup.bat first
    pause
    exit /b 1
)

REM Set environment variables for Spark
set PYTHONUNBUFFERED=1
set PYSPARK_PYTHON=.venv\Scripts\python.exe
set PYSPARK_DRIVER_PYTHON=.venv\Scripts\python.exe

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Check if models exist
if not exist models (
    echo [WARNING] Models folder not found!
    echo The app will run but may not have trained models
    echo.
)

echo [INFO] Starting Fraud Detection System...
echo.
echo The app will open in your default browser
echo Press Ctrl+C in this window to stop the app
echo ============================================
echo.

REM Run the app
streamlit run app.py --server.maxUploadSize 200

pause