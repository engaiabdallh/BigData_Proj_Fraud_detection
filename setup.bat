@echo off
echo ============================================
echo    Fraud Detection System - Setup
echo ============================================
echo.

REM Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed!
    echo Please install Python 3.9-3.11 from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo [OK] Python found
python --version
echo.

REM Check Java installation
java -version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Java not found in PATH
    echo Please install Java 8 or 11 from https://adoptium.net/
    echo Or set JAVA_HOME environment variable
) else (
    echo [OK] Java found
    java -version 2>&1 | findstr /i "version"
)
echo.

REM Create virtual environment
echo [INFO] Creating virtual environment...
if exist .venv (
    echo [INFO] Virtual environment already exists
    choice /C YN /M "Do you want to recreate it? "
    if errorlevel 2 goto skip_venv
    rmdir /s /q .venv
)

python -m venv .venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment created

:skip_venv

REM Activate and install dependencies
echo.
echo [INFO] Installing dependencies...
call .venv\Scripts\activate.bat

REM Try pip first, fallback to uv
pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [WARNING] pip install failed, trying with uv...
    pip install uv
    uv pip install -r requirements.txt
)

echo.
echo ============================================
echo    Setup Complete!
echo ============================================
echo.
echo To run the app:
echo   Double-click run.bat
echo   OR run: streamlit run app.py
echo.
pause