@echo off
REM Run the Streamlit application on Windows

REM Activate virtual environment
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Run install.bat first.
    exit /b 1
)

REM Check if .env exists
if not exist ".env" (
    echo ⚠ .env file not found!
    exit /b 1
)

REM Check if API keys are configured
findstr /M "your_api_key_here" .env >nul 2>&1
if %errorlevel% equ 0 (
    echo ⚠ Warning: API keys not configured in .env
    echo The application will work but won't fetch real data from API
    echo.
)

REM Run the app
echo 🚀 Starting Football Predictions System...
echo Opening http://localhost:8501
echo.
echo Press Ctrl+C to stop
echo.

streamlit run main.py
