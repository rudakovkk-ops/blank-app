@echo off
REM Installation script for Windows
REM Football Prediction System

echo 🚀 Football Predictions System Installation
echo ===========================================
echo.

REM Check Python
echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ❌ Python is required but not installed
    exit /b 1
)

echo ✓ Python detected
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv .venv
echo ✓ Virtual environment created
echo.

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo ✓ Virtual environment activated
echo.

REM Install dependencies
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
echo ✓ Dependencies installed
echo.

REM Create .env if not exists
if not exist ".env" (
    echo Creating .env file...
    (
        echo # API Keys
        echo API_KEY_SPORTS=your_api_key_here
        echo API_KEY_FOOTBALL_DATA=your_api_key_here
        echo.
        echo # Settings
        echo LOG_LEVEL=INFO
        echo SCHEDULER_ENABLED=True
    ) > .env
    echo ✓ Created .env ^(edit with your API keys^)
) else (
    echo ✓ .env already exists
)

echo.
echo ===========================================
echo ✅ Installation completed!
echo ===========================================
echo.
echo Next steps:
echo 1. Edit .env file with your API keys
echo 2. Activate venv: .venv\Scripts\activate.bat
echo 3. Run: streamlit run main.py
echo.
echo Get API keys from:
echo   - https://dashboard.api-sports.io/register
echo   - https://www.football-data.org/client/register
echo.
pause
