#!/bin/bash
# Installation and setup script for Football Prediction System

echo "🚀 Football Predictions System Installation"
echo "==========================================="
echo ""

# Check Python
echo "Checking Python installation..."
python3 --version

if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is required but not installed"
    exit 1
fi

echo "✓ Python detected"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv

echo "✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✓ Dependencies installed"
echo ""

# Create .env if not exists
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# API Keys
API_KEY_SPORTS=your_api_key_here
API_KEY_FOOTBALL_DATA=your_api_key_here

# Settings
LOG_LEVEL=INFO
SCHEDULER_ENABLED=True
EOF
    echo "✓ Created .env (edit with your API keys)"
else
    echo "✓ .env already exists"
fi

echo ""
echo "==========================================="
echo "✅ Installation completed!"
echo "==========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Activate venv: source .venv/bin/activate"
echo "3. Run: streamlit run main.py"
echo ""
echo "Get API keys from:"
echo "  - https://dashboard.api-sports.io/register"
echo "  - https://www.football-data.org/client/register"
echo ""
