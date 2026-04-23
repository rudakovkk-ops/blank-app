#!/bin/bash
# Run the Streamlit application

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Виртуальное окружение не найдено. Сначала запустите install.sh."
    exit 1
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠ Файл .env не найден"
    exit 1
fi

# Check if API keys are configured
if grep -q "your_api_key_here" .env; then
    echo "⚠ Предупреждение: API-ключи не настроены в .env"
    echo "Приложение запустится, но не сможет получить реальные данные из API"
    echo ""
fi

# Run the app
echo "🚀 Запуск системы футбольных прогнозов..."
echo "Открывайте http://localhost:8501"
echo ""
echo "Нажмите Ctrl+C для остановки"
echo ""

streamlit run main.py
