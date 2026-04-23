# Football Match Predictions System

Канонический интерфейс проекта - Streamlit-приложение из [main.py](main.py). Остальной код обслуживает этот UI: загрузку данных, live-обновления, Rudy-инференс, backfill и диагностику.

## Что делает проект

- Показывает матчи на сегодня и live-матчи с прогнозами исходов.
- Строит прогнозы исходов на базе rule-based модели Rudy.
- Показывает live snapshot-статистику и состояние backfill-очереди.
- Поддерживает фоновое обновление данных и подготовку dashboard-кэша.

## Канонический сценарий запуска

1. Создайте виртуальное окружение и установите зависимости.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Создайте локальный файл окружения.

```bash
cp .env.example .env
```

3. Заполните в `.env` API-ключи.

Обязательные переменные:

```env
API_KEY_SPORTS=your_api_sports_key
API_KEY_FOOTBALL_DATA=your_football_data_key
```

4. Запустите основной интерфейс.

```bash
streamlit run main.py
```

Или через вспомогательный скрипт:

```bash
bash run.sh
```

Единственная поддерживаемая точка входа интерфейса: [main.py](main.py).

## Основные команды

Запуск UI:

```bash
streamlit run main.py
```

## Структура проекта

```text
blank-app/
├── main.py
├── prediction_service.py
├── config/
├── data/
├── evaluation/
├── models/
├── scheduler/
├── README.md
├── README_NEW.md
└── QUICKSTART.md
```

Ключевые модули:

- [main.py](main.py): канонический Streamlit UI.
- [prediction_service.py](prediction_service.py): Rudy-only инференс и сбор диагностик для UI.
- [data/api_client.py](data/api_client.py): работа с API-Sports и enrichment match details.
- [data/data_service.py](data/data_service.py): кэш и выдача live/today fixtures для интерфейса.
- [scheduler/auto_updater.py](scheduler/auto_updater.py): обновление данных, сбор live snapshots, фоновая подготовка dashboard.

## Возможности интерфейса

- `Dashboard`: матчи на сегодня и сводные метрики основной модели.
- `Live Predictions`: live-таблица матчей со счётом, минутой, odds, shot/possession статистикой и in-play прогнозом.
- `Models`: сводка текущей Rudy-модели.
- `Feature Diagnostics`: диагностический блок для выбранного матча.
- `ROI Analysis`: обзор ROI и betting-performance.
- `Settings`: cache management, feature-engineering параметры, snapshot/backfill статус.
- `About`: краткая справка по системе.

## Переменные окружения

Ключевые настройки в `.env` и [config/settings.py](config/settings.py):

```env
API_KEY_SPORTS=...
API_KEY_FOOTBALL_DATA=...
LOG_LEVEL=INFO
SCHEDULER_ENABLED=True
LIVE_SNAPSHOTS_ENABLED=true
```

## Источник правды по документации

[README.md](README.md) - канонический файл документации.

[README_NEW.md](README_NEW.md) и [QUICKSTART.md](QUICKSTART.md) сохранены только как короткие указатели на этот файл для обратной совместимости.
