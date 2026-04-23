"""
Константы приложения
"""

# Типы результатов матча
MATCH_OUTCOMES = {
    "1": "Home Win",
    "X": "Draw",
    "2": "Away Win",
}

# Статусы матча
MATCH_STATUS = {
    "NOT_STARTED": "not_started",
    "LIVE": "live",
    "FINISHED": "finished",
    "POSTPONED": "postponed",
}

# Метрики моделей
METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "logloss",
    "roi",
    "sharpe_ratio",
    "max_drawdown",
]

# Названия моделей
MODEL_NAMES = {
    "rudy": "Rudy",
    "logistic_regression": "Логистическая регрессия",
    "random_forest": "Случайный лес",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "neural_network": "Нейросеть",
    "ensemble": "Ансамбль",
    "bayesian": "Байесовский подход",
}

# Цвета для графиков
COLORS = {
    "primary": "#1f77b4",
    "success": "#2ca02c",
    "danger": "#d62728",
    "warning": "#ff7f0e",
    "info": "#17a2b8",
    "light": "#f8f9fa",
    "dark": "#343a40",
}

# Kelly Criterion коэффициент для ставок
KELLY_FRACTION = 0.25  # 25% от Kelly Criterion

# Минимальная вероятность для ставки
MIN_PROBABILITY = 0.55  # 55%

# Типы ошибок
ERROR_TYPES = [
    "API_ERROR",
    "DATA_ERROR",
    "MODEL_ERROR",
    "CACHE_ERROR",
]
