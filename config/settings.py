"""
Глобальные настройки приложения
"""
import os
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime

# Загрузка переменных из .env
load_dotenv()

# Пути
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data_cache"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
DATABASE_DIR = PROJECT_ROOT / "database"

# Создание директорий при необходимости
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, DATABASE_DIR]:
    dir_path.mkdir(exist_ok=True)

# API Ключи
API_KEY_SPORTS = os.getenv("API_KEY_SPORTS", "")
API_KEY_FOOTBALL_DATA = os.getenv("API_KEY_FOOTBALL_DATA", "")

# API Endpoints
API_SPORTS_HOST = "v3.football.api-sports.io"
API_SPORTS_BASE_URL = f"https://{API_SPORTS_HOST}"
API_FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"
API_SPORTS_MIN_REQUEST_INTERVAL_SECONDS = float(os.getenv("API_SPORTS_MIN_REQUEST_INTERVAL_SECONDS", "1.2"))
API_SPORTS_RATE_LIMIT_COOLDOWN_SECONDS = float(os.getenv("API_SPORTS_RATE_LIMIT_COOLDOWN_SECONDS", "20"))

# Заголовки для API
HEADERS_SPORTS = {'x-apisports-key': API_KEY_SPORTS}
HEADERS_FOOTBALL_DATA = {'X-Auth-Token': API_KEY_FOOTBALL_DATA}


def get_current_football_season(current_date: datetime | None = None) -> int:
    """Вернуть стартовый год текущего футбольного сезона для европейского календаря."""
    current_date = current_date or datetime.now()
    return current_date.year if current_date.month >= 7 else current_date.year - 1

# Параметры кэширования
CACHE_TTL_SHORT = 30 * 60  # 30 минут для LIVE данных
CACHE_TTL_MEDIUM = 5 * 60 * 60  # 5 часов для статистики
CACHE_TTL_LONG = 86400  # 24 часа для истории

# Параметры обработки данных
# В Rudy-only режиме оставляем только глубину исторического окна для выборки матчей.
TRAINING_DATA_SEASONS = 3
TRAINING_DATA_SEASONS_DYNAMIC = False
TRAINING_DATA_SEASON_MIN_THRESHOLD = 300

# Инкрементальное обучение отключено (оставлено для обратной совместимости импорта).
INCREMENTAL_LEARNING_ENABLED = False
INCREMENTAL_LEARNING_RETRAIN_FREQUENCY_DAYS = 0
INCREMENTAL_LEARNING_RECENT_DATA_WINDOW_DAYS = 0
MIN_MATCHES_HISTORY = 5  # Минимум матчей для расчета формы
H2H_MATCHES_WINDOW = int(os.getenv("H2H_MATCHES_WINDOW", "5"))  # Последние личные встречи
ROLLING_WINDOW = 10  # Окно для скользящего среднего
TEMPORAL_DECAY_ALPHA = float(os.getenv("TEMPORAL_DECAY_ALPHA", "0.65"))

# Пороговая политика публикации прогноза
PREDICTION_ABSTAIN_ENABLED = os.getenv("PREDICTION_ABSTAIN_ENABLED", "true").lower() == "true"
PREDICTION_MIN_CONFIDENCE = float(os.getenv("PREDICTION_MIN_CONFIDENCE", "0.50"))
PREDICTION_MIN_MARGIN = float(os.getenv("PREDICTION_MIN_MARGIN", "0.08"))
PREDICTION_POLICY_MIN_COVERAGE = float(os.getenv("PREDICTION_POLICY_MIN_COVERAGE", "0.60"))
PREDICTION_POLICY_MIN_ACCURACY_GAIN = float(os.getenv("PREDICTION_POLICY_MIN_ACCURACY_GAIN", "0.015"))
PREDICTION_POLICY_MAX_COVERAGE = float(os.getenv("PREDICTION_POLICY_MAX_COVERAGE", "0.75"))
PREDICTION_POLICY_CONFIDENCE_FLOOR = float(os.getenv("PREDICTION_POLICY_CONFIDENCE_FLOOR", "0.58"))
PREDICTION_POLICY_MARGIN_FLOOR = float(os.getenv("PREDICTION_POLICY_MARGIN_FLOOR", "0.10"))
PREDICTION_POLICY_COVERAGE_PENALTY = float(os.getenv("PREDICTION_POLICY_COVERAGE_PENALTY", "0.10"))

# Продовая стратегия выбора базовой модели и гибридного инференса
MODELS_ENABLED = False
PROD_PRIMARY_MODEL_STRATEGY = os.getenv("PROD_PRIMARY_MODEL_STRATEGY", "rudy")
HYBRID_LR_ENSEMBLE_WEIGHT = float(os.getenv("HYBRID_LR_ENSEMBLE_WEIGHT", "0.68"))
HYBRID_PREDICTION_CONFIDENCE_FLOOR = float(os.getenv("HYBRID_PREDICTION_CONFIDENCE_FLOOR", "0.577"))

# Более строгий отбор post-hoc calibration, ориентированный на надежность сигнала
CALIBRATION_MIN_BRIER_IMPROVEMENT = float(os.getenv("CALIBRATION_MIN_BRIER_IMPROVEMENT", "0.002"))
CALIBRATION_MAX_ECE_DEGRADATION = float(os.getenv("CALIBRATION_MAX_ECE_DEGRADATION", "0.0"))

# Rolling / time-based CV отключен в Rudy-only режиме
OPTUNA_TIME_SERIES_SPLITS = 0
ENSEMBLE_OOF_SPLITS = int(os.getenv("ENSEMBLE_OOF_SPLITS", "4"))

# Rudy-only проект использует встроенные признаки h2h и forma на базе локальной истории.
# Группы признаков удалены - оставлена простая Rudy-only конфигурация.

# Параметры enrichment матчевых деталей
DETAIL_BACKFILL_ENABLED = os.getenv("DETAIL_BACKFILL_ENABLED", "true").lower() == "true"
DETAIL_BACKFILL_MAX_FIXTURES_PER_RUN = int(os.getenv("DETAIL_BACKFILL_MAX_FIXTURES_PER_RUN", "1000"))
DETAIL_BACKFILL_MAX_FIXTURES_PER_UPDATE = int(os.getenv("DETAIL_BACKFILL_MAX_FIXTURES_PER_UPDATE", "120"))
DETAIL_BACKFILL_INCLUDE_STATISTICS = os.getenv("DETAIL_BACKFILL_INCLUDE_STATISTICS", "true").lower() == "true"
DETAIL_BACKFILL_INCLUDE_ODDS = os.getenv("DETAIL_BACKFILL_INCLUDE_ODDS", "true").lower() == "true"
BACKFILL_QUEUE_BATCH_SIZE = int(os.getenv("BACKFILL_QUEUE_BATCH_SIZE", "120"))
BACKFILL_QUEUE_MAX_ATTEMPTS = int(os.getenv("BACKFILL_QUEUE_MAX_ATTEMPTS", "5"))
BACKFILL_DEFAULT_COOLDOWN_MINUTES = int(os.getenv("BACKFILL_DEFAULT_COOLDOWN_MINUTES", "60"))
BACKFILL_RATE_LIMIT_COOLDOWN_MINUTES = int(os.getenv("BACKFILL_RATE_LIMIT_COOLDOWN_MINUTES", "240"))
BACKFILL_INCOMPLETE_COOLDOWN_MINUTES = int(os.getenv("BACKFILL_INCOMPLETE_COOLDOWN_MINUTES", "90"))

# Параметры live in-play модели
IN_PLAY_MAX_MINUTE = 95.0
IN_PLAY_SCORE_WEIGHT_BASE = 1.1
IN_PLAY_SCORE_WEIGHT_LATE = 2.4
IN_PLAY_DRAW_TIED_BASE = 0.1
IN_PLAY_DRAW_TIED_LATE = 1.8
IN_PLAY_DRAW_DEFICIT_BASE = 0.7
IN_PLAY_DRAW_DEFICIT_LATE = 2.6
IN_PLAY_PRESSURE_WEIGHT_BASE = 0.18
IN_PLAY_PRESSURE_WEIGHT_LATE = 0.45
IN_PLAY_PRESSURE_DRAW_PENALTY = 0.18
IN_PLAY_SHOTS_ON_TARGET_WEIGHT = 0.55
IN_PLAY_TOTAL_SHOTS_WEIGHT = 0.12
IN_PLAY_POSSESSION_WEIGHT = 0.08
IN_PLAY_CORNERS_WEIGHT = 0.10

# Параметры сохранения live snapshots
LIVE_SNAPSHOTS_ENABLED = os.getenv("LIVE_SNAPSHOTS_ENABLED", "true").lower() == "true"
LIVE_SNAPSHOT_MIN_MINUTE = int(os.getenv("LIVE_SNAPSHOT_MIN_MINUTE", "1"))
LIVE_SNAPSHOT_MAX_MINUTE = int(os.getenv("LIVE_SNAPSHOT_MAX_MINUTE", "120"))
LIVE_SNAPSHOT_POLL_MINUTES = int(os.getenv("LIVE_SNAPSHOT_POLL_MINUTES", "5"))
IN_PLAY_ML_MIN_ROWS = int(os.getenv("IN_PLAY_ML_MIN_ROWS", "50"))
IN_PLAY_ML_RETRAIN_MIN_NEW_LABELS = int(os.getenv("IN_PLAY_ML_RETRAIN_MIN_NEW_LABELS", "10"))
IN_PLAY_ML_RETRAIN_CHECK_MINUTES = int(os.getenv("IN_PLAY_ML_RETRAIN_CHECK_MINUTES", "30"))
DASHBOARD_PREDICTIONS_POLL_MINUTES = int(os.getenv("DASHBOARD_PREDICTIONS_POLL_MINUTES", "15"))
DASHBOARD_PREDICTIONS_CACHE_TTL = int(os.getenv("DASHBOARD_PREDICTIONS_CACHE_TTL", "1800"))

# Параметры рейтингов и силы календаря
ELO_BASE_RATING = 1500.0
ELO_K_FACTOR = 20.0
ELO_DEFAULT_HOME_ADVANTAGE = 50.0
ELO_SEASON_CARRYOVER = 0.75
ELO_HOME_ADVANTAGE_CARRYOVER = 0.7
LEAGUE_HOME_ADVANTAGE_K_FACTOR = 6.0
MIN_LEAGUE_HOME_ADVANTAGE = 15.0
MAX_LEAGUE_HOME_ADVANTAGE = 85.0

# Подборка признаков для UI-диагностики
FEATURE_DIAGNOSTIC_COLUMNS = [
    "ranking_difference",
    "home_shots_total_avg",
    "away_shots_total_avg",
    "shots_total_difference",
    "home_shots_on_target_avg",
    "away_shots_on_target_avg",
    "shots_on_target_difference",
    "home_possession_avg",
    "away_possession_avg",
    "possession_difference",
    "home_corners_avg",
    "away_corners_avg",
    "corners_difference",
    "home_intensity_proxy_avg",
    "away_intensity_proxy_avg",
    "intensity_proxy_difference",
    "home_halftime_goal_diff_avg",
    "away_halftime_goal_diff_avg",
    "halftime_goal_diff_difference",
    "home_odds",
    "draw_odds",
    "away_odds",
    "implied_home_probability",
    "implied_draw_probability",
    "implied_away_probability",
    "market_overround",
    "market_home_edge",
    "normalized_home_probability",
    "normalized_draw_probability",
    "normalized_away_probability",
    "market_favorite_probability",
    "market_probability_gap",
    "poisson_home_xg",
    "poisson_away_xg",
    "poisson_total_xg",
    "poisson_goal_diff_xg",
    "poisson_home_win_probability",
    "poisson_draw_probability",
    "poisson_away_win_probability",
    "odds_data_available",
    "statistics_data_available",
    "home_form",
    "away_form",
    "form_difference",
    "home_weighted_form",
    "away_weighted_form",
    "weighted_form_difference",
    "home_weighted_goal_diff_avg",
    "away_weighted_goal_diff_avg",
    "home_win_rate",
    "away_win_rate",
    "win_rate_difference",
    "home_elo",
    "away_elo",
    "league_home_advantage",
    "elo_difference",
    "contextual_elo_difference",
    "home_strength_of_schedule",
    "away_strength_of_schedule",
    "strength_of_schedule_difference",
    "home_adjusted_points_avg",
    "away_adjusted_points_avg",
    "adjusted_points_difference",
    "home_adjusted_goal_diff_avg",
    "away_adjusted_goal_diff_avg",
    "adjusted_goal_diff_difference",
    "home_top_tier_points_avg",
    "away_top_tier_points_avg",
    "top_tier_points_difference",
    "home_mid_tier_points_avg",
    "away_mid_tier_points_avg",
    "mid_tier_points_difference",
    "home_bottom_tier_points_avg",
    "away_bottom_tier_points_avg",
    "bottom_tier_points_difference",
    "h2h_matches_played",
    "h2h_home_points_avg",
    "h2h_away_points_avg",
    "h2h_points_difference",
    "h2h_home_win_rate",
    "h2h_away_win_rate",
    "h2h_win_rate_difference",
    "h2h_draw_rate",
    "h2h_goal_diff_avg",
    "h2h_total_goals_avg",
    "h2h_home_goals_avg",
    "h2h_away_goals_avg",
]

# Параметры моделей
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
RANDOM_STATE = 42
MAX_FEATURE_MISSING_RATIO = float(os.getenv("MAX_FEATURE_MISSING_RATIO", "0.35"))
MIN_FEATURE_VARIANCE = float(os.getenv("MIN_FEATURE_VARIANCE", "0.0001"))
FEATURE_SEASON_INSTABILITY_MAX = float(os.getenv("FEATURE_SEASON_INSTABILITY_MAX", "0.65"))
FEATURE_SEASON_MISSING_STD_MAX = float(os.getenv("FEATURE_SEASON_MISSING_STD_MAX", "0.22"))
FEATURE_MIN_SEASON_ROWS = int(os.getenv("FEATURE_MIN_SEASON_ROWS", "40"))

# Training-параметры качества удалены в Rudy-only режиме.
# Используются только для совместимости со старыми конфигами.

# Optuna-параметры удалены - в Rudy-only проекте нет оптимизации гиперпараметров.

# Параметры boosting/ensemble/LightGBM/XGBoost удалены - не используются в Rudy-only режиме.

# Параметры расписания обновления данных
SCHEDULER_ENABLED = os.getenv("SCHEDULER_ENABLED", "true").lower() == "true"
SCHEDULE_TIME_1 = os.getenv("SCHEDULE_TIME_1", "00:00")  # Первая загрузка (UTC)
SCHEDULE_TIME_2 = os.getenv("SCHEDULE_TIME_2", "12:00")  # Вторая загрузка (UTC)
# Параметры переобучения отключены, оставлены только для совместимости со старыми конфигами.
MODEL_RETRAIN_SCHEDULE_TIME = os.getenv("MODEL_RETRAIN_SCHEDULE_TIME", "00:00")
MODEL_RETRAIN_FREQUENCY_HOURS = 0

# Логирование
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Лиги для отслеживания
TRACKED_LEAGUES = {
    310: "Superliga",
    311: "1st Division",
    312: "1a Divisió",
    342: "Premier League",
    218: "Bundesliga",
    219: "2. Liga",
    419: "Premyer Liqa",
    418: "Birinci Dasta",
    116: "Premier League",
    144: "Jupiler Pro League",
    145: "Challenger Pro League",
    315: "Premijer Liga",
    316: "1st League - FBiH",
    172: "First League",
    173: "Second League",
    210: "HNL",
    211: "First NL",
    318: "1. Division",
    319: "2. Division",
    119: "Superliga",
    120: "1. Division",
    39: "Premier League",
    40: "Championship",
    329: "Meistriliiga",
    328: "Esiliiga A",
    244: "Veikkausliiga",
    1087: "Ykkösliiga",
    61: "Ligue 1",
    62: "Ligue 2",
    327: "Erovnuli Liga",
    326: "Erovnuli Liga 2",
    78: "Bundesliga",
    79: "2. Bundesliga",
    197: "Super League 1",
    494: "Super League 2",
    271: "NB I",
    272: "NB II",
    164: "Úrvalsdeild",
    357: "Premier Division",
    358: "First Division",
    383: "Ligat Ha'al",
    382: "Liga Leumit",
    135: "Serie A",
    136: "Serie B",
    664: "Superliga",
    365: "Virsliga",
    362: "A Lyga",
    361: "1 Lyga",
    261: "National Division",
    393: "Premier League",
    392: "Challenge League",
    394: "Super Liga",
    355: "First League",
    356: "Second League",
    88: "Eredivisie",
    89: "Eerste Divisie",
    103: "Eliteserien",
    104: "1. Division",
    106: "Ekstraklasa",
    107: "I Liga",
    94: "Primeira Liga",
    95: "Segunda Liga",
    283: "Liga I",
    284: "Liga II",
    235: "Premier League",
    236: "First League",
    179: "Premiership",
    180: "Championship",
    286: "Super Liga",
    287: "Prva Liga",
    332: "Super Liga",
    506: "2. liga",
    373: "1. SNL",
    374: "2. SNL",
    140: "La Liga",
    141: "Segunda División",
    113: "Allsvenskan",
    114: "Superettan",
    207: "Super League",
    208: "Challenge League",
    203: "Süper Lig",
    204: "1. Lig",
    333: "Premier League",
    334: "Persha Liga",
    110: "Premier League",
    111: "FAW Championship",
}

# Размер ставки (стартовый банк объеме точек)
INITIAL_BANKROLL = 1000
MIN_IMPLIED_ODDS = 1.2  # Минимальные odds для ставки
MAX_IMPLIED_ODDS = 10.0  # Максимальные odds для ставки
