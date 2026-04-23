"""
Клиент для работы с API Football (api-sports.io)
"""
import requests
import logging
from copy import deepcopy
from threading import Lock
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from time import monotonic, sleep
from config.settings import (
    API_SPORTS_BASE_URL,
    API_SPORTS_MIN_REQUEST_INTERVAL_SECONDS,
    API_SPORTS_RATE_LIMIT_COOLDOWN_SECONDS,
    HEADERS_SPORTS,
    TRACKED_LEAGUES,
)

logger = logging.getLogger(__name__)


class FootballAPIClient:
    """Клиент для API-Sports"""

    STAT_ALIASES = {
        'shots on goal': 'shots_on_target',
        'shots off goal': 'shots_off_target',
        'total shots': 'total_shots',
        'blocked shots': 'blocked_shots',
        'ball possession': 'possession',
        'corner kicks': 'corners',
        'offsides': 'offsides',
        'goalkeeper saves': 'goalkeeper_saves',
        'total passes': 'total_passes',
        'passes accurate': 'accurate_passes',
        'fouls': 'fouls',
        'yellow cards': 'yellow_cards',
        'red cards': 'red_cards',
        'tackles': 'tackles',
        'interceptions': 'interceptions',
    }
    
    def __init__(self, base_url: str = API_SPORTS_BASE_URL, headers: Dict = None):
        self.base_url = base_url
        self.headers = headers or HEADERS_SPORTS
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.max_retries = 3
        self.retry_delay = 1
        self.min_request_interval_seconds = max(0.0, API_SPORTS_MIN_REQUEST_INTERVAL_SECONDS)
        self.rate_limit_cooldown_seconds = max(
            self.min_request_interval_seconds,
            API_SPORTS_RATE_LIMIT_COOLDOWN_SECONDS,
        )
        self.last_error_reason = None
        self._request_timing_lock = Lock()
        self._next_request_ts = 0.0

    @staticmethod
    def _is_rate_limit_payload(errors: Any) -> bool:
        """Определить, сообщает ли payload об ограничении частоты запросов."""
        if not errors:
            return False

        if isinstance(errors, dict):
            haystack = " ".join(str(value) for value in errors.values()).lower()
            keys = " ".join(str(key) for key in errors.keys()).lower()
            haystack = f"{keys} {haystack}"
        else:
            haystack = str(errors).lower()

        return 'ratelimit' in haystack or 'rate limit' in haystack or 'too many requests' in haystack

    def _wait_for_request_slot(self) -> None:
        """Соблюдать минимальный интервал между запросами ко всему API-клиенту."""
        sleep_for = 0.0
        with self._request_timing_lock:
            now = monotonic()
            if self._next_request_ts > now:
                sleep_for = self._next_request_ts - now
                now = self._next_request_ts
            self._next_request_ts = now + self.min_request_interval_seconds

        if sleep_for > 0:
            sleep(sleep_for)

    def _apply_rate_limit_backoff(self, retry_after: Optional[Any] = None) -> float:
        """Отложить следующие запросы после rate limit и вернуть фактическую паузу."""
        try:
            retry_after_seconds = float(retry_after) if retry_after is not None else 0.0
        except (TypeError, ValueError):
            retry_after_seconds = 0.0

        cooldown = max(self.rate_limit_cooldown_seconds, retry_after_seconds)
        with self._request_timing_lock:
            self._next_request_ts = max(self._next_request_ts, monotonic() + cooldown)
        return cooldown
    
    def _get(
        self, 
        endpoint: str, 
        params: Optional[Dict] = None,
        timeout: int = 15
    ) -> Optional[Dict]:
        """
        Выполнить GET запрос к API
        
        Args:
            endpoint: Путь к endpoint'у
            params: Параметры запроса
            timeout: Таймаут в секундах
        
        Returns:
            Ответ API или None в случае ошибки
        """
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                self._wait_for_request_slot()
                response = self.session.get(url, params=params, timeout=timeout)
                
                # Успешный ответ
                if response.status_code == 200:
                    data = response.json()
                    if data.get("errors"):
                        logger.warning(f"API Warning: {data.get('errors')}")
                        if self._is_rate_limit_payload(data.get("errors")):
                            self.last_error_reason = "rate_limit"
                            retry_after = self._apply_rate_limit_backoff(
                                response.headers.get('Retry-After')
                            )
                            logger.warning(
                                "API payload reported rate limit. Retrying in %.0fs...",
                                retry_after,
                            )
                            continue

                    self.last_error_reason = None
                    return data
                
                # Rate limit - пауза и повтор
                elif response.status_code == 429:
                    self.last_error_reason = "rate_limit"
                    retry_after = self._apply_rate_limit_backoff(
                        response.headers.get('Retry-After', self.retry_delay)
                    )
                    logger.warning(f"Rate limit. Retrying in {retry_after:.0f}s...")
                    continue
                
                # Другие ошибки
                else:
                    self.last_error_reason = f"http_{response.status_code}"
                    logger.error(f"API Error {response.status_code} on attempt {attempt + 1}: {response.text}")
                    if attempt < self.max_retries - 1:
                        sleep(self.retry_delay * (attempt + 1))
                    continue
                    
            except requests.exceptions.Timeout:
                self.last_error_reason = "timeout"
                logger.error(f"Timeout на {endpoint} (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    sleep(self.retry_delay * (attempt + 1))
                continue
            except requests.exceptions.ConnectionError as e:
                self.last_error_reason = "connection_error"
                logger.error(f"Connection Error: {e} (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    sleep(self.retry_delay * (attempt + 1))
                continue
            except Exception as e:
                self.last_error_reason = "unexpected_error"
                logger.error(f"Unexpected error on {endpoint}: {e}")
                return None
        
        logger.error(f"Failed to get {endpoint} after {self.max_retries} attempts")
        return None
    
    def get_live_fixtures(self) -> List[Dict]:
        """
        Получить все LIVE матчи
        
        Returns:
            Список матчей
        """
        response = self._get("fixtures", params={"live": "all"})
        return response.get('response', []) if response else []
    
    def get_fixtures(
        self, 
        league: Optional[int] = None,
        season: Optional[int] = None,
        date: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict]:
        """
        Получить матчи с фильтрацией
        
        Args:
            league: ID лиги
            season: Сезон
            date: Дата в формате YYYY-MM-DD
            status: Статус матча (LIVE, FINISHED и т.д.)
        
        Returns:
            Список матчей
        """
        params = {}
        if league:
            params['league'] = league
        if season:
            params['season'] = season
        if date:
            params['date'] = date
        if status:
            params['status'] = status
        
        response = self._get("fixtures", params=params)
        return response.get('response', []) if response else []
    
    def get_fixture_statistics(self, fixture_id: int) -> Dict:
        """
        Получить статистику матча
        
        Args:
            fixture_id: ID матча
        
        Returns:
            Статистика матча
        """
        response = self._get("fixtures/statistics", params={"fixture": fixture_id})
        return response.get('response', []) if response else []
    
    def get_fixture_events(self, fixture_id: int) -> List[Dict]:
        """
        Получить события матча (голы, карточки и т.д.)
        
        Args:
            fixture_id: ID матча
        
        Returns:
            Список событий
        """
        response = self._get("fixtures/events", params={"fixture": fixture_id})
        return response.get('response', []) if response else []
    
    def get_teams(self, league: int, season: int) -> List[Dict]:
        """
        Получить команды лиги
        
        Args:
            league: ID лиги
            season: Сезон
        
        Returns:
            Список команд
        """
        response = self._get("teams", params={"league": league, "season": season})
        return response.get('response', []) if response else []
    
    def get_standings(self, league: int, season: int) -> List[Dict]:
        """
        Получить турнирную таблицу
        
        Args:
            league: ID лиги
            season: Сезон
        
        Returns:
            Турнирная таблица
        """
        response = self._get("standings", params={"league": league, "season": season})
        standings = response.get('response', []) if response else []
        
        if standings and len(standings) > 0:
            return standings[0].get('league', {}).get('standings', [])
        return []
    
    def get_odds(self, fixture_id: int, bookmaker: Optional[str] = None) -> List[Dict]:
        """
        Получить коэффициенты на матч
        
        Args:
            fixture_id: ID матча
            bookmaker: Букмекер (опционально)
        
        Returns:
            Список коэффициентов
        """
        params = {"fixture": fixture_id}
        if bookmaker:
            params['bookmaker'] = bookmaker
        
        response = self._get("odds", params=params)
        return response.get('response', []) if response else []

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        """Преобразовать числовое значение из API в float."""
        if value is None:
            return None

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            normalized = value.strip().replace('%', '').replace(',', '.')
            if not normalized:
                return None
            try:
                return float(normalized)
            except ValueError:
                return None

        return None

    @classmethod
    def _normalize_statistics_payload(cls, statistics: List[Dict]) -> Dict[int, Dict[str, float]]:
        """Нормализовать statistics endpoint в словарь по team_id."""
        normalized_stats = {}

        for team_stats in statistics or []:
            team_id = team_stats.get('team', {}).get('id')
            if team_id is None:
                continue

            team_payload = {}
            for stat in team_stats.get('statistics', []):
                stat_name = str(stat.get('type', '')).strip().lower()
                stat_key = cls.STAT_ALIASES.get(stat_name)
                if stat_key is None:
                    continue

                stat_value = cls._safe_float(stat.get('value'))
                if stat_value is not None:
                    team_payload[stat_key] = stat_value

            normalized_stats[team_id] = team_payload

        return normalized_stats

    @classmethod
    def _extract_match_winner_odds(cls, odds_response: List[Dict]) -> Dict[str, float]:
        """
        Извлечь только коэффициенты 1X2 (P1/Draw/P2).
        Берем исключительно рынок Match Winner, все остальные игнорируем.
        """
        result = {}
        bookmaker_name = 'Unknown'
        
        for provider in odds_response or []:
            bookmakers = provider.get('bookmakers', [])
            for bookmaker in bookmakers:
                bookmaker_name = bookmaker.get('name', 'Unknown')
                for bet in bookmaker.get('bets', []):
                    bet_name = str(bet.get('name', '')).strip().lower()
                    
                    # MATCH WINNER (P1/Draw/P2)
                    if bet_name == 'match winner':
                        values = {
                            str(item.get('value', '')).strip().lower(): cls._safe_float(item.get('odd'))
                            for item in bet.get('values', [])
                        }
                        if values.get('home') and values.get('draw') and values.get('away'):
                            result['1'] = values['home']
                            result['X'] = values['draw']
                            result['2'] = values['away']
        
        # Возвращаем только если собран полный 1X2 набор.
        if '1' in result and 'X' in result and '2' in result:
            result['bookmaker'] = bookmaker_name
            return result
        
        return {}

    def enrich_fixture_details(
        self,
        fixture: Dict,
        include_statistics: bool = True,
        include_odds: bool = True,
        force: bool = False,
    ) -> Dict:
        """Дозагрузить statistics и odds для одного fixture и вернуть обогащенный payload."""
        fixture_data = deepcopy(fixture)
        fixture_id = fixture_data.get('fixture', {}).get('id')
        if fixture_id is None:
            return fixture_data

        detail_state = fixture_data.get('details_loaded', {}).copy()
        detail_errors = fixture_data.get('detail_errors', {}).copy()
        home_team_id = fixture_data.get('teams', {}).get('home', {}).get('id')
        away_team_id = fixture_data.get('teams', {}).get('away', {}).get('id')

        if include_statistics and (force or not detail_state.get('statistics')):
            statistics_response = self.get_fixture_statistics(fixture_id)
            normalized_stats = self._normalize_statistics_payload(statistics_response)
            statistics_payload = {
                'home': normalized_stats.get(home_team_id, {}),
                'away': normalized_stats.get(away_team_id, {}),
            }
            if statistics_payload['home'] or statistics_payload['away']:
                fixture_data['statistics'] = statistics_payload
                detail_state['statistics'] = True
                detail_errors.pop('statistics', None)
            else:
                detail_state['statistics'] = bool(
                    fixture_data.get('statistics', {}).get('home')
                    or fixture_data.get('statistics', {}).get('away')
                )
                if not detail_state['statistics'] and self.last_error_reason:
                    detail_errors['statistics'] = self.last_error_reason

        if include_odds and (force or not detail_state.get('odds')):
            odds_response = self.get_odds(fixture_id)
            normalized_odds = self._extract_match_winner_odds(odds_response)
            if normalized_odds:
                fixture_data['odds'] = normalized_odds
                detail_state['odds'] = True
                detail_errors.pop('odds', None)
            else:
                current_odds = fixture_data.get('odds', {})
                detail_state['odds'] = all(current_odds.get(key) is not None for key in ('1', 'X', '2'))
                if not detail_state['odds'] and self.last_error_reason:
                    detail_errors['odds'] = self.last_error_reason

        fixture_data['details_loaded'] = detail_state
        if detail_errors:
            fixture_data['detail_errors'] = detail_errors
        elif 'detail_errors' in fixture_data:
            fixture_data.pop('detail_errors', None)
        return fixture_data
    
    def get_seasons(self) -> List[int]:
        """
        Получить доступные сезоны
        
        Returns:
            Список годов сезонов
        """
        response = self._get("leagues/seasons")
        return response.get('response', []) if response else []
    
    def get_head_to_head(self, team1_id: int, team2_id: int, last: int = 10) -> List[Dict]:
        """
        Получить историю встреч между командами
        
        Args:
            team1_id: ID первой команды
            team2_id: ID второй команды
            last: Количество последних встреч
        
        Returns:
            Список встреч
        """
        response = self._get(
            "fixtures/headtohead",
            params={"h2h": f"{team1_id}-{team2_id}", "last": last}
        )
        return response.get('response', []) if response else []
    
    def get_team_stats(
        self,
        team_id: int,
        season: int,
        league: int
    ) -> Dict:
        """
        Получить статистику команды за сезон
        
        Args:
            team_id: ID команды
            season: Сезон
            league: ID лиги
        
        Returns:
            Словарь со статистикой
        """
        fixtures = self.get_fixtures(league=league, season=season)
        
        team_stats = {
            'played': 0,
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'goals_for': 0,
            'goals_against': 0,
            'goal_difference': 0,
        }
        
        for match in fixtures:
            if match.get('fixture', {}).get('status', {}).get('short') != 'FT':
                continue
            
            teams = match.get('teams', {})
            goals = match.get('goals', {})
            
            is_home = teams.get('home', {}).get('id') == team_id
            is_away = teams.get('away', {}).get('id') == team_id
            
            if is_home or is_away:
                team_stats['played'] += 1
                
                home_goals = goals.get('home')
                away_goals = goals.get('away')
                
                if is_home:
                    team_stats['goals_for'] += home_goals
                    team_stats['goals_against'] += away_goals
                    
                    if home_goals > away_goals:
                        team_stats['wins'] += 1
                    elif home_goals == away_goals:
                        team_stats['draws'] += 1
                    else:
                        team_stats['losses'] += 1
                else:
                    team_stats['goals_for'] += away_goals
                    team_stats['goals_against'] += home_goals
                    
                    if away_goals > home_goals:
                        team_stats['wins'] += 1
                    elif away_goals == home_goals:
                        team_stats['draws'] += 1
                    else:
                        team_stats['losses'] += 1
        
        team_stats['goal_difference'] = (
            team_stats['goals_for'] - team_stats['goals_against']
        )
        
        return team_stats
    
    def get_fixtures_range(
        self,
        league: int,
        season: int,
        date_from: str,
        date_to: str
    ) -> List[Dict]:
        """
        Получить матчи за диапазон дат
        
        Args:
            league: ID лиги
            season: Сезон
            date_from: Дата начала (YYYY-MM-DD)
            date_to: Дата конца (YYYY-MM-DD)
        
        Returns:
            Список матчей
        """
        params = {
            'league': league,
            'season': season,
            'from': date_from,
            'to': date_to,
            'status': 'FT'  # Только завершенные матчи
        }
        
        response = self._get("fixtures", params=params)
        return response.get('response', []) if response else []
    
    def close(self):
        """Закрыть сессию"""
        self.session.close()
