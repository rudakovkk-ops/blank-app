"""
Сервис для получения и кеширования данных матчей
"""
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
from pathlib import Path
from data.api_client import FootballAPIClient
from config.settings import (
    CACHE_TTL_SHORT,
    CACHE_TTL_MEDIUM,
    DATA_DIR,
    TRACKED_LEAGUES,
    get_current_football_season,
)

logger = logging.getLogger(__name__)


class DataService:
    """Сервис для управления данными с кешированием"""
    
    def __init__(self):
        self.api_client = FootballAPIClient()
        self.cache_dir = DATA_DIR / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_ttl = {
            'live': CACHE_TTL_SHORT,
            'fixtures': CACHE_TTL_MEDIUM,
            'standings': CACHE_TTL_MEDIUM,
        }
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Получить путь кеша"""
        return self.cache_dir / f"{cache_key}.json"
    
    def _is_cache_valid(self, cache_key: str, ttl: int) -> bool:
        """Проверить валидность кеша"""
        cache_path = self._get_cache_path(cache_key)
        if not cache_path.exists():
            return False
        
        try:
            mtime = cache_path.stat().st_mtime
            age = (datetime.now() - datetime.fromtimestamp(mtime)).total_seconds()
            return age < ttl
        except Exception as e:
            logger.error(f"Error checking cache: {e}")
            return False
    
    def _read_cache(self, cache_key: str) -> Optional[Dict]:
        """Прочитать кеш"""
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading cache: {e}")
            return None
    
    def _write_cache(self, cache_key: str, data: Dict) -> bool:
        """Записать кеш"""
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
            return True
        except Exception as e:
            logger.error(f"Error writing cache: {e}")
            return False
    
    def get_live_fixtures(self, use_cache: bool = True) -> List[Dict]:
        """
        Получить LIVE матчи
        
        Args:
            use_cache:使用缓存
        
        Returns:
            Список матчей
        """
        cache_key = "live_fixtures"
        
        # Проверить кеш
        if use_cache and self._is_cache_valid(cache_key, self.cache_ttl['live']):
            cached = self._read_cache(cache_key)
            if cached:
                logger.info(f"Using cached live fixtures ({len(cached.get('response', []))} matches)")
                return cached.get('response', [])
        
        # Получить с API
        logger.info("Fetching live fixtures from API...")
        fixtures = self.api_client.get_live_fixtures()
        
        # Кешировать результат
        if fixtures:
            self._write_cache(cache_key, {"response": fixtures})
        
        return fixtures
    
    def get_today_fixtures(self, league_id: Optional[int] = None, use_cache: bool = True) -> List[Dict]:
        """
        Получить матчи на сегодня
        
        Args:
            league_id: ID лиги (опционально)
            use_cache: Использовать кеш
        
        Returns:
            Список матчей
        """
        today = datetime.now().strftime("%Y-%m-%d")
        cache_key = f"fixtures_today_{league_id or 'all'}"
        
        # Проверить кеш
        if use_cache and self._is_cache_valid(cache_key, self.cache_ttl['fixtures']):
            cached = self._read_cache(cache_key)
            if cached:
                logger.info(f"Using cached fixtures for {today}")
                return cached.get('response', [])
        
        # Получить с API
        logger.info(f"Fetching fixtures for {today}...")
        params = {"date": today}
        if league_id:
            params["league"] = league_id
        
        fixtures = self.api_client.get_fixtures(**params)
        
        # Кешировать результат
        if fixtures:
            self._write_cache(cache_key, {"response": fixtures})
        
        return fixtures

    def get_today_fixtures_tracked_leagues(self, force_refresh: bool = False) -> List[Dict]:
        """Получить все сегодняшние матчи только из отслеживаемых чемпионатов."""
        today = datetime.now().strftime("%Y-%m-%d")
        cache_key = "fixtures_today_tracked_leagues"
        season_candidates = [get_current_football_season()]

        if not force_refresh and self._is_cache_valid(cache_key, self.cache_ttl['fixtures']):
            cached = self._read_cache(cache_key)
            if cached:
                logger.info(
                    "Using cached tracked-leagues fixtures for %s (%s matches)",
                    today,
                    len(cached.get('response', [])),
                )
                return cached.get('response', [])

        fixtures_by_id: Dict[int, Dict] = {}
        logger.info("Fetching tracked-leagues fixtures for %s...", today)
        for league_id in TRACKED_LEAGUES.keys():
            fixtures = []
            for season in season_candidates:
                try:
                    fixtures = self.api_client.get_fixtures(date=today, league=league_id, season=season)
                except Exception as e:
                    logger.warning(
                        "Failed to fetch fixtures for league %s season %s: %s",
                        league_id,
                        season,
                        e,
                    )
                    fixtures = []

                if fixtures:
                    break

            for fixture in fixtures or []:
                fixture_id = fixture.get('fixture', {}).get('id')
                if fixture_id is None:
                    continue
                fixtures_by_id[int(fixture_id)] = fixture

        merged_fixtures = sorted(
            fixtures_by_id.values(),
            key=lambda item: item.get('fixture', {}).get('date', ''),
        )
        self._write_cache(cache_key, {"response": merged_fixtures})
        logger.info("Tracked-leagues fixtures prepared: %s matches", len(merged_fixtures))
        return merged_fixtures
    
    def get_upcoming_fixtures(
        self, 
        days: int = 7, 
        league_id: Optional[int] = None,
        use_cache: bool = True
    ) -> List[Dict]:
        """
        Получить предстоящие матчи на N дней вперед
        
        Args:
            days: Количество дней
            league_id: ID лиги
            use_cache: Использовать кеш
        
        Returns:
            Список матчей
        """
        cache_key = f"fixtures_upcoming_{days}d_{league_id or 'all'}"
        
        if use_cache and self._is_cache_valid(cache_key, self.cache_ttl['fixtures']):
            cached = self._read_cache(cache_key)
            if cached:
                logger.info(f"Using cached upcoming fixtures")
                return cached.get('response', [])
        
        all_fixtures = []
        today = datetime.now()
        
        # Получить матчи для каждого дня
        for i in range(days):
            date = (today + timedelta(days=i)).strftime("%Y-%m-%d")
            logger.info(f"Fetching fixtures for {date}...")
            
            params = {"date": date}
            if league_id:
                params["league"] = league_id
            
            fixtures = self.api_client.get_fixtures(**params)
            if fixtures:
                all_fixtures.extend(fixtures)
        
        # Кешировать результат
        if all_fixtures:
            self._write_cache(cache_key, {"response": all_fixtures})
        
        return all_fixtures
    
    def get_standings(self, league_id: int, season: int, use_cache: bool = True) -> List[List[Dict]]:
        """
        Получить турнирную таблицу
        
        Args:
            league_id: ID лиги
            season: Сезон
            use_cache: Использовать кеш
        
        Returns:
            Турнирная таблица
        """
        cache_key = f"standings_{league_id}_{season}"
        
        if use_cache and self._is_cache_valid(cache_key, self.cache_ttl['standings']):
            cached = self._read_cache(cache_key)
            if cached:
                logger.info(f"Using cached standings for league {league_id}")
                return cached.get('response', [])
        
        logger.info(f"Fetching standings for league {league_id}...")
        standings = self.api_client.get_standings(league_id, season)
        
        if standings:
            self._write_cache(cache_key, {"response": standings})
        
        return standings
    
    def clear_cache(self, cache_pattern: Optional[str] = None) -> bool:
        """
        Очистить кеш
        
        Args:
            cache_pattern: Паттерн файлов для удаления (по умолчанию все)
        
        Returns:
            True если успешно
        """
        try:
            for cache_file in self.cache_dir.glob(f"{cache_pattern or '*'}.json"):
                cache_file.unlink()
                logger.info(f"Cleared cache: {cache_file.name}")
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
