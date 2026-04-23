"""
Управление кэшем данных (локальное хранилище)
"""
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from config.settings import (
    DATABASE_DIR,
    BACKFILL_DEFAULT_COOLDOWN_MINUTES,
    BACKFILL_RATE_LIMIT_COOLDOWN_MINUTES,
    BACKFILL_INCOMPLETE_COOLDOWN_MINUTES,
)

logger = logging.getLogger(__name__)


class CacheManager:
    """Менеджер кэша с использованием SQLite"""
    
    def __init__(self, db_path: str = None):
        """
        Инициализировать менеджер кэша
        
        Args:
            db_path: Путь к БД (по умолчанию database/cache.db)
        """
        if db_path is None:
            db_path = DATABASE_DIR / "cache.db"
        
        self.db_path = Path(db_path)
        self._init_db()
    
    def _init_db(self):
        """Инициализировать таблицы БД"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Таблица кэша
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        key TEXT UNIQUE NOT NULL,
                        value TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP,
                        ttl INTEGER
                    )
                ''')
                
                # Таблица истории загрузок
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS load_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        endpoint TEXT NOT NULL,
                        params TEXT,
                        status TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Таблица данных матчей
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS fixtures (
                        id INTEGER PRIMARY KEY,
                        fixture_id INTEGER UNIQUE NOT NULL,
                        league_id INTEGER,
                        season INTEGER,
                        date TEXT,
                        home_team_id INTEGER,
                        away_team_id INTEGER,
                        home_team TEXT,
                        away_team TEXT,
                        status TEXT,
                        home_goals INTEGER,
                        away_goals INTEGER,
                        data TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Таблица команд
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS teams (
                        id INTEGER PRIMARY KEY,
                        team_id INTEGER UNIQUE NOT NULL,
                        name TEXT,
                        league_id INTEGER,
                        season INTEGER,
                        data TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Таблица турнирной таблицы
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS standings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        league_id INTEGER,
                        season INTEGER,
                        team_id INTEGER,
                        rank INTEGER,
                        points INTEGER,
                        data TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(league_id, season, team_id)
                    )
                ''')

                # Таблица live snapshots для будущего in-play train set
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS live_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        fixture_id INTEGER NOT NULL,
                        league_id INTEGER,
                        season INTEGER,
                        home_team_id INTEGER,
                        away_team_id INTEGER,
                        status TEXT,
                        elapsed_minute INTEGER,
                        home_goals INTEGER,
                        away_goals INTEGER,
                        home_shots_on_target REAL,
                        away_shots_on_target REAL,
                        home_total_shots REAL,
                        away_total_shots REAL,
                        home_possession REAL,
                        away_possession REAL,
                        home_corners REAL,
                        away_corners REAL,
                        base_prediction TEXT,
                        in_play_prediction TEXT,
                        data TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(fixture_id, elapsed_minute)
                    )
                ''')

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS backfill_queue (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        fixture_id INTEGER UNIQUE NOT NULL,
                        league_id INTEGER,
                        season INTEGER,
                        requested_statistics INTEGER DEFAULT 1,
                        requested_odds INTEGER DEFAULT 1,
                        status TEXT DEFAULT 'pending',
                        attempts INTEGER DEFAULT 0,
                        last_error TEXT,
                        next_attempt_at TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_attempt_at TIMESTAMP,
                        completed_at TIMESTAMP
                    )
                ''')

                existing_columns = {
                    row[1] for row in cursor.execute("PRAGMA table_info(backfill_queue)").fetchall()
                }
                if 'next_attempt_at' not in existing_columns:
                    cursor.execute("ALTER TABLE backfill_queue ADD COLUMN next_attempt_at TIMESTAMP")
                
                # Таблица статистики точности RudySuper
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS rudy_super_stats (
                        fixture_id INTEGER PRIMARY KEY,
                        league_id INTEGER,
                        season INTEGER,
                        match_date TEXT,
                        home_team TEXT,
                        away_team TEXT,
                        predicted_label TEXT,
                        actual_label TEXT,
                        is_correct INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                conn.commit()
                logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")

    @staticmethod
    def _build_next_attempt_timestamp(minutes_from_now: int) -> str:
        return (datetime.now() + timedelta(minutes=minutes_from_now)).isoformat()

    @staticmethod
    def _cooldown_minutes_for_error(error: str) -> int:
        error = (error or '').lower()
        if 'rate_limit' in error or '429' in error:
            return BACKFILL_RATE_LIMIT_COOLDOWN_MINUTES
        if 'incomplete' in error:
            return BACKFILL_INCOMPLETE_COOLDOWN_MINUTES
        return BACKFILL_DEFAULT_COOLDOWN_MINUTES

    @staticmethod
    def _categorize_backfill_error(error: str) -> str:
        error = (error or '').lower()
        if 'rate_limit' in error or '429' in error:
            return 'rate_limit'
        if 'incomplete' in error or 'timeout' in error or 'connection_error' in error:
            return 'incomplete'
        if 'fixture_not_found' in error:
            return 'missing_fixture'
        if not error:
            return 'unknown'
        return 'other'
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """
        Сохранить значение в кэш
        
        Args:
            key: Ключ кэша
            value: Значение (будет сохранено как JSON)
            ttl: Время жизни в секундах (None = без срока)
        
        Returns:
            True если успешно, False иначе
        """
        try:
            json_value = json.dumps(value)
            expires_at = None
            
            if ttl:
                expires_at = (datetime.now() + timedelta(seconds=ttl)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO cache (key, value, expires_at, ttl)
                    VALUES (?, ?, ?, ?)
                ''', (key, json_value, expires_at, ttl))
                conn.commit()
            
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Получить значение из кэша
        
        Args:
            key: Ключ кэша
            default: Значение по умолчанию
        
        Returns:
            Значение из кэша или default
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT value, expires_at FROM cache WHERE key = ?',
                    (key,)
                )
                row = cursor.fetchone()
                
                if not row:
                    return default
                
                value, expires_at = row
                
                # Проверка истечения срока
                if expires_at and datetime.fromisoformat(expires_at) < datetime.now():
                    cursor.execute('DELETE FROM cache WHERE key = ?', (key,))
                    conn.commit()
                    return default
                
                return json.loads(value)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return default
    
    def delete(self, key: str) -> bool:
        """
        Удалить значение из кэша
        
        Args:
            key: Ключ кэша
        
        Returns:
            True если успешно
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM cache WHERE key = ?', (key,))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    def clear(self) -> bool:
        """Очистить весь кэш"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM cache')
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    def cleanup_expired(self) -> int:
        """
        Удалить устаревшие записи
        
        Returns:
            Количество удаленных записей
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM cache WHERE expires_at IS NOT NULL "
                    "AND expires_at < datetime('now')"
                )
                conn.commit()
                return cursor.rowcount
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
            return 0
    
    def save_fixture(self, fixture_data: Dict) -> bool:
        """Сохранить данные матча в БД"""
        try:
            fixture_id = fixture_data['fixture']['id']
            league_id = fixture_data['league']['id']
            season = fixture_data['league']['season']
            date = fixture_data['fixture']['date']
            
            home_team = fixture_data['teams']['home']
            away_team = fixture_data['teams']['away']
            home_goals = fixture_data['goals']['home']
            away_goals = fixture_data['goals']['away']
            status = fixture_data['fixture']['status']['short']
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO fixtures (
                        fixture_id, league_id, season, date,
                        home_team_id, away_team_id, home_team, away_team,
                        status, home_goals, away_goals, data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    fixture_id, league_id, season, date,
                    home_team['id'], away_team['id'],
                    home_team['name'], away_team['name'],
                    status, home_goals, away_goals,
                    json.dumps(fixture_data)
                ))
                conn.commit()
            
            return True
        except Exception as e:
            logger.error(f"Error saving fixture: {e}")
            return False
    
    def get_fixtures_by_league_season(
        self,
        league_id: int,
        season: int
    ) -> List[Dict]:
        """Получить все матчи лиги за сезон из кэша"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM fixtures
                    WHERE league_id = ? AND season = ?
                    ORDER BY date DESC
                ''', (league_id, season))
                
                rows = cursor.fetchall()
                fixtures = []
                for row in rows:
                    row_dict = dict(row)
                    raw_data = row_dict.get('data')
                    if raw_data:
                        fixtures.append(json.loads(raw_data))
                    else:
                        fixtures.append(row_dict)
                return fixtures
        except Exception as e:
            logger.error(f"Error getting fixtures by league/season: {e}")
            return []

    def get_fixture_by_id(self, fixture_id: int) -> Optional[Dict]:
        """Получить fixture по ID из локального кэша."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT data FROM fixtures WHERE fixture_id = ?', (fixture_id,))
                row = cursor.fetchone()
                if not row or not row[0]:
                    return None
                return json.loads(row[0])
        except Exception as e:
            logger.error(f"Error getting fixture by id: {e}")
            return None

    @staticmethod
    def _fixture_needs_statistics(fixture_data: Dict) -> bool:
        statistics = fixture_data.get('statistics', {})
        return not statistics.get('home') or not statistics.get('away')

    @staticmethod
    def _fixture_needs_odds(fixture_data: Dict) -> bool:
        odds = fixture_data.get('odds', {})
        return any(odds.get(key) is None for key in ('1', 'X', '2'))

    def enqueue_fixture_for_backfill(self, fixture_data: Dict) -> bool:
        """Поставить fixture в очередь enrichment, если у него не хватает details."""
        try:
            fixture_id = fixture_data['fixture']['id']
            needs_statistics = self._fixture_needs_statistics(fixture_data)
            needs_odds = self._fixture_needs_odds(fixture_data)
            if not needs_statistics and not needs_odds:
                return False

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO backfill_queue (
                        fixture_id, league_id, season, requested_statistics, requested_odds,
                        status, updated_at, next_attempt_at
                    ) VALUES (?, ?, ?, ?, ?, 'pending', CURRENT_TIMESTAMP, NULL)
                    ON CONFLICT(fixture_id) DO UPDATE SET
                        requested_statistics = excluded.requested_statistics,
                        requested_odds = excluded.requested_odds,
                        status = CASE
                            WHEN backfill_queue.status = 'completed' THEN backfill_queue.status
                            ELSE 'pending'
                        END,
                        next_attempt_at = CASE
                            WHEN backfill_queue.status = 'completed' THEN backfill_queue.next_attempt_at
                            ELSE NULL
                        END,
                        updated_at = CURRENT_TIMESTAMP
                ''', (
                    fixture_id,
                    fixture_data['league']['id'],
                    fixture_data['league']['season'],
                    int(needs_statistics),
                    int(needs_odds),
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error enqueueing fixture for backfill: {e}")
            return False

    def enqueue_missing_fixture_details(self, limit: Optional[int] = None) -> int:
        """Просканировать cached fixtures и добавить в очередь missing details."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                query = 'SELECT data FROM fixtures ORDER BY updated_at DESC'
                if limit is not None:
                    query += f' LIMIT {int(limit)}'
                cursor.execute(query)
                rows = cursor.fetchall()

            queued = 0
            for row in rows:
                if not row[0]:
                    continue
                fixture_data = json.loads(row[0])
                if self.enqueue_fixture_for_backfill(fixture_data):
                    queued += 1
            return queued
        except Exception as e:
            logger.error(f"Error enqueueing missing fixture details: {e}")
            return 0

    def claim_backfill_batch(self, batch_size: int, max_attempts: int = 5) -> List[Dict]:
        """Забрать батч задач из очереди enrichment."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM backfill_queue
                    WHERE status IN ('pending', 'failed')
                      AND attempts < ?
                      AND (next_attempt_at IS NULL OR next_attempt_at <= CURRENT_TIMESTAMP OR next_attempt_at <= ?)
                    ORDER BY
                        CASE status WHEN 'pending' THEN 0 ELSE 1 END,
                        COALESCE(next_attempt_at, last_attempt_at, created_at) ASC
                    LIMIT ?
                ''', (max_attempts, datetime.now().isoformat(), batch_size))
                rows = [dict(row) for row in cursor.fetchall()]
                if not rows:
                    return []

                fixture_ids = [(row['fixture_id'],) for row in rows]
                cursor.executemany('''
                    UPDATE backfill_queue
                    SET status = 'in_progress',
                        attempts = attempts + 1,
                        last_attempt_at = CURRENT_TIMESTAMP,
                        next_attempt_at = NULL,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE fixture_id = ?
                ''', fixture_ids)
                conn.commit()
                return rows
        except Exception as e:
            logger.error(f"Error claiming backfill batch: {e}")
            return []

    def mark_backfill_completed(self, fixture_id: int) -> bool:
        """Пометить задачу enrichment как завершенную."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE backfill_queue
                    SET status = 'completed',
                        last_error = NULL,
                        next_attempt_at = NULL,
                        completed_at = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE fixture_id = ?
                ''', (fixture_id,))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error marking backfill completed: {e}")
            return False

    def mark_backfill_failed(self, fixture_id: int, error: str) -> bool:
        """Пометить задачу enrichment как неудачную с сохранением ошибки."""
        try:
            cooldown_minutes = self._cooldown_minutes_for_error(error)
            next_attempt_at = self._build_next_attempt_timestamp(cooldown_minutes)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE backfill_queue
                    SET status = 'failed',
                        last_error = ?,
                        next_attempt_at = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE fixture_id = ?
                ''', (error[:500], next_attempt_at, fixture_id))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error marking backfill failed: {e}")
            return False

    def get_backfill_queue_status(self) -> Dict[str, int]:
        """Вернуть агрегированную статистику очереди enrichment."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT status, COUNT(*)
                    FROM backfill_queue
                    GROUP BY status
                ''')
                rows = cursor.fetchall()
                status_map = {status: count for status, count in rows}
                return {
                    'pending': int(status_map.get('pending', 0)),
                    'in_progress': int(status_map.get('in_progress', 0)),
                    'completed': int(status_map.get('completed', 0)),
                    'failed': int(status_map.get('failed', 0)),
                    'total': int(sum(status_map.values())),
                }
        except Exception as e:
            logger.error(f"Error getting backfill queue status: {e}")
            return {'pending': 0, 'in_progress': 0, 'completed': 0, 'failed': 0, 'total': 0}

    def get_backfill_failed_breakdown(self) -> Dict[str, int]:
        """Вернуть разбиение failed queue по типам ошибок."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT last_error, COUNT(*) FROM backfill_queue WHERE status = ? GROUP BY last_error', ('failed',))
                rows = cursor.fetchall()

            breakdown = {
                'rate_limit': 0,
                'incomplete': 0,
                'missing_fixture': 0,
                'other': 0,
                'unknown': 0,
            }
            for error, count in rows:
                category = self._categorize_backfill_error(error or '')
                breakdown[category] = breakdown.get(category, 0) + int(count)
            breakdown['total'] = int(sum(value for key, value in breakdown.items() if key != 'total'))
            return breakdown
        except Exception as e:
            logger.error(f"Error getting backfill failed breakdown: {e}")
            return {'rate_limit': 0, 'incomplete': 0, 'missing_fixture': 0, 'other': 0, 'unknown': 0, 'total': 0}

    def retry_failed_backfill(self, category: Optional[str] = None, reset_attempts: bool = True) -> int:
        """Вернуть failed queue обратно в pending по категории ошибки."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT fixture_id, last_error FROM backfill_queue WHERE status = ?', ('failed',))
                rows = cursor.fetchall()

                fixture_ids = []
                for fixture_id, last_error in rows:
                    error_category = self._categorize_backfill_error(last_error or '')
                    if category is None or error_category == category:
                        fixture_ids.append((fixture_id,))

                if not fixture_ids:
                    return 0

                if reset_attempts:
                    cursor.executemany('''
                        UPDATE backfill_queue
                        SET status = 'pending',
                            attempts = 0,
                            last_error = NULL,
                            next_attempt_at = NULL,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE fixture_id = ?
                    ''', fixture_ids)
                else:
                    cursor.executemany('''
                        UPDATE backfill_queue
                        SET status = 'pending',
                            last_error = NULL,
                            next_attempt_at = NULL,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE fixture_id = ?
                    ''', fixture_ids)
                conn.commit()
                return len(fixture_ids)
        except Exception as e:
            logger.error(f"Error retrying failed backfill: {e}")
            return 0

    def reset_old_failed_backfill(self, older_than_hours: int = 24) -> int:
        """Сбросить старые failed queue items в pending, чтобы дать им новую жизнь."""
        try:
            cutoff = (datetime.now() - timedelta(hours=older_than_hours)).isoformat()
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE backfill_queue
                    SET status = 'pending',
                        attempts = 0,
                        last_error = NULL,
                        next_attempt_at = NULL,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE status = 'failed'
                      AND COALESCE(last_attempt_at, updated_at, created_at) <= ?
                ''', (cutoff,))
                conn.commit()
                return int(cursor.rowcount)
        except Exception as e:
            logger.error(f"Error resetting old failed backfill: {e}")
            return 0

    def save_live_snapshot(
        self,
        fixture_data: Dict,
        base_prediction: Dict | None = None,
        in_play_prediction: Dict | None = None,
    ) -> bool:
        """Сохранить live snapshot матча для будущего in-play обучения."""
        try:
            fixture_id = fixture_data['fixture']['id']
            league_id = fixture_data['league']['id']
            season = fixture_data['league']['season']
            home_team_id = fixture_data['teams']['home']['id']
            away_team_id = fixture_data['teams']['away']['id']
            status = fixture_data['fixture']['status']['short']
            elapsed_minute = fixture_data['fixture']['status'].get('elapsed')
            home_goals = fixture_data['goals']['home']
            away_goals = fixture_data['goals']['away']
            statistics = fixture_data.get('statistics', {})

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO live_snapshots (
                        fixture_id, league_id, season, home_team_id, away_team_id,
                        status, elapsed_minute, home_goals, away_goals,
                        home_shots_on_target, away_shots_on_target,
                        home_total_shots, away_total_shots,
                        home_possession, away_possession,
                        home_corners, away_corners,
                        base_prediction, in_play_prediction, data, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    fixture_id,
                    league_id,
                    season,
                    home_team_id,
                    away_team_id,
                    status,
                    elapsed_minute,
                    home_goals,
                    away_goals,
                    statistics.get('home', {}).get('shots_on_target'),
                    statistics.get('away', {}).get('shots_on_target'),
                    statistics.get('home', {}).get('total_shots'),
                    statistics.get('away', {}).get('total_shots'),
                    statistics.get('home', {}).get('possession'),
                    statistics.get('away', {}).get('possession'),
                    statistics.get('home', {}).get('corners'),
                    statistics.get('away', {}).get('corners'),
                    json.dumps(base_prediction or {}),
                    json.dumps(in_play_prediction or {}),
                    json.dumps(fixture_data),
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving live snapshot: {e}")
            return False

    def get_live_snapshot_count(self) -> int:
        """Получить количество сохраненных live snapshots."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM live_snapshots')
                return int(cursor.fetchone()[0])
        except Exception as e:
            logger.error(f"Error counting live snapshots: {e}")
            return 0

    def get_live_snapshot_training_rows(self, only_finished: bool = True) -> List[Dict]:
        """Вернуть snapshots, обогащенные итоговым результатом матча, как заготовку под train set."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                query = '''
                    SELECT
                        s.fixture_id,
                        s.elapsed_minute,
                        s.status,
                        s.home_goals AS snapshot_home_goals,
                        s.away_goals AS snapshot_away_goals,
                        s.home_shots_on_target,
                        s.away_shots_on_target,
                        s.home_total_shots,
                        s.away_total_shots,
                        s.home_possession,
                        s.away_possession,
                        s.home_corners,
                        s.away_corners,
                        s.base_prediction,
                        s.in_play_prediction,
                        s.data,
                        f.home_goals AS final_home_goals,
                        f.away_goals AS final_away_goals,
                        f.status AS final_status
                    FROM live_snapshots s
                    LEFT JOIN fixtures f ON f.fixture_id = s.fixture_id
                '''
                if only_finished:
                    query += " WHERE f.status = 'FT'"
                query += ' ORDER BY s.fixture_id, s.elapsed_minute'
                cursor.execute(query)
                rows = []
                for row in cursor.fetchall():
                    row_dict = dict(row)
                    final_home_goals = row_dict.get('final_home_goals')
                    final_away_goals = row_dict.get('final_away_goals')
                    if final_home_goals is None or final_away_goals is None:
                        row_dict['result'] = None
                    elif final_home_goals > final_away_goals:
                        row_dict['result'] = 0
                    elif final_home_goals == final_away_goals:
                        row_dict['result'] = 1
                    else:
                        row_dict['result'] = 2
                    rows.append(row_dict)
                return rows
        except Exception as e:
            logger.error(f"Error loading live snapshot training rows: {e}")
            return []
    
    def log_api_call(self, endpoint: str, params: Dict = None, status: str = "success"):
        """Логировать API запрос"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO load_history (endpoint, params, status)
                    VALUES (?, ?, ?)
                ''', (endpoint, json.dumps(params or {}), status))
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging API call: {e}")
    
    def get_stats(self) -> Dict:
        """Получить статистику кэша"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('SELECT COUNT(*) FROM cache')
                cache_count = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM fixtures')
                fixtures_count = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM load_history')
                api_calls = cursor.fetchone()[0]
                
                return {
                    'cache_entries': cache_count,
                    'fixtures_cached': fixtures_count,
                    'api_calls_logged': api_calls,
                    'live_snapshots': self.get_live_snapshot_count(),
                    'backfill_queue': self.get_backfill_queue_status(),
                }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}


    # ──────────────────────────────────────────────────────────────────────
    # RudySuper accuracy stats
    # ──────────────────────────────────────────────────────────────────────

    def save_rudy_super_stat(
        self,
        fixture_id: int,
        league_id: int,
        season: int,
        match_date: str,
        home_team: str,
        away_team: str,
        predicted_label: str,
        actual_label: str,
    ) -> bool:
        """Сохранить или обновить запись о предсказании RudySuper для матча."""
        try:
            is_correct = 1 if predicted_label == actual_label else 0
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO rudy_super_stats
                    (fixture_id, league_id, season, match_date, home_team, away_team,
                     predicted_label, actual_label, is_correct)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (fixture_id, league_id, season, match_date, home_team, away_team,
                      predicted_label, actual_label, is_correct))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving rudy_super_stat: {e}")
            return False

    def get_rudy_super_stats_summary(self) -> dict:
        """Получить агрегированную статистику точности RudySuper из БД."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT
                        COUNT(*) AS total,
                        SUM(is_correct) AS correct,
                        SUM(CASE WHEN actual_label = 'Победа хозяев' THEN 1 ELSE 0 END) AS p1_total,
                        SUM(CASE WHEN actual_label = 'Победа хозяев' AND is_correct = 1 THEN 1 ELSE 0 END) AS p1_correct,
                        SUM(CASE WHEN actual_label = 'Ничья' THEN 1 ELSE 0 END) AS draw_total,
                        SUM(CASE WHEN actual_label = 'Ничья' AND is_correct = 1 THEN 1 ELSE 0 END) AS draw_correct,
                        SUM(CASE WHEN actual_label = 'Победа гостей' THEN 1 ELSE 0 END) AS p2_total,
                        SUM(CASE WHEN actual_label = 'Победа гостей' AND is_correct = 1 THEN 1 ELSE 0 END) AS p2_correct,
                        MAX(match_date) AS last_update
                    FROM rudy_super_stats
                ''')
                row = cursor.fetchone()
                if not row or row[0] == 0:
                    return {
                        'total_matches': 0, 'total_correct': 0,
                        'p1_total': 0, 'p1_correct': 0,
                        'draw_total': 0, 'draw_correct': 0,
                        'p2_total': 0, 'p2_correct': 0,
                        'last_update': None,
                    }
                return {
                    'total_matches': int(row[0] or 0),
                    'total_correct': int(row[1] or 0),
                    'p1_total': int(row[2] or 0),
                    'p1_correct': int(row[3] or 0),
                    'draw_total': int(row[4] or 0),
                    'draw_correct': int(row[5] or 0),
                    'p2_total': int(row[6] or 0),
                    'p2_correct': int(row[7] or 0),
                    'last_update': row[8],
                }
        except Exception as e:
            logger.error(f"Error getting rudy_super_stats_summary: {e}")
            return {
                'total_matches': 0, 'total_correct': 0,
                'p1_total': 0, 'p1_correct': 0,
                'draw_total': 0, 'draw_correct': 0,
                'p2_total': 0, 'p2_correct': 0,
                'last_update': None,
            }

    def get_rudy_super_processed_fixture_ids(self) -> set:
        """Вернуть множество fixture_id, уже сохранённых в rudy_super_stats."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT fixture_id FROM rudy_super_stats')
                return {row[0] for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"Error getting processed fixture ids: {e}")
            return set()
