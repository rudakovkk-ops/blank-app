"""
Обработка и очистка данных
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from config.settings import (
    TEST_SIZE,
    VALIDATION_SIZE,
    RANDOM_STATE,
    TRAINING_DATA_SEASONS,
    MIN_MATCHES_HISTORY,
)

logger = logging.getLogger(__name__)


class DataProcessor:
    """Обработка и подготовка данных для моделей"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()

    @staticmethod
    def _safe_float(value) -> float | None:
        """Безопасно привести значение к float."""
        if value is None:
            return None

        if isinstance(value, (int, float, np.integer, np.floating)):
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
    def _get_score_value(cls, fixture: Dict, phase: str, side: str) -> float | None:
        """Получить значение счета из блока score."""
        return cls._safe_float(fixture.get('score', {}).get(phase, {}).get(side))

    @classmethod
    def _get_fixture_stat(cls, fixture: Dict, side: str, key: str) -> float | None:
        """Получить нормализованную match-stat из fixture payload."""
        return cls._safe_float(fixture.get('statistics', {}).get(side, {}).get(key))

    @classmethod
    def _get_fixture_odd(cls, fixture: Dict, key: str) -> float | None:
        """Получить нормализованный odds value из fixture payload."""
        return cls._safe_float(fixture.get('odds', {}).get(key))
    
    def parse_fixtures_to_dataframe(self, fixtures: List[Dict]) -> pd.DataFrame:
        """
        Преобразовать матчи из API в DataFrame
        
        Args:
            fixtures: Список матчей из API
        
        Returns:
            DataFrame с матчами
        """
        records = []
        
        for fixture in fixtures:
            try:
                record = {
                    'fixture_id': fixture['fixture']['id'],
                    'date': fixture['fixture']['date'],
                    'league_id': fixture['league']['id'],
                    'league_name': fixture['league']['name'],
                    'league_country': fixture['league'].get('country'),
                    'season': fixture['league']['season'],
                    'round': fixture['league'].get('round', ''),
                    
                    # Информация команд
                    'home_team_id': fixture['teams']['home']['id'],
                    'home_team': fixture['teams']['home']['name'],
                    'away_team_id': fixture['teams']['away']['id'],
                    'away_team': fixture['teams']['away']['name'],
                    
                    # Результат
                    'home_goals': fixture['goals']['home'],
                    'away_goals': fixture['goals']['away'],
                    'halftime_home_goals': self._get_score_value(fixture, 'halftime', 'home'),
                    'halftime_away_goals': self._get_score_value(fixture, 'halftime', 'away'),
                    'status': fixture['fixture']['status']['short'],
                    
                    # Коэффициенты
                    'home_odds': self._get_fixture_odd(fixture, '1'),
                    'draw_odds': self._get_fixture_odd(fixture, 'X'),
                    'away_odds': self._get_fixture_odd(fixture, '2'),
                    'over_2_5_odds': self._get_fixture_odd(fixture, 'over_2_5'),
                    'under_2_5_odds': self._get_fixture_odd(fixture, 'under_2_5'),

                    # Матчевая статистика
                    'home_total_shots': self._get_fixture_stat(fixture, 'home', 'total_shots'),
                    'away_total_shots': self._get_fixture_stat(fixture, 'away', 'total_shots'),
                    'home_shots_on_target': self._get_fixture_stat(fixture, 'home', 'shots_on_target'),
                    'away_shots_on_target': self._get_fixture_stat(fixture, 'away', 'shots_on_target'),
                    'home_possession': self._get_fixture_stat(fixture, 'home', 'possession'),
                    'away_possession': self._get_fixture_stat(fixture, 'away', 'possession'),
                    'home_corners': self._get_fixture_stat(fixture, 'home', 'corners'),
                    'away_corners': self._get_fixture_stat(fixture, 'away', 'corners'),
                    'home_total_passes': self._get_fixture_stat(fixture, 'home', 'total_passes'),
                    'away_total_passes': self._get_fixture_stat(fixture, 'away', 'total_passes'),
                    'home_accurate_passes': self._get_fixture_stat(fixture, 'home', 'accurate_passes'),
                    'away_accurate_passes': self._get_fixture_stat(fixture, 'away', 'accurate_passes'),
                    'home_fouls': self._get_fixture_stat(fixture, 'home', 'fouls'),
                    'away_fouls': self._get_fixture_stat(fixture, 'away', 'fouls'),
                    'home_tackles': self._get_fixture_stat(fixture, 'home', 'tackles'),
                    'away_tackles': self._get_fixture_stat(fixture, 'away', 'tackles'),
                    'home_interceptions': self._get_fixture_stat(fixture, 'home', 'interceptions'),
                    'away_interceptions': self._get_fixture_stat(fixture, 'away', 'interceptions'),
                }
                
                records.append(record)
            except Exception as e:
                logger.warning(f"Error parsing fixture {fixture.get('fixture', {}).get('id')}: {e}")
                continue
        
        df = pd.DataFrame(records)
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def filter_finished_matches(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Отфильтровать только завершенные матчи
        
        Args:
            df: DataFrame с матчами
        
        Returns:
            DataFrame только с завершенными матчами
        """
        return df[df['status'] == 'FT'].copy()
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Удалить дубликаты матчей"""
        return df.drop_duplicates(subset=['fixture_id']).copy()
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
        """
        Обработать пропущенные значения
        
        Args:
            df: DataFrame
            strategy: 'drop' (удалить) или 'mean' (заполнить средним)
        
        Returns:
            Обработанный DataFrame
        """
        if strategy == 'drop':
            return df.dropna().copy()
        elif strategy == 'mean':
            df = df.copy()
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col].fillna(df[col].mean(), inplace=True)
            return df
        else:
            return df.copy()
    
    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Удалить выбросы (z-score метод)
        
        Args:
            df: DataFrame
            columns: Колонки для проверки
            threshold: Порог z-score
        
        Returns:
            DataFrame без выбросов
        """
        if columns is None:
            columns = ['home_goals', 'away_goals']
        
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores <= threshold]
        
        return df.reset_index(drop=True)
    
    def split_train_val_test(
        self,
        df: pd.DataFrame,
        test_size: float = TEST_SIZE,
        val_size: float = VALIDATION_SIZE
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Разделить данные на обучение, валидацию и тест ВРЕМЕННЫМ ПОРЯДКОМ.
        
        CRITICAL: Data MUST be sorted by date for proper temporal validation.
        Temporal leakage occurs if train indices > test indices by date.
        
        Args:
            df: DataFrame (должен быть отсортирован по дате)
            test_size: Доля тестовых данных
            val_size: Доля валидационных данных
        
        Returns:
            Кортеж (train, val, test)
        """
        # Ensure temporal order: sort by 'date' if present
        if 'date' in df.columns:
            df = df.sort_values('date', ignore_index=False).copy()
            logger.info(
                f"Temporal sort applied: date range {df['date'].min()} to {df['date'].max()}"
            )
        else:
            logger.warning(
                "split_train_val_test: 'date' column not found. "
                "Assuming data is pre-sorted by date. If not, temporal leakage risk!"
            )
        
        n = len(df)
        test_idx = int(n * (1 - test_size - val_size))
        val_idx = int(n * (1 - test_size))
        
        train_df = df.iloc[:test_idx].copy()
        val_df = df.iloc[test_idx:val_idx].copy()
        test_df = df.iloc[val_idx:].copy()
        
        return train_df, val_df, test_df
    
    def create_target_variable(self, df: pd.DataFrame, col_name: str = 'result') -> pd.DataFrame:
        """
        Создать целевую переменную
        0 = Home Win, 1 = Draw, 2 = Away Win
        
        Args:
            df: DataFrame с результатами
            col_name: Имя столбца результата
        
        Returns:
            DataFrame с новым столбцом
        """
        df = df.copy()
        
        df[col_name] = 0
        df.loc[df['home_goals'] == df['away_goals'], col_name] = 1
        df.loc[df['home_goals'] < df['away_goals'], col_name] = 2
        
        return df
    
    def normalize_features(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str] = None,
        fit: bool = False
    ) -> pd.DataFrame:
        """
        Нормализовать числовые признаки
        
        Args:
            df: DataFrame
            numeric_cols: Список колонок для нормализации
            fit: Обучить scaler (True для train, False для val/test)
        
        Returns:
            DataFrame с нормализованными признаками
        """
        df = df.copy()
        
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if fit:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        else:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        
        return df
    
    def get_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        Получить отчет о качестве данных
        
        Args:
            df: DataFrame
        
        Returns:
            Словарь с метриками качества
        """
        report = {
            'total_matches': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicates': df.duplicated().sum(),
            'date_range': f"{df['date'].min()} to {df['date'].max()}" if 'date' in df.columns else 'N/A',
        }
        
        return report
