"""
Инженерия признаков для прогнозирования матчей
"""
import pandas as pd
import numpy as np
import logging
from math import lgamma
from typing import Tuple, Dict, List, Optional
from datetime import datetime, timedelta
from config.settings import (
    H2H_MATCHES_WINDOW,
    MIN_MATCHES_HISTORY,
    ROLLING_WINDOW,
    TEMPORAL_DECAY_ALPHA,
    ELO_BASE_RATING,
    ELO_K_FACTOR,
    ELO_DEFAULT_HOME_ADVANTAGE,
    ELO_SEASON_CARRYOVER,
    ELO_HOME_ADVANTAGE_CARRYOVER,
    LEAGUE_HOME_ADVANTAGE_K_FACTOR,
    MIN_LEAGUE_HOME_ADVANTAGE,
    MAX_LEAGUE_HOME_ADVANTAGE,
)

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Создание и инженерия признаков"""

    RANKING_TIEBREAK_FIELDS = ('points', 'goal_diff', 'goals_for')
    BASE_ELO = ELO_BASE_RATING
    ELO_K_FACTOR = ELO_K_FACTOR
    ELO_HOME_ADVANTAGE = ELO_DEFAULT_HOME_ADVANTAGE
    ELO_SEASON_CARRYOVER = ELO_SEASON_CARRYOVER
    ELO_HOME_ADVANTAGE_CARRYOVER = ELO_HOME_ADVANTAGE_CARRYOVER
    LEAGUE_HOME_ADVANTAGE_K_FACTOR = LEAGUE_HOME_ADVANTAGE_K_FACTOR
    MIN_LEAGUE_HOME_ADVANTAGE = MIN_LEAGUE_HOME_ADVANTAGE
    MAX_LEAGUE_HOME_ADVANTAGE = MAX_LEAGUE_HOME_ADVANTAGE

    @staticmethod
    def _uses_contextual_standings(standings: Dict) -> bool:
        """Определить, что standings заданы по контексту лиги и сезона."""
        if not standings:
            return False

        sample_key = next(iter(standings))
        return isinstance(sample_key, tuple)

    @staticmethod
    def _summarize_team_history(
        history: List[Dict],
        last_n: int = MIN_MATCHES_HISTORY,
        role: Optional[str] = None,
    ) -> Dict[str, float]:
        """Собрать агрегаты по последним матчам команды."""
        if role is None:
            relevant_history = history
        elif role == 'home':
            relevant_history = [match for match in history if match['is_home']]
        else:
            relevant_history = [match for match in history if not match['is_home']]

        recent_matches = relevant_history[-last_n:]

        default_form = 1.0
        if role == 'home':
            default_form = 1.2
        elif role == 'away':
            default_form = 0.8

        if not recent_matches:
            return {
                'form': default_form,
                'weighted_form': default_form,
                'win_rate': 0.33,
                'weighted_win_rate': 0.33,
                'points': default_form * last_n,
                'goals_for': 1.0,
                'goals_against': 1.0,
                'goal_diff': 0.0,
                'weighted_goal_diff': 0.0,
                'halftime_goals_for': 0.5,
                'halftime_goals_against': 0.5,
                'halftime_goal_diff': 0.0,
                'shots_total': 11.0,
                'shots_on_target': 4.0,
                'weighted_shots_on_target': 4.0,
                'possession': 50.0,
                'weighted_possession': 50.0,
                'corners': 4.5,
                'intensity_proxy': 0.08,
                'opponent_rank': 10.0,
                'opponent_elo': FeatureEngineer.BASE_ELO,
                'strength_of_schedule': 1.0,
                'adjusted_points': 1.0,
                'adjusted_goal_diff': 0.0,
                'top_points': 1.0,
                'mid_points': 1.0,
                'bottom_points': 1.0,
                'top_share': 0.0,
                'mid_share': 0.0,
                'bottom_share': 0.0,
            }

        match_count = len(recent_matches)
        weights = np.array([
            TEMPORAL_DECAY_ALPHA ** (match_count - index - 1)
            for index in range(match_count)
        ], dtype=float)
        if float(weights.sum()) == 0.0:
            weights = np.ones(match_count, dtype=float)
        weights = weights / weights.sum()

        def weighted_mean(key: str, default: float = 0.0) -> float:
            values = np.array([float(match.get(key, default)) for match in recent_matches], dtype=float)
            return float(np.dot(values, weights))

        total_points = sum(match['points'] for match in recent_matches)
        goals_for = sum(match['goals_for'] for match in recent_matches)
        goals_against = sum(match['goals_against'] for match in recent_matches)
        wins = sum(match['won'] for match in recent_matches)
        opponent_rank = sum(match['opponent_rank'] for match in recent_matches)
        opponent_elo = sum(match['opponent_elo'] for match in recent_matches)
        strength_of_schedule = sum(match['opponent_strength'] for match in recent_matches)
        adjusted_points = sum(match['adjusted_points'] for match in recent_matches)
        adjusted_goal_diff = sum(match['adjusted_goal_diff'] for match in recent_matches)
        halftime_goals_for = sum(match['halftime_goals_for'] for match in recent_matches)
        halftime_goals_against = sum(match['halftime_goals_against'] for match in recent_matches)
        shots_total = sum(match['shots_total'] for match in recent_matches)
        shots_on_target = sum(match['shots_on_target'] for match in recent_matches)
        possession = sum(match['possession'] for match in recent_matches)
        corners = sum(match['corners'] for match in recent_matches)
        intensity_proxy = sum(match['intensity_proxy'] for match in recent_matches)

        top_matches = [match for match in recent_matches if match['opponent_tier'] == 'top']
        mid_matches = [match for match in recent_matches if match['opponent_tier'] == 'mid']
        bottom_matches = [match for match in recent_matches if match['opponent_tier'] == 'bottom']

        return {
            'form': total_points / match_count,
            'weighted_form': weighted_mean('points'),
            'win_rate': wins / match_count,
            'weighted_win_rate': weighted_mean('won'),
            'points': float(total_points),
            'goals_for': goals_for / match_count,
            'goals_against': goals_against / match_count,
            'goal_diff': (goals_for - goals_against) / match_count,
            'weighted_goal_diff': weighted_mean('goals_for') - weighted_mean('goals_against'),
            'halftime_goals_for': halftime_goals_for / match_count,
            'halftime_goals_against': halftime_goals_against / match_count,
            'halftime_goal_diff': (halftime_goals_for - halftime_goals_against) / match_count,
            'shots_total': shots_total / match_count,
            'shots_on_target': shots_on_target / match_count,
            'weighted_shots_on_target': weighted_mean('shots_on_target', 4.0),
            'possession': possession / match_count,
            'weighted_possession': weighted_mean('possession', 50.0),
            'corners': corners / match_count,
            'intensity_proxy': intensity_proxy / match_count,
            'opponent_rank': opponent_rank / match_count,
            'opponent_elo': opponent_elo / match_count,
            'strength_of_schedule': strength_of_schedule / match_count,
            'adjusted_points': adjusted_points / match_count,
            'adjusted_goal_diff': adjusted_goal_diff / match_count,
            'top_points': (
                sum(match['points'] for match in top_matches) / len(top_matches)
                if top_matches else 1.0
            ),
            'mid_points': (
                sum(match['points'] for match in mid_matches) / len(mid_matches)
                if mid_matches else 1.0
            ),
            'bottom_points': (
                sum(match['points'] for match in bottom_matches) / len(bottom_matches)
                if bottom_matches else 1.0
            ),
            'top_share': len(top_matches) / match_count,
            'mid_share': len(mid_matches) / match_count,
            'bottom_share': len(bottom_matches) / match_count,
        }

    @staticmethod
    def _expected_home_score(
        home_elo: float,
        away_elo: float,
        home_advantage: Optional[float] = None,
    ) -> float:
        """Ожидаемый результат домашней команды по Elo."""
        home_advantage = (
            FeatureEngineer.ELO_HOME_ADVANTAGE if home_advantage is None else home_advantage
        )
        elo_gap = (home_elo + home_advantage) - away_elo
        return 1.0 / (1.0 + 10 ** (-elo_gap / 400.0))

    @staticmethod
    def _decay_toward_baseline(value: float, baseline: float, carryover: float) -> float:
        """Плавно вернуть рейтинг к базовому уровню при старте нового сезона."""
        return baseline + (value - baseline) * carryover

    @staticmethod
    def _rank_to_strength(rank: int, league_size: int) -> float:
        """Нормировать место в таблице в силу соперника от 0 до 1."""
        if league_size <= 0:
            return 0.5

        bounded_rank = min(max(rank, 1), league_size)
        return (league_size - bounded_rank + 1) / league_size

    @staticmethod
    def _compose_opponent_strength(rank: int, league_size: int, elo: float) -> float:
        """Свести rank и Elo соперника в единый показатель силы календаря."""
        rank_strength = FeatureEngineer._rank_to_strength(rank, league_size)
        elo_strength = elo / FeatureEngineer.BASE_ELO
        return 0.5 * rank_strength + 0.5 * elo_strength

    @staticmethod
    def _rank_to_tier(rank: int, league_size: int) -> str:
        """Классифицировать соперника как top/mid/bottom по долям таблицы."""
        if league_size <= 2:
            return 'mid'

        top_cutoff = max(1, int(np.ceil(league_size / 3)))
        bottom_start = max(top_cutoff + 1, league_size - top_cutoff + 1)

        if rank <= top_cutoff:
            return 'top'
        if rank >= bottom_start:
            return 'bottom'
        return 'mid'

    @staticmethod
    def _calculate_streak(history: List[Dict], key: str) -> int:
        """Посчитать длину текущей серии, начиная с последнего матча."""
        streak = 0
        for match in reversed(history):
            if match.get(key):
                streak += 1
            else:
                break
        return streak

    @staticmethod
    def _build_team_history_features(
        df: pd.DataFrame,
        last_n: int = MIN_MATCHES_HISTORY,
    ) -> pd.DataFrame:
        """Построить rolling-признаки для команды по всем прошлым матчам."""
        feature_rows = []
        team_histories: Dict[int, List[Dict]] = {}
        league_team_elos: Dict[int, Dict[int, float]] = {}
        league_home_advantages: Dict[int, float] = {}
        season_stats_by_context: Dict[tuple, Dict[int, Dict[str, float]]] = {}
        initialized_contexts = set()

        for _, row in df.iterrows():
            current_date = row['date']
            league_id = row['league_id']
            home_team_id = row['home_team_id']
            away_team_id = row['away_team_id']
            context_key = (league_id, row['season'])

            if context_key not in initialized_contexts:
                previous_elos = league_team_elos.get(league_id, {})
                league_team_elos[league_id] = {
                    team_id: FeatureEngineer._decay_toward_baseline(
                        elo,
                        FeatureEngineer.BASE_ELO,
                        FeatureEngineer.ELO_SEASON_CARRYOVER,
                    )
                    for team_id, elo in previous_elos.items()
                }
                league_home_advantages[league_id] = FeatureEngineer._decay_toward_baseline(
                    league_home_advantages.get(league_id, FeatureEngineer.ELO_HOME_ADVANTAGE),
                    FeatureEngineer.ELO_HOME_ADVANTAGE,
                    FeatureEngineer.ELO_HOME_ADVANTAGE_CARRYOVER,
                )
                initialized_contexts.add(context_key)

            season_stats = season_stats_by_context.setdefault(context_key, {})
            season_stats.setdefault(home_team_id, {'points': 0, 'goal_diff': 0, 'goals_for': 0})
            season_stats.setdefault(away_team_id, {'points': 0, 'goal_diff': 0, 'goals_for': 0})

            league_size = len(season_stats)
            home_rank = FeatureEngineer._compute_rank_from_stats(season_stats, home_team_id)
            away_rank = FeatureEngineer._compute_rank_from_stats(season_stats, away_team_id)

            team_elos = league_team_elos.setdefault(league_id, {})
            league_home_advantage = league_home_advantages.get(
                league_id,
                FeatureEngineer.ELO_HOME_ADVANTAGE,
            )
            home_elo = team_elos.get(home_team_id, FeatureEngineer.BASE_ELO)
            away_elo = team_elos.get(away_team_id, FeatureEngineer.BASE_ELO)

            home_history = team_histories.get(home_team_id, [])
            away_history = team_histories.get(away_team_id, [])

            home_overall = FeatureEngineer._summarize_team_history(home_history, last_n=last_n)
            home_home = FeatureEngineer._summarize_team_history(home_history, last_n=last_n, role='home')
            home_away = FeatureEngineer._summarize_team_history(home_history, last_n=last_n, role='away')

            away_overall = FeatureEngineer._summarize_team_history(away_history, last_n=last_n)
            away_home = FeatureEngineer._summarize_team_history(away_history, last_n=last_n, role='home')
            away_away = FeatureEngineer._summarize_team_history(away_history, last_n=last_n, role='away')

            home_last_date = home_history[-1]['date'] if home_history else None
            away_last_date = away_history[-1]['date'] if away_history else None

            feature_rows.append({
                'home_form': home_overall['form'],
                'away_form': away_overall['form'],
                'home_weighted_form': home_overall['weighted_form'],
                'away_weighted_form': away_overall['weighted_form'],
                'home_home_form': home_home['form'],
                'home_away_form': home_away['form'],
                'away_home_form': away_home['form'],
                'away_away_form': away_away['form'],
                'home_win_rate': home_overall['win_rate'],
                'away_win_rate': away_overall['win_rate'],
                'home_weighted_win_rate': home_overall['weighted_win_rate'],
                'away_weighted_win_rate': away_overall['weighted_win_rate'],
                'home_home_win_rate': home_home['win_rate'],
                'home_away_win_rate': home_away['win_rate'],
                'away_home_win_rate': away_home['win_rate'],
                'away_away_win_rate': away_away['win_rate'],
                'home_points_last_n': home_overall['points'],
                'away_points_last_n': away_overall['points'],
                'home_goals_for_avg': home_overall['goals_for'],
                'away_goals_for_avg': away_overall['goals_for'],
                'home_goals_against_avg': home_overall['goals_against'],
                'away_goals_against_avg': away_overall['goals_against'],
                'home_goal_diff_avg': home_overall['goal_diff'],
                'away_goal_diff_avg': away_overall['goal_diff'],
                'home_weighted_goal_diff_avg': home_overall['weighted_goal_diff'],
                'away_weighted_goal_diff_avg': away_overall['weighted_goal_diff'],
                'home_halftime_goals_for_avg': home_overall['halftime_goals_for'],
                'away_halftime_goals_for_avg': away_overall['halftime_goals_for'],
                'home_halftime_goals_against_avg': home_overall['halftime_goals_against'],
                'away_halftime_goals_against_avg': away_overall['halftime_goals_against'],
                'home_halftime_goal_diff_avg': home_overall['halftime_goal_diff'],
                'away_halftime_goal_diff_avg': away_overall['halftime_goal_diff'],
                'home_shots_total_avg': home_overall['shots_total'],
                'away_shots_total_avg': away_overall['shots_total'],
                'home_shots_on_target_avg': home_overall['shots_on_target'],
                'away_shots_on_target_avg': away_overall['shots_on_target'],
                'home_weighted_shots_on_target_avg': home_overall['weighted_shots_on_target'],
                'away_weighted_shots_on_target_avg': away_overall['weighted_shots_on_target'],
                'home_possession_avg': home_overall['possession'],
                'away_possession_avg': away_overall['possession'],
                'home_weighted_possession_avg': home_overall['weighted_possession'],
                'away_weighted_possession_avg': away_overall['weighted_possession'],
                'home_corners_avg': home_overall['corners'],
                'away_corners_avg': away_overall['corners'],
                'home_intensity_proxy_avg': home_overall['intensity_proxy'],
                'away_intensity_proxy_avg': away_overall['intensity_proxy'],
                'home_opponent_rank_avg': home_overall['opponent_rank'],
                'away_opponent_rank_avg': away_overall['opponent_rank'],
                'home_opponent_elo_avg': home_overall['opponent_elo'],
                'away_opponent_elo_avg': away_overall['opponent_elo'],
                'home_strength_of_schedule': home_overall['strength_of_schedule'],
                'away_strength_of_schedule': away_overall['strength_of_schedule'],
                'home_adjusted_points_avg': home_overall['adjusted_points'],
                'away_adjusted_points_avg': away_overall['adjusted_points'],
                'home_adjusted_goal_diff_avg': home_overall['adjusted_goal_diff'],
                'away_adjusted_goal_diff_avg': away_overall['adjusted_goal_diff'],
                'home_top_tier_points_avg': home_overall['top_points'],
                'away_top_tier_points_avg': away_overall['top_points'],
                'home_mid_tier_points_avg': home_overall['mid_points'],
                'away_mid_tier_points_avg': away_overall['mid_points'],
                'home_bottom_tier_points_avg': home_overall['bottom_points'],
                'away_bottom_tier_points_avg': away_overall['bottom_points'],
                'home_top_tier_share': home_overall['top_share'],
                'away_top_tier_share': away_overall['top_share'],
                'home_mid_tier_share': home_overall['mid_share'],
                'away_mid_tier_share': away_overall['mid_share'],
                'home_bottom_tier_share': home_overall['bottom_share'],
                'away_bottom_tier_share': away_overall['bottom_share'],
                'home_win_streak': FeatureEngineer._calculate_streak(home_history, 'won'),
                'away_win_streak': FeatureEngineer._calculate_streak(away_history, 'won'),
                'home_unbeaten_streak': FeatureEngineer._calculate_streak(home_history, 'unbeaten'),
                'away_unbeaten_streak': FeatureEngineer._calculate_streak(away_history, 'unbeaten'),
                'home_rest_days': (current_date - home_last_date).days if home_last_date is not None else 7,
                'away_rest_days': (current_date - away_last_date).days if away_last_date is not None else 7,
                'home_elo': home_elo,
                'away_elo': away_elo,
                'league_home_advantage': league_home_advantage,
                'ranking_difference': away_rank - home_rank,
            })

            home_goals = row.get('home_goals')
            away_goals = row.get('away_goals')
            if pd.isna(home_goals) or pd.isna(away_goals):
                continue

            home_points = 3 if home_goals > away_goals else 1 if home_goals == away_goals else 0
            away_points = 3 if away_goals > home_goals else 1 if home_goals == away_goals else 0
            home_goal_diff = float(home_goals) - float(away_goals)
            away_goal_diff = -home_goal_diff

            home_opponent_tier = FeatureEngineer._rank_to_tier(away_rank, league_size)
            away_opponent_tier = FeatureEngineer._rank_to_tier(home_rank, league_size)

            home_opponent_strength = FeatureEngineer._compose_opponent_strength(
                away_rank,
                league_size,
                away_elo,
            )
            away_opponent_strength = FeatureEngineer._compose_opponent_strength(
                home_rank,
                league_size,
                home_elo,
            )

            home_adjusted_points = home_points * home_opponent_strength
            away_adjusted_points = away_points * away_opponent_strength
            home_adjusted_goal_diff = home_goal_diff * home_opponent_strength
            away_adjusted_goal_diff = away_goal_diff * away_opponent_strength

            home_halftime_goals = float(row.get('halftime_home_goals')) if not pd.isna(row.get('halftime_home_goals')) else 0.5
            away_halftime_goals = float(row.get('halftime_away_goals')) if not pd.isna(row.get('halftime_away_goals')) else 0.5
            home_total_shots = float(row.get('home_total_shots')) if not pd.isna(row.get('home_total_shots')) else 11.0
            away_total_shots = float(row.get('away_total_shots')) if not pd.isna(row.get('away_total_shots')) else 11.0
            home_shots_on_target = float(row.get('home_shots_on_target')) if not pd.isna(row.get('home_shots_on_target')) else 4.0
            away_shots_on_target = float(row.get('away_shots_on_target')) if not pd.isna(row.get('away_shots_on_target')) else 4.0
            home_possession = float(row.get('home_possession')) if not pd.isna(row.get('home_possession')) else 50.0
            away_possession = float(row.get('away_possession')) if not pd.isna(row.get('away_possession')) else 50.0
            home_corners = float(row.get('home_corners')) if not pd.isna(row.get('home_corners')) else 4.5
            away_corners = float(row.get('away_corners')) if not pd.isna(row.get('away_corners')) else 4.5
            home_accurate_passes = float(row.get('home_accurate_passes')) if not pd.isna(row.get('home_accurate_passes')) else 350.0
            away_accurate_passes = float(row.get('away_accurate_passes')) if not pd.isna(row.get('away_accurate_passes')) else 350.0
            home_tackles = float(row.get('home_tackles')) if not pd.isna(row.get('home_tackles')) else 16.0
            away_tackles = float(row.get('away_tackles')) if not pd.isna(row.get('away_tackles')) else 16.0
            home_interceptions = float(row.get('home_interceptions')) if not pd.isna(row.get('home_interceptions')) else 8.0
            away_interceptions = float(row.get('away_interceptions')) if not pd.isna(row.get('away_interceptions')) else 8.0
            home_fouls = float(row.get('home_fouls')) if not pd.isna(row.get('home_fouls')) else 12.0
            away_fouls = float(row.get('away_fouls')) if not pd.isna(row.get('away_fouls')) else 12.0
            home_intensity_proxy = (home_tackles + home_interceptions + home_fouls) / max(away_accurate_passes, 1.0)
            away_intensity_proxy = (away_tackles + away_interceptions + away_fouls) / max(home_accurate_passes, 1.0)

            team_histories.setdefault(home_team_id, []).append({
                'date': current_date,
                'is_home': True,
                'goals_for': float(home_goals),
                'goals_against': float(away_goals),
                'halftime_goals_for': home_halftime_goals,
                'halftime_goals_against': away_halftime_goals,
                'shots_total': home_total_shots,
                'shots_on_target': home_shots_on_target,
                'possession': home_possession,
                'corners': home_corners,
                'intensity_proxy': home_intensity_proxy,
                'points': home_points,
                'won': home_points == 3,
                'unbeaten': home_points >= 1,
                'opponent_rank': away_rank,
                'opponent_tier': home_opponent_tier,
                'opponent_elo': away_elo,
                'opponent_strength': home_opponent_strength,
                'adjusted_points': home_adjusted_points,
                'adjusted_goal_diff': home_adjusted_goal_diff,
            })
            team_histories.setdefault(away_team_id, []).append({
                'date': current_date,
                'is_home': False,
                'goals_for': float(away_goals),
                'goals_against': float(home_goals),
                'halftime_goals_for': away_halftime_goals,
                'halftime_goals_against': home_halftime_goals,
                'shots_total': away_total_shots,
                'shots_on_target': away_shots_on_target,
                'possession': away_possession,
                'corners': away_corners,
                'intensity_proxy': away_intensity_proxy,
                'points': away_points,
                'won': away_points == 3,
                'unbeaten': away_points >= 1,
                'opponent_rank': home_rank,
                'opponent_tier': away_opponent_tier,
                'opponent_elo': home_elo,
                'opponent_strength': away_opponent_strength,
                'adjusted_points': away_adjusted_points,
                'adjusted_goal_diff': away_adjusted_goal_diff,
            })

            season_stats[home_team_id]['points'] += home_points
            season_stats[home_team_id]['goal_diff'] += home_goal_diff
            season_stats[home_team_id]['goals_for'] += float(home_goals)

            season_stats[away_team_id]['points'] += away_points
            season_stats[away_team_id]['goal_diff'] += away_goal_diff
            season_stats[away_team_id]['goals_for'] += float(away_goals)

            expected_home = FeatureEngineer._expected_home_score(
                home_elo,
                away_elo,
                league_home_advantage,
            )
            expected_away = 1.0 - expected_home
            score_home = 1.0 if home_points == 3 else 0.5 if home_points == 1 else 0.0
            score_away = 1.0 - score_home

            team_elos[home_team_id] = home_elo + FeatureEngineer.ELO_K_FACTOR * (score_home - expected_home)
            team_elos[away_team_id] = away_elo + FeatureEngineer.ELO_K_FACTOR * (score_away - expected_away)
            league_home_advantages[league_id] = float(np.clip(
                league_home_advantage
                + FeatureEngineer.LEAGUE_HOME_ADVANTAGE_K_FACTOR * (score_home - expected_home),
                FeatureEngineer.MIN_LEAGUE_HOME_ADVANTAGE,
                FeatureEngineer.MAX_LEAGUE_HOME_ADVANTAGE,
            ))

        return pd.DataFrame(feature_rows, index=df.index)
    
    @staticmethod
    def calculate_team_form(
        df: pd.DataFrame,
        team_col: str,
        goals_col: str,
        last_n: int = MIN_MATCHES_HISTORY
    ) -> pd.Series:
        """
        Рассчитать форму команды (средние голы за последние N матчей)
        
        Args:
            df: DataFrame
            team_col: Столбец с ID команды
            goals_col: Столбец с голами
            last_n: Количество последних матчей
        
        Returns:
            Series с форму команды
        """
        form = []
        
        for idx in range(len(df)):
            team_id = df.iloc[idx][team_col]
            
            # Получить целевую переменную, если её еще нет
            is_home = team_col == 'home_team_id'
            other_team_col = 'home_team_id' if team_col == 'away_team_id' else 'away_team_id'
            
            # Фильтруем матчи этой команды до текущего матча
            if is_home:
                team_matches = df.iloc[:idx][df.iloc[:idx]['home_team_id'] == team_id]
            else:
                team_matches = df.iloc[:idx][df.iloc[:idx]['away_team_id'] == team_id]
            
            if len(team_matches) >= last_n:
                avg_goals = team_matches.iloc[-last_n:][goals_col].mean()
            elif len(team_matches) > 0:
                avg_goals = team_matches[goals_col].mean()
            else:
                avg_goals = 0.5  # Default значение
            
            form.append(avg_goals)
        
        return pd.Series(form)
    
    @staticmethod
    def calculate_win_rate(
        df: pd.DataFrame,
        team_col: str,
        result_col: str,
        last_n: int = MIN_MATCHES_HISTORY
    ) -> pd.Series:
        """
        Рассчитать процент побед команды
        
        Args:
            df: DataFrame с результатами
            team_col: Столбец с ID команды
            result_col: Столбец с результатом (0=Win, 1=Draw, 2=Loss)
            last_n: Количество последних матчей
        
        Returns:
            Series с win rate
        """
        win_rates = []
        
        for idx in range(len(df)):
            team_id = df.iloc[idx][team_col]
            is_home = team_col == 'home_team_id'
            
            # Фильтруем матчи
            if is_home:
                team_matches = df.iloc[:idx][df.iloc[:idx]['home_team_id'] == team_id]
                wins = (team_matches[result_col] == 0).sum()
            else:
                team_matches = df.iloc[:idx][df.iloc[:idx]['away_team_id'] == team_id]
                wins = (team_matches[result_col] == 2).sum()
            
            if len(team_matches) >= last_n:
                win_rate = wins / last_n
            elif len(team_matches) > 0:
                win_rate = wins / len(team_matches)
            else:
                win_rate = 0.33  # Default
            
            win_rates.append(win_rate)
        
        return pd.Series(win_rates)
    
    @staticmethod
    def calculate_rest_days(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Рассчитать дни отдыха между матчами
        
        Args:
            df: DataFrame с датами матчей
        
        Returns:
            Кортеж (home_rest_days, away_rest_days)
        """
        home_rest = []
        away_rest = []
        
        for idx in range(len(df)):
            current_date = df.iloc[idx]['date']
            home_team_id = df.iloc[idx]['home_team_id']
            away_team_id = df.iloc[idx]['away_team_id']
            
            # Последний матч домашней команды
            prev_home = df.iloc[:idx][df.iloc[:idx]['home_team_id'] == home_team_id]
            if len(prev_home) > 0:
                last_home_date = prev_home.iloc[-1]['date']
                home_rest.append((current_date - last_home_date).days)
            else:
                home_rest.append(7)  # Default
            
            # Последний матч выездной команды
            prev_away = df.iloc[:idx][df.iloc[:idx]['away_team_id'] == away_team_id]
            if len(prev_away) > 0:
                last_away_date = prev_away.iloc[-1]['date']
                away_rest.append((current_date - last_away_date).days)
            else:
                away_rest.append(7)  # Default
        
        return pd.Series(home_rest), pd.Series(away_rest)
    
    @staticmethod
    def calculate_ranking_difference(
        df: pd.DataFrame,
        standings_dict: Dict[int, Dict]
    ) -> pd.Series:
        """
        Рассчитать разницу в рейтингах команд
        
        Args:
            df: DataFrame с матчами
            standings_dict: Словарь {team_id: rank}
        
        Returns:
            Series с разницей рейтингов
        """
        rank_gaps = []
        
        for idx, row in df.iterrows():
            home_rank = standings_dict.get(row['home_team_id'], {}).get('rank', 10)
            away_rank = standings_dict.get(row['away_team_id'], {}).get('rank', 10)
            
            # Отрицательный gap = домашняя команда выше
            gap = away_rank - home_rank
            rank_gaps.append(gap)
        
        return pd.Series(rank_gaps)

    @staticmethod
    def calculate_contextual_ranking_difference(
        df: pd.DataFrame,
        standings_lookup: Dict[tuple, Dict[int, Dict]],
    ) -> pd.Series:
        """Рассчитать разницу рейтингов с учетом лиги и сезона матча."""
        rank_gaps = []

        for _, row in df.iterrows():
            standings_dict = standings_lookup.get((row['league_id'], row['season']), {})
            home_rank = standings_dict.get(row['home_team_id'], {}).get('rank', 10)
            away_rank = standings_dict.get(row['away_team_id'], {}).get('rank', 10)
            rank_gaps.append(away_rank - home_rank)

        return pd.Series(rank_gaps)

    @staticmethod
    def _compute_rank_from_stats(
        season_stats: Dict[int, Dict[str, float]],
        team_id: int,
    ) -> int:
        """Определить место команды по накопленной статистике на текущий момент."""
        team_stats = season_stats.get(team_id, {'points': 0, 'goal_diff': 0, 'goals_for': 0})
        better_teams = 0

        for other_team_id, other_stats in season_stats.items():
            if other_team_id == team_id:
                continue

            other_tuple = tuple(other_stats[field] for field in FeatureEngineer.RANKING_TIEBREAK_FIELDS)
            team_tuple = tuple(team_stats[field] for field in FeatureEngineer.RANKING_TIEBREAK_FIELDS)
            if other_tuple > team_tuple:
                better_teams += 1

        return better_teams + 1

    @staticmethod
    def calculate_time_aware_ranking_difference(df: pd.DataFrame) -> pd.Series:
        """Рассчитать ranking_difference по таблице на момент матча без look-ahead."""
        rank_gaps = []
        stats_by_context: Dict[tuple, Dict[int, Dict[str, float]]] = {}

        for _, row in df.iterrows():
            context_key = (row['league_id'], row['season'])
            season_stats = stats_by_context.setdefault(context_key, {})

            home_team_id = row['home_team_id']
            away_team_id = row['away_team_id']

            season_stats.setdefault(home_team_id, {'points': 0, 'goal_diff': 0, 'goals_for': 0})
            season_stats.setdefault(away_team_id, {'points': 0, 'goal_diff': 0, 'goals_for': 0})

            home_rank = FeatureEngineer._compute_rank_from_stats(season_stats, home_team_id)
            away_rank = FeatureEngineer._compute_rank_from_stats(season_stats, away_team_id)
            rank_gaps.append(away_rank - home_rank)

            home_goals = row.get('home_goals')
            away_goals = row.get('away_goals')
            if pd.isna(home_goals) or pd.isna(away_goals):
                continue

            home_points = 3 if home_goals > away_goals else 1 if home_goals == away_goals else 0
            away_points = 3 if away_goals > home_goals else 1 if home_goals == away_goals else 0

            season_stats[home_team_id]['points'] += home_points
            season_stats[home_team_id]['goal_diff'] += float(home_goals) - float(away_goals)
            season_stats[home_team_id]['goals_for'] += float(home_goals)

            season_stats[away_team_id]['points'] += away_points
            season_stats[away_team_id]['goal_diff'] += float(away_goals) - float(home_goals)
            season_stats[away_team_id]['goals_for'] += float(away_goals)

        return pd.Series(rank_gaps)
    
    @staticmethod
    def calculate_head_to_head_stats(
        df: pd.DataFrame,
        h2h_dict: Dict[str, Dict]
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Рассчитать статистику личных встреч
        
        Args:
            df: DataFrame с матчами
            h2h_dict: Словарь {f"{team1_id}-{team2_id}": stats}
        
        Returns:
            Кортеж (home_h2h_wins, away_h2h_wins)
        """
        home_h2h = []
        away_h2h = []
        
        for idx, row in df.iterrows():
            h2h_key1 = f"{row['home_team_id']}-{row['away_team_id']}"
            h2h_key2 = f"{row['away_team_id']}-{row['home_team_id']}"
            
            stats = h2h_dict.get(h2h_key1) or h2h_dict.get(h2h_key2, {'home_wins': 0, 'away_wins': 0})
            
            home_h2h.append(stats.get('home_wins', 0))
            away_h2h.append(stats.get('away_wins', 0))
        
        return pd.Series(home_h2h), pd.Series(away_h2h)

    @staticmethod
    def _build_head_to_head_features(
        df: pd.DataFrame,
        last_n: int = H2H_MATCHES_WINDOW,
    ) -> pd.DataFrame:
        """Построить признаки личных встреч по последним очным матчам без look-ahead."""
        feature_rows = []
        pair_histories: Dict[tuple[int, int], List[Dict[str, float]]] = {}

        for _, row in df.iterrows():
            home_team_id = int(row['home_team_id'])
            away_team_id = int(row['away_team_id'])
            pair_key = tuple(sorted((home_team_id, away_team_id)))
            history = pair_histories.get(pair_key, [])[-last_n:]

            if history:
                current_home_points = []
                current_away_points = []
                home_win_flags = []
                away_win_flags = []
                draw_flags = []
                goal_diffs = []
                total_goals = []
                home_goals_series = []
                away_goals_series = []

                for match in history:
                    if match['home_team_id'] == home_team_id:
                        current_home_goals = match['home_goals']
                        current_away_goals = match['away_goals']
                    else:
                        current_home_goals = match['away_goals']
                        current_away_goals = match['home_goals']

                    if current_home_goals > current_away_goals:
                        home_points = 3.0
                        away_points = 0.0
                    elif current_home_goals < current_away_goals:
                        home_points = 0.0
                        away_points = 3.0
                    else:
                        home_points = 1.0
                        away_points = 1.0

                    current_home_points.append(home_points)
                    current_away_points.append(away_points)
                    home_win_flags.append(1.0 if home_points == 3.0 else 0.0)
                    away_win_flags.append(1.0 if away_points == 3.0 else 0.0)
                    draw_flags.append(1.0 if home_points == 1.0 else 0.0)
                    goal_diffs.append(current_home_goals - current_away_goals)
                    total_goals.append(current_home_goals + current_away_goals)
                    home_goals_series.append(current_home_goals)
                    away_goals_series.append(current_away_goals)

                feature_rows.append({
                    'h2h_matches_played': float(len(history)),
                    'h2h_home_points_avg': float(np.mean(current_home_points)),
                    'h2h_away_points_avg': float(np.mean(current_away_points)),
                    'h2h_home_win_rate': float(np.mean(home_win_flags)),
                    'h2h_away_win_rate': float(np.mean(away_win_flags)),
                    'h2h_draw_rate': float(np.mean(draw_flags)),
                    'h2h_goal_diff_avg': float(np.mean(goal_diffs)),
                    'h2h_total_goals_avg': float(np.mean(total_goals)),
                    'h2h_home_goals_avg': float(np.mean(home_goals_series)),
                    'h2h_away_goals_avg': float(np.mean(away_goals_series)),
                })
            else:
                feature_rows.append({
                    'h2h_matches_played': 0.0,
                    'h2h_home_points_avg': 1.0,
                    'h2h_away_points_avg': 1.0,
                    'h2h_home_win_rate': 0.33,
                    'h2h_away_win_rate': 0.33,
                    'h2h_draw_rate': 0.34,
                    'h2h_goal_diff_avg': 0.0,
                    'h2h_total_goals_avg': 2.5,
                    'h2h_home_goals_avg': 1.25,
                    'h2h_away_goals_avg': 1.25,
                })

            home_goals = row.get('home_goals')
            away_goals = row.get('away_goals')
            if pd.isna(home_goals) or pd.isna(away_goals):
                continue

            pair_histories.setdefault(pair_key, []).append({
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'home_goals': float(home_goals),
                'away_goals': float(away_goals),
            })

        return pd.DataFrame(feature_rows, index=df.index)
    
    @staticmethod
    def create_feature_matrix(
        df: pd.DataFrame,
        standings: Dict = None,
        time_aware_standings: bool = False,
    ) -> pd.DataFrame:
        """
        Создать матрицу признаков для модели
        
        Args:
            df: DataFrame с матчами
            standings: Словарь турнирной таблицы
        
        Returns:
            DataFrame с признаками
        """
        base_columns = ['fixture_id', 'date', 'home_team_id', 'away_team_id']
        features_df = df[base_columns].copy()
        team_features = FeatureEngineer._build_team_history_features(df)
        features_df = pd.concat([features_df, team_features], axis=1)
        h2h_features = FeatureEngineer._build_head_to_head_features(df)
        features_df = pd.concat([features_df, h2h_features], axis=1)

        if 'season' in df.columns:
            season_series = pd.to_numeric(df['season'], errors='coerce')
            baseline_season = float(season_series.min()) if not season_series.dropna().empty else 0.0
            current_season = float(season_series.max()) if not season_series.dropna().empty else 0.0
            features_df['season_year'] = season_series.fillna(baseline_season)
            features_df['season_offset'] = season_series.fillna(baseline_season) - baseline_season
            features_df['is_current_season'] = (
                season_series.fillna(current_season) == current_season
            ).astype(int)

        if 'league_id' in df.columns:
            league_dummies = pd.get_dummies(
                df['league_id'].astype('Int64').astype(str),
                prefix='league',
                dtype=float,
            )
            features_df = pd.concat([features_df, league_dummies], axis=1)

        if 'league_country' in df.columns:
            country_labels = df['league_country'].fillna('Unknown').astype(str).str.strip()
            country_dummies = pd.get_dummies(
                country_labels,
                prefix='country',
                dtype=float,
            )
            features_df = pd.concat([features_df, country_dummies], axis=1)
        
        # Разница в рейтингах
        if time_aware_standings:
            if 'ranking_difference' not in features_df.columns:
                features_df['ranking_difference'] = FeatureEngineer.calculate_time_aware_ranking_difference(df)
        elif standings:
            if FeatureEngineer._uses_contextual_standings(standings):
                features_df['ranking_difference'] = FeatureEngineer.calculate_contextual_ranking_difference(
                    df, standings
                )
            else:
                features_df['ranking_difference'] = FeatureEngineer.calculate_ranking_difference(
                    df, standings
                )
        else:
            features_df['ranking_difference'] = 0
        
        # Коэффициенты и missingness-индикаторы.
        odds_columns = ['home_odds', 'draw_odds', 'away_odds', 'over_2_5_odds', 'under_2_5_odds']
        odds_frame = df.reindex(columns=odds_columns)
        features_df['odds_data_available'] = odds_frame.notna().all(axis=1).astype(float)
        features_df['odds_partial_available'] = odds_frame.notna().any(axis=1).astype(float)
        features_df['home_odds_missing'] = odds_frame['home_odds'].isna().astype(float)
        features_df['draw_odds_missing'] = odds_frame['draw_odds'].isna().astype(float)
        features_df['away_odds_missing'] = odds_frame['away_odds'].isna().astype(float)
        features_df['over_2_5_odds_missing'] = odds_frame['over_2_5_odds'].isna().astype(float)
        features_df['under_2_5_odds_missing'] = odds_frame['under_2_5_odds'].isna().astype(float)
        features_df['odds_missing_count'] = odds_frame.isna().sum(axis=1).astype(float)
        features_df['odds_missing_rate'] = features_df['odds_missing_count'] / float(len(odds_columns))
        features_df['home_odds'] = odds_frame['home_odds']
        features_df['draw_odds'] = odds_frame['draw_odds']
        features_df['away_odds'] = odds_frame['away_odds']
        features_df['over_2_5_odds'] = odds_frame['over_2_5_odds']
        features_df['under_2_5_odds'] = odds_frame['under_2_5_odds']

        statistics_columns = [
            'home_total_shots',
            'away_total_shots',
            'home_shots_on_target',
            'away_shots_on_target',
            'home_possession',
            'away_possession',
            'home_corners',
            'away_corners',
        ]
        available_statistics_columns = [column for column in statistics_columns if column in df.columns]
        if available_statistics_columns:
            features_df['statistics_data_available'] = (
                df[available_statistics_columns].notna().mean(axis=1)
            ).astype(float)
            features_df['statistics_missing_count'] = (
                1.0 - features_df['statistics_data_available']
            ) * float(len(available_statistics_columns))
            features_df['statistics_missing_rate'] = (
                features_df['statistics_missing_count'] / float(len(available_statistics_columns))
            )
        else:
            features_df['statistics_data_available'] = 0.0
            features_df['statistics_missing_count'] = 0.0
            features_df['statistics_missing_rate'] = 0.0
        
        # Целевая переменная
        if 'result' in df.columns:
            features_df['result'] = df['result']
        
        return features_df

    @staticmethod
    def _poisson_distribution(rate: float, max_goals: int = 8) -> np.ndarray:
        clipped_rate = float(np.clip(rate, 0.05, 5.0))
        support = np.arange(max_goals + 1, dtype=float)
        log_pmf = -clipped_rate + support * np.log(clipped_rate) - np.array([
            lgamma(goal + 1.0) for goal in support
        ])
        distribution = np.exp(log_pmf)
        tail_mass = max(0.0, 1.0 - float(distribution.sum()))
        distribution[-1] += tail_mass
        return distribution / np.clip(distribution.sum(), 1e-12, None)

    @classmethod
    def _poisson_outcome_probabilities(
        cls,
        home_rate: float,
        away_rate: float,
        max_goals: int = 8,
    ) -> tuple[float, float, float]:
        home_distribution = cls._poisson_distribution(home_rate, max_goals=max_goals)
        away_distribution = cls._poisson_distribution(away_rate, max_goals=max_goals)
        score_matrix = np.outer(home_distribution, away_distribution)
        home_win_probability = float(np.tril(score_matrix, -1).sum())
        draw_probability = float(np.trace(score_matrix))
        away_win_probability = float(np.triu(score_matrix, 1).sum())
        total = max(home_win_probability + draw_probability + away_win_probability, 1e-12)
        return (
            home_win_probability / total,
            draw_probability / total,
            away_win_probability / total,
        )
    
    @staticmethod
    def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        """Добавить взаимодействия между признаками"""
        df = df.copy()
        
        if 'home_form' in df.columns and 'away_form' in df.columns:
            df['form_difference'] = df['home_form'] - df['away_form']
            df['form_ratio'] = (df['home_form'] + 0.1) / (df['away_form'] + 0.1)

        if 'home_weighted_form' in df.columns and 'away_weighted_form' in df.columns:
            df['weighted_form_difference'] = df['home_weighted_form'] - df['away_weighted_form']

        if 'home_weighted_goal_diff_avg' in df.columns and 'away_weighted_goal_diff_avg' in df.columns:
            df['weighted_goal_diff_difference'] = (
                df['home_weighted_goal_diff_avg'] - df['away_weighted_goal_diff_avg']
            )
        
        if 'home_win_rate' in df.columns and 'away_win_rate' in df.columns:
            df['win_rate_difference'] = df['home_win_rate'] - df['away_win_rate']

        if 'h2h_home_points_avg' in df.columns and 'h2h_away_points_avg' in df.columns:
            df['h2h_points_difference'] = df['h2h_home_points_avg'] - df['h2h_away_points_avg']

        if 'h2h_home_win_rate' in df.columns and 'h2h_away_win_rate' in df.columns:
            df['h2h_win_rate_difference'] = df['h2h_home_win_rate'] - df['h2h_away_win_rate']

        if 'home_points_last_n' in df.columns and 'away_points_last_n' in df.columns:
            df['points_difference'] = df['home_points_last_n'] - df['away_points_last_n']

        if 'home_goals_for_avg' in df.columns and 'away_goals_for_avg' in df.columns:
            df['goals_for_difference'] = df['home_goals_for_avg'] - df['away_goals_for_avg']

        if 'home_goals_against_avg' in df.columns and 'away_goals_against_avg' in df.columns:
            df['goals_against_difference'] = (
                df['home_goals_against_avg'] - df['away_goals_against_avg']
            )

        if 'home_halftime_goal_diff_avg' in df.columns and 'away_halftime_goal_diff_avg' in df.columns:
            df['halftime_goal_diff_difference'] = (
                df['home_halftime_goal_diff_avg'] - df['away_halftime_goal_diff_avg']
            )

        if 'home_shots_total_avg' in df.columns and 'away_shots_total_avg' in df.columns:
            df['shots_total_difference'] = df['home_shots_total_avg'] - df['away_shots_total_avg']

        if 'home_shots_on_target_avg' in df.columns and 'away_shots_on_target_avg' in df.columns:
            df['shots_on_target_difference'] = (
                df['home_shots_on_target_avg'] - df['away_shots_on_target_avg']
            )

        if 'home_possession_avg' in df.columns and 'away_possession_avg' in df.columns:
            df['possession_difference'] = df['home_possession_avg'] - df['away_possession_avg']

        if 'home_corners_avg' in df.columns and 'away_corners_avg' in df.columns:
            df['corners_difference'] = df['home_corners_avg'] - df['away_corners_avg']

        if 'home_intensity_proxy_avg' in df.columns and 'away_intensity_proxy_avg' in df.columns:
            df['intensity_proxy_difference'] = (
                df['home_intensity_proxy_avg'] - df['away_intensity_proxy_avg']
            )

        if 'home_goal_diff_avg' in df.columns and 'away_goal_diff_avg' in df.columns:
            df['goal_diff_difference'] = df['home_goal_diff_avg'] - df['away_goal_diff_avg']

        if 'home_opponent_rank_avg' in df.columns and 'away_opponent_rank_avg' in df.columns:
            df['opponent_rank_difference'] = df['away_opponent_rank_avg'] - df['home_opponent_rank_avg']

        if 'home_opponent_elo_avg' in df.columns and 'away_opponent_elo_avg' in df.columns:
            df['opponent_elo_difference'] = df['home_opponent_elo_avg'] - df['away_opponent_elo_avg']

        if 'home_strength_of_schedule' in df.columns and 'away_strength_of_schedule' in df.columns:
            df['strength_of_schedule_difference'] = (
                df['home_strength_of_schedule'] - df['away_strength_of_schedule']
            )

        if 'home_adjusted_points_avg' in df.columns and 'away_adjusted_points_avg' in df.columns:
            df['adjusted_points_difference'] = df['home_adjusted_points_avg'] - df['away_adjusted_points_avg']

        if 'home_adjusted_goal_diff_avg' in df.columns and 'away_adjusted_goal_diff_avg' in df.columns:
            df['adjusted_goal_diff_difference'] = (
                df['home_adjusted_goal_diff_avg'] - df['away_adjusted_goal_diff_avg']
            )

        if 'home_elo' in df.columns and 'away_elo' in df.columns:
            df['elo_difference'] = df['home_elo'] - df['away_elo']

        if (
            'home_elo' in df.columns
            and 'away_elo' in df.columns
            and 'league_home_advantage' in df.columns
        ):
            df['contextual_elo_difference'] = (
                df['home_elo'] + df['league_home_advantage'] - df['away_elo']
            )

        if 'home_top_tier_points_avg' in df.columns and 'away_top_tier_points_avg' in df.columns:
            df['top_tier_points_difference'] = (
                df['home_top_tier_points_avg'] - df['away_top_tier_points_avg']
            )

        if 'home_mid_tier_points_avg' in df.columns and 'away_mid_tier_points_avg' in df.columns:
            df['mid_tier_points_difference'] = (
                df['home_mid_tier_points_avg'] - df['away_mid_tier_points_avg']
            )

        if 'home_bottom_tier_points_avg' in df.columns and 'away_bottom_tier_points_avg' in df.columns:
            df['bottom_tier_points_difference'] = (
                df['home_bottom_tier_points_avg'] - df['away_bottom_tier_points_avg']
            )

        if 'home_win_streak' in df.columns and 'away_win_streak' in df.columns:
            df['win_streak_difference'] = df['home_win_streak'] - df['away_win_streak']

        if 'home_unbeaten_streak' in df.columns and 'away_unbeaten_streak' in df.columns:
            df['unbeaten_streak_difference'] = (
                df['home_unbeaten_streak'] - df['away_unbeaten_streak']
            )

        if 'home_home_form' in df.columns and 'away_away_form' in df.columns:
            df['venue_form_difference'] = df['home_home_form'] - df['away_away_form']

        if 'home_home_win_rate' in df.columns and 'away_away_win_rate' in df.columns:
            df['venue_win_rate_difference'] = (
                df['home_home_win_rate'] - df['away_away_win_rate']
            )
        
        if 'home_rest_days' in df.columns and 'away_rest_days' in df.columns:
            df['rest_days_difference'] = df['home_rest_days'] - df['away_rest_days']

        odds_columns = ['home_odds', 'draw_odds', 'away_odds', 'over_2_5_odds', 'under_2_5_odds']
        if any(column in df.columns for column in odds_columns):
            odds_frame = df.reindex(columns=odds_columns).copy()
            odds_frame = odds_frame.apply(pd.to_numeric, errors='coerce')
            neutral_odds = odds_frame.median(numeric_only=True).fillna(3.0)
            odds_frame = odds_frame.fillna(neutral_odds)
            odds_available_mask = None
            if 'odds_data_available' in df.columns:
                odds_available_mask = df['odds_data_available'].to_numpy(dtype=float) >= 0.99

            clipped_home_odds = odds_frame['home_odds'].clip(lower=1.01)
            clipped_draw_odds = odds_frame['draw_odds'].clip(lower=1.01)
            clipped_away_odds = odds_frame['away_odds'].clip(lower=1.01)
            clipped_over_2_5_odds = odds_frame['over_2_5_odds'].clip(lower=1.01)
            clipped_under_2_5_odds = odds_frame['under_2_5_odds'].clip(lower=1.01)
            df['log_home_odds'] = np.log(clipped_home_odds)
            df['log_draw_odds'] = np.log(clipped_draw_odds)
            df['log_away_odds'] = np.log(clipped_away_odds)
            df['log_over_2_5_odds'] = np.log(clipped_over_2_5_odds)
            df['log_under_2_5_odds'] = np.log(clipped_under_2_5_odds)
            df['odds_range'] = clipped_away_odds.combine(clipped_home_odds, max) - clipped_away_odds.combine(clipped_home_odds, min)
            df['home_away_odds_gap'] = clipped_away_odds - clipped_home_odds
            df['totals_2_5_odds_gap'] = clipped_over_2_5_odds - clipped_under_2_5_odds
            df['implied_home_probability'] = 1.0 / clipped_home_odds
            df['implied_draw_probability'] = 1.0 / clipped_draw_odds
            df['implied_away_probability'] = 1.0 / clipped_away_odds
            df['implied_over_2_5_probability'] = 1.0 / clipped_over_2_5_odds
            df['implied_under_2_5_probability'] = 1.0 / clipped_under_2_5_odds
            df['market_overround'] = (
                df['implied_home_probability']
                + df['implied_draw_probability']
                + df['implied_away_probability']
            )
            df['totals_2_5_overround'] = (
                df['implied_over_2_5_probability']
                + df['implied_under_2_5_probability']
            )
            normalized_denominator = df['market_overround'].replace(0.0, np.nan)
            df['normalized_home_probability'] = df['implied_home_probability'] / normalized_denominator
            df['normalized_draw_probability'] = df['implied_draw_probability'] / normalized_denominator
            df['normalized_away_probability'] = df['implied_away_probability'] / normalized_denominator
            normalized_probabilities = np.vstack([
                df['normalized_home_probability'].fillna(1 / 3).to_numpy(dtype=float),
                df['normalized_draw_probability'].fillna(1 / 3).to_numpy(dtype=float),
                df['normalized_away_probability'].fillna(1 / 3).to_numpy(dtype=float),
            ]).T
            sorted_probabilities = np.sort(normalized_probabilities, axis=1)
            df['market_favorite_probability'] = sorted_probabilities[:, -1]
            df['market_probability_gap'] = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
            df['market_probability_ratio'] = sorted_probabilities[:, -1] / np.clip(sorted_probabilities[:, -2], 1e-6, None)
            df['market_entropy'] = -np.sum(
                np.clip(normalized_probabilities, 1e-12, 1.0) * np.log(np.clip(normalized_probabilities, 1e-12, 1.0)),
                axis=1,
            )
            df['market_home_edge'] = df['implied_home_probability'] - df['implied_away_probability']
            df['market_draw_edge'] = df['implied_draw_probability'] - (
                (df['implied_home_probability'] + df['implied_away_probability']) / 2.0
            )

            # Для строк без реальных odds обнуляем market-сигналы до нейтральных значений,
            # чтобы модель не училась на искусственно импутированных коэффициентах.
            if odds_available_mask is not None:
                unavailable_mask = ~odds_available_mask
                if np.any(unavailable_mask):
                    df.loc[unavailable_mask, 'log_home_odds'] = float(np.log(3.0))
                    df.loc[unavailable_mask, 'log_draw_odds'] = float(np.log(3.0))
                    df.loc[unavailable_mask, 'log_away_odds'] = float(np.log(3.0))
                    df.loc[unavailable_mask, 'odds_range'] = 0.0
                    df.loc[unavailable_mask, 'home_away_odds_gap'] = 0.0
                    df.loc[unavailable_mask, 'implied_home_probability'] = 1.0 / 3.0
                    df.loc[unavailable_mask, 'implied_draw_probability'] = 1.0 / 3.0
                    df.loc[unavailable_mask, 'implied_away_probability'] = 1.0 / 3.0
                    df.loc[unavailable_mask, 'market_overround'] = 1.0
                    df.loc[unavailable_mask, 'normalized_home_probability'] = 1.0 / 3.0
                    df.loc[unavailable_mask, 'normalized_draw_probability'] = 1.0 / 3.0
                    df.loc[unavailable_mask, 'normalized_away_probability'] = 1.0 / 3.0
                    df.loc[unavailable_mask, 'market_favorite_probability'] = 1.0 / 3.0
                    df.loc[unavailable_mask, 'market_probability_gap'] = 0.0
                    df.loc[unavailable_mask, 'market_probability_ratio'] = 1.0
                    df.loc[unavailable_mask, 'market_entropy'] = float(np.log(3.0))
                    df.loc[unavailable_mask, 'market_home_edge'] = 0.0
                    df.loc[unavailable_mask, 'market_draw_edge'] = 0.0

        poisson_base_columns = [
            'home_goals_for_avg',
            'away_goals_for_avg',
            'home_goals_against_avg',
            'away_goals_against_avg',
        ]
        if all(column in df.columns for column in poisson_base_columns):
            home_attack = df['home_goals_for_avg'].clip(lower=0.1)
            away_attack = df['away_goals_for_avg'].clip(lower=0.1)
            home_defense = df['home_goals_against_avg'].clip(lower=0.1)
            away_defense = df['away_goals_against_avg'].clip(lower=0.1)

            home_lambda = 0.55 * home_attack + 0.45 * away_defense
            away_lambda = 0.55 * away_attack + 0.45 * home_defense

            if 'contextual_elo_difference' in df.columns:
                elo_adjustment = np.clip(df['contextual_elo_difference'] / 800.0, -0.35, 0.35)
                home_lambda = home_lambda * np.exp(elo_adjustment)
                away_lambda = away_lambda * np.exp(-elo_adjustment)
            elif 'elo_difference' in df.columns:
                elo_adjustment = np.clip(df['elo_difference'] / 850.0, -0.30, 0.30)
                home_lambda = home_lambda * np.exp(elo_adjustment)
                away_lambda = away_lambda * np.exp(-elo_adjustment)

            if 'normalized_home_probability' in df.columns and 'normalized_away_probability' in df.columns:
                market_skew = np.clip(
                    (df['normalized_home_probability'] - df['normalized_away_probability']) * 0.35,
                    -0.25,
                    0.25,
                )
                home_lambda = home_lambda * (1.0 + market_skew)
                away_lambda = away_lambda * (1.0 - market_skew)

            home_lambda = home_lambda.clip(lower=0.05, upper=4.5)
            away_lambda = away_lambda.clip(lower=0.05, upper=4.5)

            poisson_probabilities = [
                FeatureEngineer._poisson_outcome_probabilities(home_rate, away_rate)
                for home_rate, away_rate in zip(home_lambda.to_numpy(dtype=float), away_lambda.to_numpy(dtype=float))
            ]
            poisson_array = np.asarray(poisson_probabilities, dtype=float)

            df['poisson_home_xg'] = home_lambda
            df['poisson_away_xg'] = away_lambda
            df['poisson_total_xg'] = home_lambda + away_lambda
            df['poisson_goal_diff_xg'] = home_lambda - away_lambda
            df['poisson_home_win_probability'] = poisson_array[:, 0]
            df['poisson_draw_probability'] = poisson_array[:, 1]
            df['poisson_away_win_probability'] = poisson_array[:, 2]
        
        return df
    
    @staticmethod
    def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """Добавить временные признаки"""
        df = df.copy()
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['week_of_year'] = df['date'].dt.isocalendar().week
        
        return df
    
    @staticmethod
    def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
        """Добавить сезонные признаки"""
        df = df.copy()
        
        if 'month' in df.columns:
            # Группы месяцев
            df['season'] = pd.cut(
                df['month'],
                bins=[0, 3, 6, 9, 12],
                labels=[0, 1, 2, 3]
            )
        
        return df
