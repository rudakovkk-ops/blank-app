"""Rule-based модель Rudy на основе последних матчей из локальной SQLite базы."""
import logging
import sqlite3
from math import exp, tanh

import numpy as np
import pandas as pd

from data.cache_manager import CacheManager
from models.base_model import BaseModel

logger = logging.getLogger(__name__)


class RudyModel(BaseModel):
    """Модель Rudy: 5 последних домашних, 5 гостевых и 5 очных матчей."""

    def __init__(self, home_window: int = 5, away_window: int = 5, h2h_window: int = 5):
        super().__init__(name="Rudy", model_type="rudy")
        self.home_window = int(home_window)
        self.away_window = int(away_window)
        self.h2h_window = int(h2h_window)
        self.cache = CacheManager()
        self.is_trained = True
        self.metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": 0.0,
            "brier_score": 0.0,
            "expected_calibration_error": 0.0,
            "calibration_method": "rule_based",
            "calibration_brier_before": 0.0,
            "calibration_brier_after": 0.0,
            "calibration_brier_improvement": 0.0,
            "optuna_best_score": 0.0,
            "optuna_target_metric": "rule_based",
            "optuna_trials": 0,
            "optuna_best_params": {},
            "is_optuna_tuned": False,
            "is_probability_calibrated": False,
            "reliability_curve": [],
            "active_feature_groups": ["rudy_form", "rudy_h2h"],
            "feature_count": 0,
            "prediction_policy_enabled": False,
            "prediction_policy_confidence_threshold": 0.0,
            "prediction_policy_margin_threshold": 0.0,
            "prediction_policy_coverage": 1.0,
            "prediction_policy_selective_accuracy": 0.0,
            "rule_home_window": self.home_window,
            "rule_away_window": self.away_window,
            "rule_h2h_window": self.h2h_window,
        }

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> bool:
        del X_train, y_train
        self.is_trained = True
        return True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if X is None:
            return np.array([])
        return np.ones(len(X), dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if X is None:
            return np.zeros((0, 3), dtype=float)
        return np.ones((len(X), 3), dtype=float) / 3.0

    @staticmethod
    def _build_team_filter(team_id: int | None, team_name: str | None, column_id: str, column_name: str) -> tuple[str, list]:
        if team_id is not None:
            return f"{column_id} = ?", [int(team_id)]
        return f"{column_name} = ?", [str(team_name or "")]

    @staticmethod
    def _to_match_dicts(rows: list[tuple]) -> list[dict]:
        return [
            {
                "date": row[0],
                "home_team": row[1],
                "away_team": row[2],
                "home_goals": int(row[3] if row[3] is not None else 0),
                "away_goals": int(row[4] if row[4] is not None else 0),
            }
            for row in rows
        ]

    def _fetch_last_home_matches(
        self,
        team_id: int | None,
        team_name: str | None,
        league_id: int | None,
    ) -> list[dict]:
        filter_sql, params = self._build_team_filter(team_id, team_name, "home_team_id", "home_team")
        query = """
            SELECT date, home_team, away_team, home_goals, away_goals
            FROM fixtures
            WHERE status = 'FT'
              AND home_goals IS NOT NULL
              AND away_goals IS NOT NULL
              AND ({team_filter})
        """.format(team_filter=filter_sql)
        if league_id is not None:
            query += " AND league_id = ?"
            params.append(int(league_id))
        query += " ORDER BY date DESC LIMIT ?"
        params.append(self.home_window)

        with sqlite3.connect(self.cache.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
        return self._to_match_dicts(rows)

    def _fetch_last_away_matches(
        self,
        team_id: int | None,
        team_name: str | None,
        league_id: int | None,
    ) -> list[dict]:
        filter_sql, params = self._build_team_filter(team_id, team_name, "away_team_id", "away_team")
        query = """
            SELECT date, home_team, away_team, home_goals, away_goals
            FROM fixtures
            WHERE status = 'FT'
              AND home_goals IS NOT NULL
              AND away_goals IS NOT NULL
              AND ({team_filter})
        """.format(team_filter=filter_sql)
        if league_id is not None:
            query += " AND league_id = ?"
            params.append(int(league_id))
        query += " ORDER BY date DESC LIMIT ?"
        params.append(self.away_window)

        with sqlite3.connect(self.cache.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
        return self._to_match_dicts(rows)

    def _fetch_last_h2h_matches(
        self,
        home_team_id: int | None,
        home_team_name: str | None,
        away_team_id: int | None,
        away_team_name: str | None,
        league_id: int | None,
    ) -> list[dict]:
        home_filter_sql, home_params = self._build_team_filter(home_team_id, home_team_name, "home_team_id", "home_team")
        away_filter_sql, away_params = self._build_team_filter(away_team_id, away_team_name, "away_team_id", "away_team")
        reverse_home_filter_sql, reverse_home_params = self._build_team_filter(home_team_id, home_team_name, "away_team_id", "away_team")
        reverse_away_filter_sql, reverse_away_params = self._build_team_filter(away_team_id, away_team_name, "home_team_id", "home_team")

        params: list = []
        query = """
            SELECT date, home_team, away_team, home_goals, away_goals
            FROM fixtures
            WHERE status = 'FT'
              AND home_goals IS NOT NULL
              AND away_goals IS NOT NULL
              AND (
                    (({home_clause}) AND ({away_clause}))
                 OR (({reverse_away_clause}) AND ({reverse_home_clause}))
              )
        """.format(
            home_clause=home_filter_sql,
            away_clause=away_filter_sql,
            reverse_away_clause=reverse_away_filter_sql,
            reverse_home_clause=reverse_home_filter_sql,
        )
        params.extend(home_params)
        params.extend(away_params)
        params.extend(reverse_away_params)
        params.extend(reverse_home_params)

        if league_id is not None:
            query += " AND league_id = ?"
            params.append(int(league_id))

        query += " ORDER BY date DESC LIMIT ?"
        params.append(self.h2h_window)

        with sqlite3.connect(self.cache.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
        return self._to_match_dicts(rows)

    def _infer_league_id_for_pair(
        self,
        home_team_id: int | None,
        home_team_name: str | None,
        away_team_id: int | None,
        away_team_name: str | None,
    ) -> int | None:
        """Определить наиболее релевантную лигу для пары команд по истории очных матчей."""
        home_filter_sql, home_params = self._build_team_filter(home_team_id, home_team_name, "home_team_id", "home_team")
        away_filter_sql, away_params = self._build_team_filter(away_team_id, away_team_name, "away_team_id", "away_team")
        reverse_home_filter_sql, reverse_home_params = self._build_team_filter(home_team_id, home_team_name, "away_team_id", "away_team")
        reverse_away_filter_sql, reverse_away_params = self._build_team_filter(away_team_id, away_team_name, "home_team_id", "home_team")

        query = """
            SELECT league_id
            FROM fixtures
            WHERE status = 'FT'
              AND league_id IS NOT NULL
              AND (
                    (({home_clause}) AND ({away_clause}))
                 OR (({reverse_away_clause}) AND ({reverse_home_clause}))
              )
            GROUP BY league_id
            ORDER BY COUNT(*) DESC, MAX(date) DESC
            LIMIT 1
        """.format(
            home_clause=home_filter_sql,
            away_clause=away_filter_sql,
            reverse_away_clause=reverse_away_filter_sql,
            reverse_home_clause=reverse_home_filter_sql,
        )

        params = []
        params.extend(home_params)
        params.extend(away_params)
        params.extend(reverse_away_params)
        params.extend(reverse_home_params)

        with sqlite3.connect(self.cache.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()
        return int(row[0]) if row and row[0] is not None else None

    @staticmethod
    def _points_and_goal_diff(matches: list[dict], as_home: bool) -> tuple[float, float, float, float]:
        if not matches:
            return 0.0, 0.0, 0.0, 0.0

        points = 0.0
        draw_count = 0
        goals_for = 0.0
        goals_against = 0.0

        for match in matches:
            home_goals = float(match["home_goals"])
            away_goals = float(match["away_goals"])
            if as_home:
                gf, ga = home_goals, away_goals
            else:
                gf, ga = away_goals, home_goals

            goals_for += gf
            goals_against += ga

            if gf > ga:
                points += 3.0
            elif gf == ga:
                points += 1.0
                draw_count += 1

        count = float(len(matches))
        ppg_norm = (points / count) / 3.0
        gd_avg = (goals_for - goals_against) / count
        draw_rate = draw_count / count
        goals_for_avg = goals_for / count
        return ppg_norm, gd_avg, draw_rate, goals_for_avg

    @staticmethod
    def _h2h_points_for_home_team(matches: list[dict], home_team_name: str) -> tuple[float, float]:
        if not matches:
            return 0.0, 0.0

        home_points = 0.0
        away_points = 0.0
        draw_count = 0

        for match in matches:
            hg = float(match["home_goals"])
            ag = float(match["away_goals"])
            home_is_team1 = match["home_team"] == home_team_name

            if hg == ag:
                home_points += 1.0
                away_points += 1.0
                draw_count += 1
                continue

            team1_won = (hg > ag and home_is_team1) or (ag > hg and not home_is_team1)
            if team1_won:
                home_points += 3.0
            else:
                away_points += 3.0

        max_points = 3.0 * len(matches)
        return home_points / max_points, draw_count / float(len(matches))

    @staticmethod
    def _gd_to_unit(gd_avg: float) -> float:
        return 0.5 + 0.5 * tanh(gd_avg / 2.0)

    @staticmethod
    def _safe_softmax_pair(home_strength: float, away_strength: float) -> tuple[float, float]:
        exp_home = exp(home_strength)
        exp_away = exp(away_strength)
        denom = max(exp_home + exp_away, 1e-9)
        return exp_home / denom, exp_away / denom

    def _compute_probabilities(
        self,
        home_matches: list[dict],
        away_matches: list[dict],
        h2h_matches: list[dict],
        home_team_name: str,
    ) -> np.ndarray:
        if not home_matches and not away_matches and not h2h_matches:
            return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float)

        home_ppg, home_gd_avg, home_draw_rate, _ = self._points_and_goal_diff(home_matches, as_home=True)
        away_ppg, away_gd_avg, away_draw_rate, _ = self._points_and_goal_diff(away_matches, as_home=False)
        h2h_home_ppg, h2h_draw_rate = self._h2h_points_for_home_team(h2h_matches, home_team_name)

        home_strength = 0.45 * home_ppg + 0.25 * self._gd_to_unit(home_gd_avg) + 0.30 * h2h_home_ppg
        away_strength = 0.45 * away_ppg + 0.25 * self._gd_to_unit(away_gd_avg) + 0.30 * (1.0 - h2h_home_ppg)

        avg_draw_rate = (home_draw_rate + away_draw_rate + h2h_draw_rate) / 3.0
        draw_probability = float(np.clip(0.14 + 0.28 * avg_draw_rate + 0.20 * (1.0 - abs(home_strength - away_strength)), 0.10, 0.42))

        home_share, away_share = self._safe_softmax_pair(home_strength, away_strength)
        non_draw = 1.0 - draw_probability
        home_probability = non_draw * home_share
        away_probability = non_draw * away_share

        probabilities = np.array([home_probability, draw_probability, away_probability], dtype=float)
        probabilities = probabilities / np.clip(probabilities.sum(), 1e-9, None)
        return probabilities

    def predict_from_fixture(self, fixture: dict) -> dict:
        teams = fixture.get("teams", {})
        league = fixture.get("league", {})

        home_team = teams.get("home", {})
        away_team = teams.get("away", {})
        home_team_name = str(home_team.get("name", ""))
        away_team_name = str(away_team.get("name", ""))
        home_team_id = home_team.get("id")
        away_team_id = away_team.get("id")
        league_id = league.get("id")
        if league_id is None:
            league_id = self._infer_league_id_for_pair(home_team_id, home_team_name, away_team_id, away_team_name)

        home_matches = self._fetch_last_home_matches(home_team_id, home_team_name, league_id)
        away_matches = self._fetch_last_away_matches(away_team_id, away_team_name, league_id)
        h2h_matches = self._fetch_last_h2h_matches(
            home_team_id,
            home_team_name,
            away_team_id,
            away_team_name,
            league_id,
        )

        probabilities = self._compute_probabilities(home_matches, away_matches, h2h_matches, home_team_name)

        return {
            "probabilities": probabilities,
            "context": {
                "home_team": home_team_name,
                "away_team": away_team_name,
                "league": league.get("name", "Неизвестная лига"),
                "league_id": league_id,
                "home_recent_home": home_matches,
                "away_recent_away": away_matches,
                "h2h_recent": h2h_matches,
                "home_window": self.home_window,
                "away_window": self.away_window,
                "h2h_window": self.h2h_window,
            },
        }
