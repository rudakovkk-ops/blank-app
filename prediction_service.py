"""Сервис инференса для Streamlit-интерфейса."""
import logging
import json
from datetime import datetime

import numpy as np
import pandas as pd

from config.settings import (
    MODELS_DIR,
    TRACKED_LEAGUES,
    TRAINING_DATA_SEASONS,
    FEATURE_DIAGNOSTIC_COLUMNS,
    DASHBOARD_PREDICTIONS_CACHE_TTL,
    LIVE_SNAPSHOTS_ENABLED,
    LIVE_SNAPSHOT_MIN_MINUTE,
    LIVE_SNAPSHOT_MAX_MINUTE,
    IN_PLAY_ML_MIN_ROWS,
    PREDICTION_ABSTAIN_ENABLED,
    PREDICTION_MIN_CONFIDENCE,
    PREDICTION_MIN_MARGIN,
    PREDICTION_POLICY_CONFIDENCE_FLOOR,
    PREDICTION_POLICY_MARGIN_FLOOR,
)
from data.api_client import FootballAPIClient
from data.cache_manager import CacheManager
from data.data_processor import DataProcessor
from data.feature_engineer import FeatureEngineer
from models.base_model import BaseModel
from models.rudy_model import RudyModel
from models.rudy2_model import Rudy2Model
from models.rudy3_model import Rudy3Model

logger = logging.getLogger(__name__)

# Проект работает только в режиме rule-based Rudy-моделей, обучение и загрузка других моделей удалены.
MODELS_ENABLED = False


class PredictionService:
    """Загрузка обученных моделей и подготовка признаков для инференса."""

    DASHBOARD_CACHE_KEY = "dashboard_precomputed_predictions"
    DASHBOARD_STATE_CACHE_KEY = "dashboard_prediction_preparation_state"

    CLASS_LABELS = {
        0: "Победа хозяев",
        1: "Ничья",
        2: "Победа гостей",
    }

    PROBABILITY_KEY_ALIASES = {
        "Победа хозяев": ("Победа хозяев", "Home Win"),
        "Ничья": ("Ничья", "Draw"),
        "Победа гостей": ("Победа гостей", "Away Win"),
    }

    MODEL_LABELS = {
        "rudy": "Rudy",
        "rudy2": "Rudy2",
        "rudy3": "Rudy3",
    }

    MODEL_ABBREV = {
        "rudy": "RUD",
        "rudy2": "RU2",
        "rudy3": "RU3",
    }

    MODEL_ORDER = ["rudy", "rudy2", "rudy3"]

    def __init__(self):
        self.api_client = FootballAPIClient()
        self.cache = CacheManager()
        self.rudy_model = RudyModel()
        self.rudy2_model = Rudy2Model()
        self.rudy3_model = Rudy3Model()
        self._history_cache = {}
        self._standings_cache = {}
        self._roi_analysis_cache = None
        self.trainer = None
        self.processor = DataProcessor()
        self.feature_engineer = FeatureEngineer()
        self.training_columns = []
        self.feature_defaults = {}
        self.loaded_models = {}
        self.trained_in_play_snapshot_model = None
        self.primary_model_key = "rudy"

    def _build_probability_map(self, probabilities: np.ndarray) -> dict:
        """Преобразовать массив вероятностей в словарь outcome -> probability."""
        return {
            self.CLASS_LABELS[index]: float(probabilities[index])
            for index in range(min(len(probabilities), len(self.CLASS_LABELS)))
        }

    @classmethod
    def get_probability_value(cls, probability_map: dict, outcome_label: str) -> float:
        """Получить вероятность по русскому label с fallback на старые английские ключи."""
        for key in cls.PROBABILITY_KEY_ALIASES.get(outcome_label, (outcome_label,)):
            if key in probability_map:
                return float(probability_map[key])
        return 0.0

    def _predict_with_model(self, model: BaseModel, model_key: str, normalized_row: pd.DataFrame) -> dict:
        """Сделать предсказание уже на подготовленной строке признаков."""
        probabilities = model.get_calibrated_probabilities(normalized_row)[0]
        prediction_class = int(np.argmax(probabilities))
        probability_map = self._build_probability_map(probabilities)
        sorted_probabilities = np.sort(np.asarray(probabilities, dtype=float))
        confidence = float(sorted_probabilities[-1])
        margin = float(sorted_probabilities[-1] - sorted_probabilities[-2])
        entropy = float(-np.sum(np.clip(probabilities, 1e-12, 1.0) * np.log(np.clip(probabilities, 1e-12, 1.0))))

        policy = model.metrics.get('prediction_policy', {}) or {}
        confidence_threshold = max(
            float(policy.get('confidence_threshold', PREDICTION_MIN_CONFIDENCE)),
            float(PREDICTION_POLICY_CONFIDENCE_FLOOR),
        )
        margin_threshold = max(
            float(policy.get('margin_threshold', PREDICTION_MIN_MARGIN)),
            float(PREDICTION_POLICY_MARGIN_FLOOR),
        )
        abstain_enabled = bool(policy.get('abstain_enabled', False)) and PREDICTION_ABSTAIN_ENABLED
        is_abstained = abstain_enabled and (
            confidence < confidence_threshold or margin < margin_threshold
        )
        abstain_reason = None
        if is_abstained:
            if confidence < confidence_threshold and margin < margin_threshold:
                abstain_reason = 'low_confidence_and_small_margin'
            elif confidence < confidence_threshold:
                abstain_reason = 'low_confidence'
            else:
                abstain_reason = 'small_margin'

        return {
            "model_key": model_key,
            "model_label": self.MODEL_LABELS.get(model_key, model_key),
            "prediction_class": prediction_class,
            "raw_prediction_label": self.CLASS_LABELS.get(prediction_class, "Неизвестно"),
            "prediction_label": (
                "Нет уверенного прогноза" if is_abstained
                else self.CLASS_LABELS.get(prediction_class, "Неизвестно")
            ),
            "confidence": confidence,
            "confidence_margin": margin,
            "uncertainty_entropy": entropy,
            "is_abstained": is_abstained,
            "abstain_reason": abstain_reason,
            "confidence_threshold": confidence_threshold,
            "margin_threshold": margin_threshold,
            "probabilities": probability_map,
            "prediction_mode": "pre_match",
        }

    def _enrich_fixture_for_inference(self, fixture: dict) -> dict:
        """Дозагрузить odds и live/historical statistics для текущего fixture перед инференсом."""
        return self.api_client.enrich_fixture_details(
            fixture,
            include_statistics=True,
            include_odds=True,
        )

    def _load_trained_in_play_snapshot_model(self):
        """In-play ML модель удалена вместе с обучающим контуром."""
        return None

    def _predict_with_rudy(self, fixture: dict) -> dict:
        """Rule-based прогноз Rudy по схеме 5-5-5 из локальной базы матчей."""
        result = self.rudy_model.predict_from_fixture(fixture)
        probabilities = np.asarray(result.get("probabilities", np.ones(3) / 3.0), dtype=float)
        probabilities = probabilities / np.clip(np.sum(probabilities), 1e-12, None)
        prediction_class = int(np.argmax(probabilities))
        sorted_probabilities = np.sort(probabilities)
        confidence = float(sorted_probabilities[-1])
        margin = float(sorted_probabilities[-1] - sorted_probabilities[-2])
        entropy = float(-np.sum(np.clip(probabilities, 1e-12, 1.0) * np.log(np.clip(probabilities, 1e-12, 1.0))))

        return {
            "model_key": "rudy",
            "model_label": self.MODEL_LABELS.get("rudy", "Rudy"),
            "prediction_class": prediction_class,
            "raw_prediction_label": self.CLASS_LABELS.get(prediction_class, "Неизвестно"),
            "prediction_label": self.CLASS_LABELS.get(prediction_class, "Неизвестно"),
            "confidence": confidence,
            "confidence_margin": margin,
            "uncertainty_entropy": entropy,
            "is_abstained": False,
            "abstain_reason": None,
            "confidence_threshold": 0.0,
            "margin_threshold": 0.0,
            "probabilities": self._build_probability_map(probabilities),
            "prediction_mode": "rudy_rule_based",
            "rudy_context": result.get("context", {}),
        }

    def _predict_with_rudy2(self, fixture: dict) -> dict:
        """Rule-based прогноз Rudy2 по схеме 4-4-4 из локальной базы матчей."""
        result = self.rudy2_model.predict_from_fixture(fixture)
        probabilities = np.asarray(result.get("probabilities", np.ones(3) / 3.0), dtype=float)
        probabilities = probabilities / np.clip(np.sum(probabilities), 1e-12, None)
        prediction_class = int(np.argmax(probabilities))
        sorted_probabilities = np.sort(probabilities)
        confidence = float(sorted_probabilities[-1])
        margin = float(sorted_probabilities[-1] - sorted_probabilities[-2])
        entropy = float(-np.sum(np.clip(probabilities, 1e-12, 1.0) * np.log(np.clip(probabilities, 1e-12, 1.0))))

        return {
            "model_key": "rudy2",
            "model_label": self.MODEL_LABELS.get("rudy2", "Rudy2"),
            "prediction_class": prediction_class,
            "raw_prediction_label": self.CLASS_LABELS.get(prediction_class, "Неизвестно"),
            "prediction_label": self.CLASS_LABELS.get(prediction_class, "Неизвестно"),
            "confidence": confidence,
            "confidence_margin": margin,
            "uncertainty_entropy": entropy,
            "is_abstained": False,
            "abstain_reason": None,
            "confidence_threshold": 0.0,
            "margin_threshold": 0.0,
            "probabilities": self._build_probability_map(probabilities),
            "prediction_mode": "rudy2_rule_based",
            "rudy_context": result.get("context", {}),
        }

    def _predict_with_rudy3(self, fixture: dict) -> dict:
        """Rule-based прогноз Rudy3 по схеме 6-6-6 из локальной базы матчей."""
        result = self.rudy3_model.predict_from_fixture(fixture)
        probabilities = np.asarray(result.get("probabilities", np.ones(3) / 3.0), dtype=float)
        probabilities = probabilities / np.clip(np.sum(probabilities), 1e-12, None)
        prediction_class = int(np.argmax(probabilities))
        sorted_probabilities = np.sort(probabilities)
        confidence = float(sorted_probabilities[-1])
        margin = float(sorted_probabilities[-1] - sorted_probabilities[-2])
        entropy = float(-np.sum(np.clip(probabilities, 1e-12, 1.0) * np.log(np.clip(probabilities, 1e-12, 1.0))))

        return {
            "model_key": "rudy3",
            "model_label": self.MODEL_LABELS.get("rudy3", "Rudy3"),
            "prediction_class": prediction_class,
            "raw_prediction_label": self.CLASS_LABELS.get(prediction_class, "Неизвестно"),
            "prediction_label": self.CLASS_LABELS.get(prediction_class, "Неизвестно"),
            "confidence": confidence,
            "confidence_margin": margin,
            "uncertainty_entropy": entropy,
            "is_abstained": False,
            "abstain_reason": None,
            "confidence_threshold": 0.0,
            "margin_threshold": 0.0,
            "probabilities": self._build_probability_map(probabilities),
            "prediction_mode": "rudy3_rule_based",
            "rudy_context": result.get("context", {}),
        }

    def _predict_with_rudy_super(self, fixture: dict) -> dict:
        """Агрегированный прогноз RudySuper, сравнивающий Rudy/Rudy2/Rudy3."""
        # Получаем предсказания от всех трёх моделей
        pred_rudy = self._predict_with_rudy(fixture)
        pred_rudy2 = self._predict_with_rudy2(fixture)
        pred_rudy3 = self._predict_with_rudy3(fixture)

        # Собираем вероятности и агрегируем (усредняем)
        probs_rudy = np.asarray([v for v in pred_rudy['probabilities'].values()], dtype=float)
        probs_rudy2 = np.asarray([v for v in pred_rudy2['probabilities'].values()], dtype=float)
        probs_rudy3 = np.asarray([v for v in pred_rudy3['probabilities'].values()], dtype=float)

        # Средние вероятности
        aggregated_probabilities = (probs_rudy + probs_rudy2 + probs_rudy3) / 3.0
        aggregated_probabilities = aggregated_probabilities / np.clip(np.sum(aggregated_probabilities), 1e-12, None)

        # Определяем лучший класс
        prediction_class = int(np.argmax(aggregated_probabilities))
        sorted_probabilities = np.sort(aggregated_probabilities)
        confidence = float(sorted_probabilities[-1])
        margin = float(sorted_probabilities[-1] - sorted_probabilities[-2])
        entropy = float(-np.sum(np.clip(aggregated_probabilities, 1e-12, 1.0) * np.log(np.clip(aggregated_probabilities, 1e-12, 1.0))))

        # Проверяем согласие моделей
        individual_predictions = [
            pred_rudy['prediction_class'],
            pred_rudy2['prediction_class'],
            pred_rudy3['prediction_class'],
        ]
        agreement_count = sum(1 for p in individual_predictions if p == prediction_class)
        agreement_level = "полное согласие (3/3)" if agreement_count == 3 else f"{agreement_count}/3 согласны"

        return {
            "model_key": "rudy_super",
            "model_label": "RudySuper",
            "prediction_class": prediction_class,
            "raw_prediction_label": self.CLASS_LABELS.get(prediction_class, "Неизвестно"),
            "prediction_label": self.CLASS_LABELS.get(prediction_class, "Неизвестно"),
            "confidence": confidence,
            "confidence_margin": margin,
            "uncertainty_entropy": entropy,
            "is_abstained": False,
            "abstain_reason": None,
            "confidence_threshold": 0.0,
            "margin_threshold": 0.0,
            "probabilities": self._build_probability_map(aggregated_probabilities),
            "prediction_mode": "rudy_super_ensemble",
            "agreement_level": agreement_level,
            "individual_predictions": {
                "rudy": pred_rudy['prediction_label'],
                "rudy2": pred_rudy2['prediction_label'],
                "rudy3": pred_rudy3['prediction_label'],
            },
            "rudy_context": pred_rudy.get("rudy_context", {}),
        }

    @staticmethod
    def _rudy_form_summary(matches: list[dict], as_home: bool, prefix: str) -> dict:
        if not matches:
            return {
                "text": f"{prefix}: недостаточно данных",
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "gf": 0,
                "ga": 0,
                "gf_avg": 0.0,
                "ga_avg": 0.0,
            }

        wins = 0
        draws = 0
        losses = 0
        gf = 0
        ga = 0
        for match in matches:
            hg = int(match.get('home_goals', 0) or 0)
            ag = int(match.get('away_goals', 0) or 0)
            team_gf = hg if as_home else ag
            team_ga = ag if as_home else hg
            gf += team_gf
            ga += team_ga

            if team_gf > team_ga:
                wins += 1
            elif team_gf == team_ga:
                draws += 1
            else:
                losses += 1

        count = max(len(matches), 1)
        gf_avg = gf / count
        ga_avg = ga / count

        if wins >= 4:
            form_note = "🔥 Форма очень сильная"
        elif losses >= 4:
            form_note = "⚠️ Форма крайне нестабильная!"
        elif wins >= losses:
            form_note = "Форма в целом положительная"
        else:
            form_note = "Форма нестабильная"

        text = (
            f"{prefix}: {wins}П {draws}Н {losses}П | "
            f"Голы: {gf} забито / {ga} пропущено | "
            f"Среднее: {gf_avg:.1f} : {ga_avg:.1f} {form_note}"
        )
        return {
            "text": text,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "gf": gf,
            "ga": ga,
            "gf_avg": gf_avg,
            "ga_avg": ga_avg,
        }

    @staticmethod
    def _rudy_scoreline_candidates(home_expected: float, away_expected: float, outcome: str) -> list[str]:
        home_round = int(np.clip(round(home_expected), 0, 5))
        away_round = int(np.clip(round(away_expected), 0, 5))

        if outcome == 'Ничья':
            draw_base = int(np.clip(round((home_expected + away_expected) / 2.0), 0, 5))
            alt_draw = int(np.clip(draw_base + 1, 0, 5))
            candidates = [f"{draw_base}:{draw_base}", f"{alt_draw}:{alt_draw}"]
        elif outcome == 'Победа хозяев':
            if home_round <= away_round:
                home_round = min(5, away_round + 1)
            alt_home = int(np.clip(home_round + 1, 0, 5))
            alt_away = int(np.clip(min(away_round, alt_home - 1), 0, 5))
            candidates = [f"{home_round}:{away_round}", f"{alt_home}:{alt_away}"]
        else:
            if away_round <= home_round:
                away_round = min(5, home_round + 1)
            alt_away = int(np.clip(away_round + 1, 0, 5))
            alt_home = int(np.clip(min(home_round, alt_away - 1), 0, 5))
            candidates = [f"{home_round}:{away_round}", f"{alt_home}:{alt_away}"]

        unique_candidates = []
        for candidate in candidates:
            if candidate not in unique_candidates:
                unique_candidates.append(candidate)
        if len(unique_candidates) == 1:
            unique_candidates.append(unique_candidates[0])
        return unique_candidates[:2]

    @staticmethod
    def _motivation_from_rank(rank: int | None, total_teams: int) -> tuple[str, float]:
        """Оценить турнирную мотивацию команды по текущему месту в таблице."""
        if rank is None or rank <= 0 or total_teams <= 0:
            return "мотивация по таблице не определена", 0.50

        title_zone = 2
        euro_zone = min(6, max(4, int(round(total_teams * 0.28))))
        relegation_zone_start = max(total_teams - 2, 1)

        if rank <= title_zone:
            return "максимальная: борьба за титул", 0.95
        if rank <= euro_zone:
            return "высокая: борьба за еврокубки", 0.82
        if rank >= relegation_zone_start:
            return "максимальная: борьба за выживание", 0.92

        middle_low = int(total_teams * 0.35)
        middle_high = int(total_teams * 0.75)
        if middle_low <= rank <= middle_high:
            return "умеренная: стабильная середина", 0.58

        return "повышенная: важны очки", 0.70

    def _build_motivation_context(self, fixture: dict) -> dict:
        """Сформировать контекст мотивации команд по турнирной таблице."""
        teams = fixture.get('teams', {})
        home = teams.get('home', {})
        away = teams.get('away', {})

        standings_map = self._get_standings_map(fixture)
        known_ranks = [
            int(team_data.get('rank'))
            for team_data in standings_map.values()
            if isinstance(team_data, dict) and team_data.get('rank') is not None
        ]
        total_teams = max(known_ranks) if known_ranks else 20

        home_rank = None
        away_rank = None
        if home.get('id') in standings_map:
            home_rank = standings_map[home.get('id')].get('rank')
        if away.get('id') in standings_map:
            away_rank = standings_map[away.get('id')].get('rank')

        home_label, home_score = self._motivation_from_rank(home_rank, total_teams)
        away_label, away_score = self._motivation_from_rank(away_rank, total_teams)

        if home_score > away_score + 0.12:
            edge = f"мотивационный перевес у {home.get('name', 'хозяев')}"
        elif away_score > home_score + 0.12:
            edge = f"мотивационный перевес у {away.get('name', 'гостей')}"
        else:
            edge = "по мотивации команды сопоставимы"

        return {
            'home_rank': home_rank,
            'away_rank': away_rank,
            'home_label': home_label,
            'away_label': away_label,
            'home_score': home_score,
            'away_score': away_score,
            'edge': edge,
        }

    def _build_rudy_narrative(self, fixture: dict, prediction: dict) -> dict:
        context = prediction.get('rudy_context', {}) or {}
        home_team = context.get('home_team') or fixture.get('teams', {}).get('home', {}).get('name', 'Хозяева')
        away_team = context.get('away_team') or fixture.get('teams', {}).get('away', {}).get('name', 'Гости')
        league_name = context.get('league') or fixture.get('league', {}).get('name', 'Неизвестная лига')

        home_matches = context.get('home_recent_home', []) or []
        away_matches = context.get('away_recent_away', []) or []
        h2h_matches = context.get('h2h_recent', []) or []

        home_form = self._rudy_form_summary(home_matches, as_home=True, prefix='Дома')
        away_form = self._rudy_form_summary(away_matches, as_home=False, prefix='В гостях')

        probabilities = prediction.get('probabilities', {})
        p_home = float(self.get_probability_value(probabilities, 'Победа хозяев'))
        p_draw = float(self.get_probability_value(probabilities, 'Ничья'))
        p_away = float(self.get_probability_value(probabilities, 'Победа гостей'))

        home_expected = (home_form['gf_avg'] + away_form['ga_avg']) / 2.0
        away_expected = (away_form['gf_avg'] + home_form['ga_avg']) / 2.0
        top_label = prediction.get('prediction_label', 'Ничья')
        score_candidates = self._rudy_scoreline_candidates(home_expected, away_expected, top_label)
        motivation = self._build_motivation_context(fixture)

        probability_summary = (
            f"Х: {p_home * 100:.0f}% | Н: {p_draw * 100:.0f}% | Г: {p_away * 100:.0f}%"
        )
        motivation_summary = (
            f"Мотивация: {home_team} (#{motivation['home_rank'] or 'Н/Д'}) — {motivation['home_label']}; "
            f"{away_team} (#{motivation['away_rank'] or 'Н/Д'}) — {motivation['away_label']}."
        )
        if top_label == 'Победа хозяев':
            verdict = (
                f"Победа хозяев: {score_candidates[0]} или {score_candidates[1]}\n"
                f"{probability_summary}\n"
                f"{motivation_summary}"
            )
        elif top_label == 'Победа гостей':
            verdict = (
                f"Победа гостей: {score_candidates[0]} или {score_candidates[1]}\n"
                f"{probability_summary}\n"
                f"{motivation_summary}"
            )
        else:
            verdict = (
                f"Ничья: {score_candidates[0]} или {score_candidates[1]}\n"
                f"{probability_summary}\n"
                f"{motivation_summary}"
            )

        h2h_summary = "нет свежих очных встреч в базе"
        if h2h_matches:
            home_h2h_wins = 0
            away_h2h_wins = 0
            draws_h2h = 0
            for match in h2h_matches:
                hg = int(match.get('home_goals', 0) or 0)
                ag = int(match.get('away_goals', 0) or 0)
                match_home = match.get('home_team')
                if hg == ag:
                    draws_h2h += 1
                elif (match_home == home_team and hg > ag) or (match_home != home_team and ag > hg):
                    home_h2h_wins += 1
                else:
                    away_h2h_wins += 1
            h2h_summary = (
                f"Очные ({len(h2h_matches)}): {home_team} {home_h2h_wins}П, "
                f"{away_team} {away_h2h_wins}П, ничьи {draws_h2h}."
            )

        commentary = "\n".join([
            f"{home_team} дома: {home_form['wins']}П {home_form['draws']}Н {home_form['losses']}П.",
            f"{away_team} в гостях: {away_form['wins']}П {away_form['draws']}Н {away_form['losses']}П.",
            h2h_summary,
            f"Турнирная мотивация: {motivation['edge']}.",
            f"Лига: {league_name}.",
            f"Наиболее вероятные счеты: {score_candidates[0]}, {score_candidates[1]}",
        ])

        return {
            'Дома': home_form['text'],
            'В гостях': away_form['text'],
            'Вывод': verdict,
            'Комментарий': commentary,
        }

    def _build_rudy_super_narrative(self, fixture: dict, prediction: dict) -> dict:
        """Построить нарратив для RudySuper на основе агрегированного предсказания."""
        context = prediction.get('rudy_context', {}) or {}
        home_team = context.get('home_team') or fixture.get('teams', {}).get('home', {}).get('name', 'Хозяева')
        away_team = context.get('away_team') or fixture.get('teams', {}).get('away', {}).get('name', 'Гости')
        agreement_level = prediction.get('agreement_level', 'Н/Д')
        individual_preds = prediction.get('individual_predictions', {})

        probabilities = prediction.get('probabilities', {})
        p_home = float(self.get_probability_value(probabilities, 'Победа хозяев'))
        p_draw = float(self.get_probability_value(probabilities, 'Ничья'))
        p_away = float(self.get_probability_value(probabilities, 'Победа гостей'))

        top_label = prediction.get('prediction_label', 'Ничья')
        probability_summary = (
            f"Х: {p_home * 100:.0f}% | Н: {p_draw * 100:.0f}% | Г: {p_away * 100:.0f}%"
        )

        # Информация об индивидуальных предсказаниях
        individual_summary = (
            f"Rudy: {individual_preds.get('rudy', 'Н/Д')} | "
            f"Rudy2: {individual_preds.get('rudy2', 'Н/Д')} | "
            f"Rudy3: {individual_preds.get('rudy3', 'Н/Д')}"
        )

        if top_label == 'Победа хозяев':
            verdict = (
                f"RudySuper: Победа хозяев ({agreement_level})\n"
                f"{probability_summary}\n"
                f"Мнения: {individual_summary}"
            )
        elif top_label == 'Победа гостей':
            verdict = (
                f"RudySuper: Победа гостей ({agreement_level})\n"
                f"{probability_summary}\n"
                f"Мнения: {individual_summary}"
            )
        else:
            verdict = (
                f"RudySuper: Ничья ({agreement_level})\n"
                f"{probability_summary}\n"
                f"Мнения: {individual_summary}"
            )

        return {
            'Вывод': verdict,
        }

    def build_rudy_today_rows(self, fixtures: list[dict]) -> list[dict]:
        """Подготовить таблицу Rudy для всех сегодняшних матчей."""
        rows = []
        for fixture in fixtures:
            if not self.is_supported_fixture(fixture):
                continue

            prediction = self.predict_fixture(fixture, model_key='rudy', skip_enrichment=True)
            if prediction is None:
                continue

            narrative = self._build_rudy_narrative(fixture, prediction)
            
            # Получаем RudySuper предсказание
            prediction_super = self._predict_with_rudy_super(fixture)
            narrative_super = self._build_rudy_super_narrative(fixture, prediction_super)
            
            teams = fixture.get('teams', {})
            league = fixture.get('league', {})
            status = fixture.get('fixture', {}).get('status', {})
            match_time = fixture.get('fixture', {}).get('date', '')[:16]

            rows.append({
                'Время': match_time,
                'Матч': f"{teams.get('home', {}).get('name', 'Н/Д')} vs {teams.get('away', {}).get('name', 'Н/Д')}",
                'Страна': league.get('country', 'Н/Д'),
                'Дома': narrative['Дома'],
                'В гостях': narrative['В гостях'],
                'Вывод': narrative['Вывод'],
                'RudySuper': narrative_super['Вывод'],
                'Комментарий': narrative['Комментарий'],
            })

        return rows

    def _get_seasons_for_fixture(self, fixture: dict) -> list[int]:
        """Получить сезоны для подготовки исторического контекста."""
        fixture_season = fixture.get("league", {}).get("season")
        if self.trainer is not None:
            seasons = self.trainer._get_active_seasons(TRAINING_DATA_SEASONS)
        else:
            from config.settings import get_current_football_season
            current = get_current_football_season()
            seasons = list(range(current - TRAINING_DATA_SEASONS + 1, current + 1))

        if fixture_season and fixture_season not in seasons:
            seasons = [fixture_season] + seasons

        return seasons

    def _load_history_df(self, fixture: dict) -> pd.DataFrame:
        """Загрузить исторические завершенные матчи для лиги."""
        league_id = fixture.get("league", {}).get("id")
        seasons = tuple(self._get_seasons_for_fixture(fixture))
        cache_key = (league_id, seasons)

        if cache_key in self._history_cache:
            return self._history_cache[cache_key].copy()

        history_fixtures = []
        for season in seasons:
            fixtures = self.cache.get_fixtures_by_league_season(league_id, season)
            if not fixtures:
                fixtures = self.api_client.get_fixtures(league=league_id, season=season)
                for item in fixtures:
                    self.cache.save_fixture(item)
            history_fixtures.extend(fixtures)

        history_df = self.processor.parse_fixtures_to_dataframe(history_fixtures)
        if history_df.empty:
            self._history_cache[cache_key] = history_df
            return history_df.copy()

        history_df = self.processor.filter_finished_matches(history_df)
        history_df = self.processor.remove_duplicates(history_df)

        base_required_columns = [
            'fixture_id',
            'date',
            'league_id',
            'season',
            'home_team_id',
            'away_team_id',
            'home_goals',
            'away_goals',
            'status',
        ]
        available_required_columns = [column for column in base_required_columns if column in history_df.columns]
        if available_required_columns:
            history_df = history_df.dropna(subset=available_required_columns).copy()

        optional_columns = ["home_odds", "draw_odds", "away_odds"]
        empty_optional_columns = [
            column for column in optional_columns
            if column in history_df.columns and history_df[column].isna().all()
        ]
        if empty_optional_columns:
            history_df = history_df.drop(columns=empty_optional_columns)

        history_df = self.processor.create_target_variable(history_df)
        history_df = history_df.sort_values("date").reset_index(drop=True)

        self._history_cache[cache_key] = history_df
        return history_df.copy()

    def _get_standings_map(self, fixture: dict) -> dict:
        """Построить словарь рангов команд для лиги и сезона."""
        league_id = fixture.get("league", {}).get("id")
        season = fixture.get("league", {}).get("season")
        cache_key = (league_id, season)

        if cache_key in self._standings_cache:
            return self._standings_cache[cache_key]

        standings_groups = self.api_client.get_standings(league_id, season)
        standings_map = {}
        for group in standings_groups:
            teams = group if isinstance(group, list) else group.get("group", [])
            for team in teams:
                team_id = team.get("team", {}).get("id")
                if team_id is not None:
                    standings_map[team_id] = {"rank": team.get("rank", 10)}

        self._standings_cache[cache_key] = standings_map
        return standings_map

    def _build_feature_row(self, fixture: dict, skip_enrichment: bool = False) -> pd.DataFrame | None:
        """Построить одну строку признаков для инференса."""
        if not skip_enrichment:
            fixture = self._enrich_fixture_for_inference(fixture)
        history_df = self._load_history_df(fixture)
        fixture_df = self.processor.parse_fixtures_to_dataframe([fixture])

        if history_df.empty or fixture_df.empty:
            return None

        common_columns = sorted(set(history_df.columns) | set(fixture_df.columns))
        history_aligned = history_df.reindex(columns=common_columns)
        fixture_aligned = fixture_df.reindex(columns=common_columns)

        valid_columns = [
            column for column in common_columns
            if not (history_aligned[column].isna().all() and fixture_aligned[column].isna().all())
        ]

        history_aligned = history_aligned[valid_columns]
        fixture_aligned = fixture_aligned[valid_columns]

        combined_df = history_aligned.reset_index(drop=True).copy()
        combined_df = combined_df.reindex(range(len(combined_df) + 1))
        next_index = len(combined_df) - 1

        for column in history_aligned.columns:
            value = fixture_aligned.iloc[0][column]
            if pd.isna(value):
                combined_df.at[next_index, column] = (
                    pd.NaT if pd.api.types.is_datetime64_any_dtype(history_aligned[column]) else np.nan
                )
            else:
                combined_df.at[next_index, column] = value
        standings_map = self._get_standings_map(fixture)

        feature_df = self.feature_engineer.create_feature_matrix(combined_df, standings_map)
        feature_df = self.feature_engineer.add_interaction_features(feature_df)
        feature_df = self.feature_engineer.add_temporal_features(feature_df)

        prediction_row = feature_df.tail(1).copy()
        defaults = {column: self.feature_defaults.get(column, 0.0) for column in self.training_columns}

        for column in self.training_columns:
            if column not in prediction_row.columns:
                prediction_row[column] = defaults[column]

        prediction_row = prediction_row[self.training_columns].fillna(defaults)
        return prediction_row

    def _normalize_row(self, feature_row: pd.DataFrame) -> pd.DataFrame:
        """Привести признаки к виду, который использовался при обучении."""
        return self.processor.normalize_features(feature_row, fit=False)

    @staticmethod
    def _build_in_play_ml_feature_row(fixture: dict, base_prediction: dict) -> pd.DataFrame:
        """Собрать feature row для обученной snapshot-based in-play модели."""
        statistics = fixture.get('statistics', {})
        base_probabilities = base_prediction.get('probabilities', {})
        status = fixture.get('fixture', {}).get('status', {})
        home_goals = float(fixture.get('goals', {}).get('home') or 0.0)
        away_goals = float(fixture.get('goals', {}).get('away') or 0.0)

        return pd.DataFrame([{
            'elapsed_minute': float(status.get('elapsed') or 0.0),
            'score_diff': home_goals - away_goals,
            'snapshot_home_goals': home_goals,
            'snapshot_away_goals': away_goals,
            'shots_on_target_diff': float(statistics.get('home', {}).get('shots_on_target') or 0.0) - float(statistics.get('away', {}).get('shots_on_target') or 0.0),
            'total_shots_diff': float(statistics.get('home', {}).get('total_shots') or 0.0) - float(statistics.get('away', {}).get('total_shots') or 0.0),
            'possession_diff': float(statistics.get('home', {}).get('possession') or 50.0) - float(statistics.get('away', {}).get('possession') or 50.0),
            'corners_diff': float(statistics.get('home', {}).get('corners') or 0.0) - float(statistics.get('away', {}).get('corners') or 0.0),
            'base_home_probability': PredictionService.get_probability_value(base_probabilities, 'Победа хозяев') or (1 / 3),
            'base_draw_probability': PredictionService.get_probability_value(base_probabilities, 'Ничья') or (1 / 3),
            'base_away_probability': PredictionService.get_probability_value(base_probabilities, 'Победа гостей') or (1 / 3),
            'base_confidence': float(base_prediction.get('confidence', 1 / 3)),
        }])

    def _predict_with_trained_in_play_model(self, fixture: dict, base_prediction: dict) -> dict | None:
        """Использовать обученную snapshot-based in-play модель, если она доступна."""
        if self.trained_in_play_snapshot_model is None:
            return None

        feature_row = self._build_in_play_ml_feature_row(fixture, base_prediction)
        probabilities = self.trained_in_play_snapshot_model.get_calibrated_probabilities(feature_row)[0]
        prediction_class = int(np.argmax(probabilities))
        sorted_probabilities = np.sort(np.asarray(probabilities, dtype=float))
        return {
            'model_key': 'in_play_snapshot',
            'model_label': 'In-Play Snapshot ML',
            'prediction_class': prediction_class,
            'prediction_label': self.CLASS_LABELS.get(prediction_class, 'Неизвестно'),
            'raw_prediction_label': self.CLASS_LABELS.get(prediction_class, 'Неизвестно'),
            'confidence': float(sorted_probabilities[-1]),
            'confidence_margin': float(sorted_probabilities[-1] - sorted_probabilities[-2]),
            'probabilities': self._build_probability_map(probabilities),
            'prediction_mode': 'in_play_ml',
            'base_prediction': base_prediction,
        }

    def _attach_live_model_status(self, prediction: dict, base_prediction: dict) -> dict:
        """Добавить в live prediction явный статус используемой live-модели для UI."""
        live_model_key = prediction.get('model_key')
        live_model_label = prediction.get('model_label', 'Неизвестно')
        prediction_mode = prediction.get('prediction_mode')

        if prediction_mode == 'in_play_ml':
            live_model_status = 'Snapshot ML активна'
        elif prediction_mode == 'in_play':
            live_model_status = 'Эвристический fallback активен'
        else:
            live_model_status = 'Только предматчевый режим'

        enriched_prediction = dict(prediction)
        enriched_prediction['live_model_key'] = live_model_key
        enriched_prediction['live_model_label'] = live_model_label
        enriched_prediction['live_model_status'] = live_model_status
        enriched_prediction['base_model_key'] = base_prediction.get('model_key')
        enriched_prediction['base_model_label'] = base_prediction.get('model_label')
        return enriched_prediction

    def _maybe_save_live_snapshot(self, fixture: dict, prediction: dict | None) -> None:
        """Сохранить live snapshot, если матч находится в рабочем диапазоне минут."""
        if not LIVE_SNAPSHOTS_ENABLED or prediction is None or not self.is_live_fixture(fixture):
            return

        elapsed = fixture.get('fixture', {}).get('status', {}).get('elapsed')
        if elapsed is None:
            return
        if elapsed < LIVE_SNAPSHOT_MIN_MINUTE or elapsed > LIVE_SNAPSHOT_MAX_MINUTE:
            return

        base_prediction = prediction.get('base_prediction', {}) if prediction else {}
        in_play_prediction = prediction if prediction.get('prediction_mode') == 'in_play' else {}
        self.cache.save_live_snapshot(
            fixture,
            base_prediction=base_prediction,
            in_play_prediction=in_play_prediction,
        )

    def is_live_fixture(self, fixture: dict) -> bool:
        """Определить, что матч уже идет и для него нужен in-play слой."""
        status_short = str(fixture.get("fixture", {}).get("status", {}).get("short", "")).upper()
        elapsed = fixture.get("fixture", {}).get("status", {}).get("elapsed")
        live_statuses = {"1H", "HT", "2H", "ET", "BT", "P", "INT", "LIVE"}
        return status_short in live_statuses or elapsed is not None

    def get_fixture_label(self, fixture: dict) -> str:
        """Собрать компактную подпись матча для UI."""
        home_name = fixture.get("teams", {}).get("home", {}).get("name", "Неизвестно")
        away_name = fixture.get("teams", {}).get("away", {}).get("name", "Неизвестно")
        league_name = fixture.get("league", {}).get("name", "Неизвестная лига")
        kickoff = fixture.get("fixture", {}).get("date", "")[:16]
        return f"{kickoff} | {league_name} | {home_name} vs {away_name}"

    def build_dashboard_prediction_rows(self, fixtures: list[dict]) -> list[dict]:
        """Подготовить готовые строки для Dashboard вне Streamlit-рендера с информацией о всех моделях."""
        rows = []
        
        for fixture in fixtures:
            if not self.is_supported_fixture(fixture):
                continue

            # Для dashboard используем cache-only путь без сетевого enrichment,
            # чтобы фоновая подготовка не зависала и быстро показывала матчи на сегодня.
            enriched = fixture
            home_team = enriched.get('teams', {}).get('home', {})
            away_team = enriched.get('teams', {}).get('away', {})
            league = enriched.get('league', {})
            
            # В Rudy-only режиме только один прогноз
            primary_pred = self.predict_fixture(
                fixture,
                model_key='rudy',
                skip_enrichment=True,
            )
            if not primary_pred:
                continue
            
            models_status = {}
            if primary_pred:
                model_key = primary_pred.get('model_key', 'rudy')
                has_prediction = bool(primary_pred.get('prediction_label'))
                is_abstained = bool(primary_pred.get('is_abstained', False))
                status = 'OK' if has_prediction and not is_abstained else 'SKIP'
                models_status[model_key] = {
                    'abbrev': self.MODEL_ABBREV.get(model_key, 'RUD'),
                    'status': status,
                    'label': primary_pred.get('prediction_label', '-'),
                    'confidence': primary_pred.get('confidence', 0.0),
                }
            
            rows.append({
                'Время': enriched.get('fixture', {}).get('date', '')[:16],
                'Хозяева': home_team.get('name', 'Н/Д'),
                'Гости': away_team.get('name', 'Н/Д'),
                'Страна': league.get('country', 'Н/Д'),
                'Модель': primary_pred.get('model_label', 'Н/Д'),
                'Коэффициенты 1/X/2': (
                    f"{enriched.get('odds', {}).get('1', '-')}/"
                    f"{enriched.get('odds', {}).get('X', '-')}/"
                    f"{enriched.get('odds', {}).get('2', '-')}"
                ),
                'Прогноз': primary_pred.get('prediction_label', 'Н/Д'),
                'Кандидат': primary_pred.get('raw_prediction_label', 'Н/Д') if primary_pred.get('is_abstained') else '-',
                'Уверенность': f"{primary_pred.get('confidence', 0.0) * 100:.1f}%",
                'models_status': models_status,
            })

        return rows

    def refresh_dashboard_prediction_cache(self, fixtures: list[dict]) -> dict:
        """Пересчитать и сохранить подготовленные строки Dashboard в cache."""
        rows = self.build_dashboard_prediction_rows(fixtures)
        payload = {
            'created_at': datetime.now().isoformat(),
            'row_count': len(rows),
            'rows': rows,
        }
        self.cache.set(
            self.DASHBOARD_CACHE_KEY,
            payload,
            ttl=DASHBOARD_PREDICTIONS_CACHE_TTL,
        )
        return payload

    def get_cached_dashboard_predictions(self) -> dict | None:
        """Прочитать заранее подготовленные Dashboard predictions из cache."""
        return self.cache.get(self.DASHBOARD_CACHE_KEY, default=None)

    def get_dashboard_prediction_state(self) -> dict:
        """Получить состояние фоновой подготовки dashboard predictions."""
        return self.cache.get(
            self.DASHBOARD_STATE_CACHE_KEY,
            default={
                'status': 'idle',
                'started_at': None,
                'finished_at': None,
                'last_error': None,
                'row_count': 0,
            },
        ) or {}

    def get_feature_diagnostics(
        self,
        fixture: dict,
        model_key: str | None = None,
        top_n: int = 15,
        skip_enrichment: bool = False,
    ) -> dict | None:
        """Вернуть диагностический срез признаков для одного матча."""
        model_key = model_key or self.primary_model_key
        if model_key in {'rudy', 'rudy2', 'rudy3'}:
            if model_key == 'rudy':
                prediction = self._predict_with_rudy(fixture)
            elif model_key == 'rudy2':
                prediction = self._predict_with_rudy2(fixture)
            else:
                prediction = self._predict_with_rudy3(fixture)
            empty_frame = pd.DataFrame(columns=["feature", "value", "normalized", "default", "delta_from_default"])
            empty_delta = pd.DataFrame(columns=["feature", "value", "default", "abs_delta"])
            return {
                "fixture_label": self.get_fixture_label(fixture),
                "prediction": prediction,
                "feature_count": 0,
                "focus_features": empty_frame,
                "top_feature_deltas": empty_delta,
                "raw_feature_row": pd.DataFrame(),
                "normalized_feature_row": pd.DataFrame(),
            }

        model = self.loaded_models.get(model_key)
        if model is None:
            return None

        if not skip_enrichment:
            fixture = self._enrich_fixture_for_inference(fixture)

        raw_feature_row = self._build_feature_row(fixture, skip_enrichment=skip_enrichment)
        if raw_feature_row is None:
            return None

        normalized_row = self._normalize_row(raw_feature_row)
        base_prediction = self._predict_with_model(model, model_key, normalized_row)
        active_feature_columns = model.feature_columns or self.training_columns

        diagnostic_raw_row = raw_feature_row.copy()
        diagnostic_normalized_row = normalized_row.copy()
        for column in active_feature_columns:
            default_value = float(self.feature_defaults.get(column, 0.0))
            if column not in diagnostic_raw_row.columns:
                diagnostic_raw_row[column] = default_value
            if column not in diagnostic_normalized_row.columns:
                diagnostic_normalized_row[column] = default_value

        diagnostic_raw_row = diagnostic_raw_row[active_feature_columns]
        diagnostic_normalized_row = diagnostic_normalized_row[active_feature_columns]

        focus_columns = [
            column for column in FEATURE_DIAGNOSTIC_COLUMNS
            if column in diagnostic_raw_row.columns and column in active_feature_columns
        ]

        focus_frame = pd.DataFrame({
            "feature": focus_columns,
            "value": [float(diagnostic_raw_row.iloc[0][column]) for column in focus_columns],
            "normalized": [float(diagnostic_normalized_row.iloc[0][column]) for column in focus_columns],
            "default": [float(self.feature_defaults.get(column, 0.0)) for column in focus_columns],
        })
        focus_frame["delta_from_default"] = focus_frame["value"] - focus_frame["default"]

        delta_frame = pd.DataFrame({
            "feature": active_feature_columns,
            "value": [float(diagnostic_raw_row.iloc[0][column]) for column in active_feature_columns],
            "default": [float(self.feature_defaults.get(column, 0.0)) for column in active_feature_columns],
        })
        delta_frame["abs_delta"] = (delta_frame["value"] - delta_frame["default"]).abs()
        top_delta_frame = delta_frame.sort_values("abs_delta", ascending=False).head(top_n).reset_index(drop=True)

        return {
            "fixture_label": self.get_fixture_label(fixture),
            "prediction": base_prediction,
            "feature_count": len(active_feature_columns),
            "focus_features": focus_frame,
            "top_feature_deltas": top_delta_frame,
            "raw_feature_row": diagnostic_raw_row,
            "normalized_feature_row": diagnostic_normalized_row,
        }

    def predict_fixture(
        self,
        fixture: dict,
        model_key: str | None = None,
        skip_enrichment: bool = False,
    ) -> dict | None:
        """Предсказать исход матча выбранной моделью."""
        model_key = model_key or self.primary_model_key
        if model_key == 'rudy':
            return self._predict_with_rudy(fixture)
        if model_key == 'rudy2':
            return self._predict_with_rudy2(fixture)
        if model_key == 'rudy3':
            return self._predict_with_rudy3(fixture)
        return None

    def predict_live_fixture(self, fixture: dict, model_key: str | None = None) -> dict | None:
        """Предсказать live матч с использованием snapshot ML или base model."""
        model_key = model_key or self.primary_model_key
        enriched_fixture = self._enrich_fixture_for_inference(fixture)
        base_prediction = self.predict_fixture(
            enriched_fixture,
            model_key=model_key,
            skip_enrichment=True,
        )
        if base_prediction is None:
            return None

        trained_prediction = self._predict_with_trained_in_play_model(enriched_fixture, base_prediction)
        if trained_prediction is not None:
            return self._attach_live_model_status(trained_prediction, base_prediction)

        # Без in-play коррекции, просто возвращаем base_prediction для live матчей
        return self._attach_live_model_status(
            base_prediction,
            base_prediction,
        )



    def enrich_fixtures(
        self,
        fixtures: list[dict],
        model_key: str | None = None,
        limit: int = 10,
        prefer_in_play: bool | None = None,
    ) -> list[dict]:
        """Обогатить список матчей предсказаниями."""
        results = []
        for fixture in fixtures:
            if not self.is_supported_fixture(fixture):
                continue

            enriched_fixture = self._enrich_fixture_for_inference(fixture)
            use_in_play_model = self.is_live_fixture(enriched_fixture) if prefer_in_play is None else prefer_in_play
            if use_in_play_model:
                prediction = self.predict_live_fixture(enriched_fixture, model_key=model_key)
            else:
                prediction = self.predict_fixture(
                    enriched_fixture,
                    model_key=model_key,
                    skip_enrichment=True,
                )

            self._maybe_save_live_snapshot(enriched_fixture, prediction)

            results.append({
                "fixture": enriched_fixture,
                "prediction": prediction,
            })

            if len(results) >= limit:
                break

        return results

    def is_supported_fixture(self, fixture: dict) -> bool:
        """Проверить, что матч можно безопасно обрабатывать в UI."""
        league_id = fixture.get("league", {}).get("id")
        if league_id in TRACKED_LEAGUES:
            return True

        if self.is_live_fixture(fixture):
            return True

        return False

    def get_reference_fixture(self, fixtures: list[dict]) -> dict | None:
        """Выбрать первый подходящий матч для сравнения моделей."""
        return next((fixture for fixture in fixtures if self.is_supported_fixture(fixture)), None)

    def get_model_metrics(self) -> list[dict]:
        """Получить сохраненные метрики загруженных моделей."""
        def _metric_row(model_key: str, model: RudyModel) -> dict:
            return {
                "model_key": model_key,
                "model_label": self.MODEL_LABELS.get(model_key, model_key),
                "accuracy": float(model.metrics.get("accuracy", 0.0)),
                "precision": float(model.metrics.get("precision", 0.0)),
                "recall": float(model.metrics.get("recall", 0.0)),
                "f1": float(model.metrics.get("f1", 0.0)),
                "roc_auc": float(model.metrics.get("roc_auc", 0.0)),
                "brier_score": float(model.metrics.get("brier_score", 0.0)),
                "expected_calibration_error": float(model.metrics.get("expected_calibration_error", 0.0)),
                "calibration_method": model.metrics.get("calibration_method", "rule_based"),
                "calibration_brier_before": float(model.metrics.get("calibration_brier_before", 0.0)),
                "calibration_brier_after": float(model.metrics.get("calibration_brier_after", 0.0)),
                "calibration_brier_improvement": float(model.metrics.get("calibration_brier_improvement", 0.0)),
                "is_probability_calibrated": bool(model.metrics.get("is_probability_calibrated", False)),
                "reliability_curve": model.metrics.get("reliability_curve", []),
                "active_feature_groups": model.metrics.get("active_feature_groups", ["rudy_form", "rudy_h2h"]),
                "feature_count": int(model.metrics.get("feature_count", 0)),
                "prediction_policy_enabled": bool(model.metrics.get("prediction_policy_enabled", False)),
                "prediction_policy_confidence_threshold": float(model.metrics.get("prediction_policy_confidence_threshold", 0.0)),
                "prediction_policy_margin_threshold": float(model.metrics.get("prediction_policy_margin_threshold", 0.0)),
                "prediction_policy_coverage": float(model.metrics.get("prediction_policy_coverage", 1.0)),
                "prediction_policy_selective_accuracy": float(model.metrics.get("prediction_policy_selective_accuracy", 0.0)),
            }

        return [
            _metric_row("rudy", self.rudy_model),
            _metric_row("rudy2", self.rudy2_model),
            _metric_row("rudy3", self.rudy3_model),
        ]

    def get_primary_model_metrics(self) -> dict:
        """Метрики основной модели для dashboard."""
        return {
            "model_label": self.MODEL_LABELS.get("rudy", "Rudy"),
            "accuracy": float(self.rudy_model.metrics.get("accuracy", 0.0)),
            "precision": float(self.rudy_model.metrics.get("precision", 0.0)),
            "recall": float(self.rudy_model.metrics.get("recall", 0.0)),
            "f1": float(self.rudy_model.metrics.get("f1", 0.0)),
            "roc_auc": float(self.rudy_model.metrics.get("roc_auc", 0.0)),
        }

    @staticmethod
    def _selected_outcome_odds(raw_features: pd.DataFrame, predictions: np.ndarray) -> np.ndarray:
        odds_matrix = raw_features[['home_odds', 'draw_odds', 'away_odds']].to_numpy(dtype=float)
        row_index = np.arange(len(raw_features))
        return odds_matrix[row_index, predictions]

    @staticmethod
    def _has_real_outcome_odds(raw_features: pd.DataFrame) -> bool:
        required_columns = {'home_odds', 'draw_odds', 'away_odds'}
        return required_columns.issubset(raw_features.columns)

    @staticmethod
    def _policy_bet_mask(model: BaseModel, probabilities: np.ndarray) -> np.ndarray:
        if probabilities.size == 0:
            return np.array([], dtype=bool)

        mask = np.ones(len(probabilities), dtype=bool)
        policy = model.metrics.get('prediction_policy', {}) or {}
        if not bool(policy.get('abstain_enabled', False)):
            return mask

        confidence = np.max(probabilities, axis=1)
        sorted_probabilities = np.sort(probabilities, axis=1)
        margin = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
        confidence_threshold = max(
            float(policy.get('confidence_threshold', 0.0)),
            float(PREDICTION_POLICY_CONFIDENCE_FLOOR),
        )
        margin_threshold = max(
            float(policy.get('margin_threshold', 0.0)),
            float(PREDICTION_POLICY_MARGIN_FLOOR),
        )
        return (confidence >= confidence_threshold) & (margin >= margin_threshold)

    def get_roi_analysis(self, force_refresh: bool = False) -> dict:
        """Рассчитать ROI-анализ с явным разрезом: holdout и полный исторический прогон."""
        del force_refresh
        if self._roi_analysis_cache is not None:
            return self._roi_analysis_cache
        self._roi_analysis_cache = {
            'summary': {
                'total_bets': 0,
                'winning_bets': 0,
                'overall_roi': None,
                'best_model': None,
            },
            'rows': [],
            'rows_holdout': [],
            'warning': 'ROI-анализ отключен: в проекте оставлена только Rudy-модель без обучаемых моделей.',
        }
        return self._roi_analysis_cache

    def get_rudy_super_accuracy_stats(self, limit: int = 50) -> dict:
        """Подсчитать статистику точности на завершённых матчах."""
        stats = {
            'total_matches': 0,
            'p1_correct': 0,
            'p1_total': 0,
            'draw_correct': 0,
            'draw_total': 0,
            'p2_correct': 0,
            'p2_total': 0,
        }
        
        try:
            finished_fixtures = []
            
            # Получаем live матчи через API
            try:
                live_fixtures = self.api_client.get_live_fixtures()
                if live_fixtures:
                    for fixture in live_fixtures:
                        status = fixture.get('fixture', {}).get('status', {}).get('short', '')
                        if status in ['FT', 'AET', 'PEN']:
                            if self.is_supported_fixture(fixture):
                                finished_fixtures.append(fixture)
            except Exception as e:
                logger.debug(f"Ошибка live: {e}")
            
            # Добавляем матчи из нескольких лиг
            if len(finished_fixtures) < limit:
                league_list = list(TRACKED_LEAGUES)[:3] if hasattr(TRACKED_LEAGUES, '__iter__') else list(TRACKED_LEAGUES.values())[:3]
                for league_id in league_list:
                    if len(finished_fixtures) >= limit:
                        break
                    try:
                        fixtures = self.api_client.get_fixtures(league=league_id, season=2026)
                        if fixtures:
                            for fixture in fixtures:
                                if len(finished_fixtures) >= limit:
                                    break
                                status = fixture.get('fixture', {}).get('status', {}).get('short', '')
                                if status in ['FT', 'AET', 'PEN']:
                                    if self.is_supported_fixture(fixture):
                                        fid = fixture.get('fixture', {}).get('id')
                                        if fid not in [f.get('fixture', {}).get('id') for f in finished_fixtures]:
                                            finished_fixtures.append(fixture)
                    except Exception as e:
                        logger.debug(f"Ошибка лига {league_id}: {e}")
            
            # Обрабатываем матчи
            for fixture in finished_fixtures[:limit]:
                try:
                    pred = self._predict_with_rudy(fixture)
                    pred_label = pred.get('prediction_label', 'Неизвестно')
                    
                    goals = fixture.get('goals', {})
                    h = int(goals.get('home') or 0)
                    a = int(goals.get('away') or 0)
                    
                    actual = 'Победа хозяев' if h > a else ('Победа гостей' if a > h else 'Ничья')
                    
                    stats['total_matches'] += 1
                    if actual == 'Победа хозяев':
                        stats['p1_total'] += 1
                        if pred_label == actual:
                            stats['p1_correct'] += 1
                    elif actual == 'Ничья':
                        stats['draw_total'] += 1
                        if pred_label == actual:
                            stats['draw_correct'] += 1
                    else:
                        stats['p2_total'] += 1
                        if pred_label == actual:
                            stats['p2_correct'] += 1
                except Exception as e:
                    logger.debug(f"Матч ошибка: {e}")
        
        except Exception as e:
            logger.error(f"Статистика ошибка: {e}")
        
        return stats

    def get_live_snapshot_stats(self) -> dict:
        """Быстрая статистика по накопленным live snapshots."""
        rows = self.cache.get_live_snapshot_training_rows(only_finished=False)
        labeled_rows = [row for row in rows if row.get('result') is not None]
        backfill_status = self.cache.get_backfill_queue_status()
        backfill_failed_breakdown = self.cache.get_backfill_failed_breakdown()
        readiness_ratio = min(len(labeled_rows) / max(IN_PLAY_ML_MIN_ROWS, 1), 1.0)
        return {
            'snapshot_count': self.cache.get_live_snapshot_count(),
            'labeled_snapshot_count': len(labeled_rows),
            'required_labeled_snapshots': IN_PLAY_ML_MIN_ROWS,
            'readiness_ratio': readiness_ratio,
            'trained_in_play_model': self.trained_in_play_snapshot_model is not None,
            'backfill_queue': backfill_status,
            'backfill_failed_breakdown': backfill_failed_breakdown,
        }

    def accumulate_rudy_super_stats(self) -> dict:
        """Загрузить завершённые матчи из API и кэша, предсказать RudySuper, сохранить в БД.

        Учитываются только матчи из поддерживаемых лиг (TRACKED_LEAGUES), дата которых
        >= 2026-04-23 (с момента запуска системы отслеживания). Загружает матчи за весь
        диапазон от TRACKING_START_DATE до текущей даты, сохраняет их, затем обрабатывает.
        """
        from config.settings import get_current_football_season
        from datetime import datetime, date
        
        TRACKING_START_DATE = '2026-04-23'
        today_str = str(date.today())  # Получаем текущую дату в формате YYYY-MM-DD
        
        already_processed = self.cache.get_rudy_super_processed_fixture_ids()
        added = 0
        errors = 0
        league_ids = list(TRACKED_LEAGUES) if isinstance(TRACKED_LEAGUES, (set, dict)) else list(TRACKED_LEAGUES)
        current_season = get_current_football_season()
        seasons = [current_season, current_season - 1]

        # Сначала загружаем матчи за весь диапазон дат прямо из API
        # Это гарантирует, что все свежие (FT) матчи от 2026-04-23 до сегодня будут загружены
        all_fixtures_to_process = []
        for league_id in league_ids:
            try:
                # Загружаем матчи за ДИАПАЗОН дат с фильтром по статусу FT
                # Это позволит ловить матчи за 24.04, 25.04 и т.д.
                api_fixtures = self.api_client.get_fixtures_range(
                    league=league_id,
                    season=current_season,
                    date_from=TRACKING_START_DATE,
                    date_to=today_str  # Загружаем до текущей даты
                )
                if api_fixtures:
                    # Сохраняем загруженные матчи в кэш-БД
                    for fixture in api_fixtures:
                        self.cache.save_fixture(fixture)
                        all_fixtures_to_process.append(fixture)
            except Exception as e:
                logger.debug(f"Ошибка загрузки матчей {league_id} из API за диапазон {TRACKING_START_DATE} - {today_str}: {e}")
                continue

        # Затем получаем все матчи из БД (включая только что загруженные)
        for league_id in league_ids:
            for season in seasons:
                try:
                    db_fixtures = self.cache.get_fixtures_by_league_season(league_id, season)
                    all_fixtures_to_process.extend(db_fixtures)
                except Exception:
                    continue

        # Обработка матчей (удаляем дубликаты по fixture_id)
        processed_fixture_ids = set()
        for fixture in all_fixtures_to_process:
            try:
                fid = fixture.get('fixture', {}).get('id')
                if fid is None or fid in already_processed or fid in processed_fixture_ids:
                    continue

                # Фильтр по дате — только матчи с 23.04.2026 или позже
                match_date = fixture.get('fixture', {}).get('date', '')[:10]
                if match_date < TRACKING_START_DATE:
                    continue

                status = fixture.get('fixture', {}).get('status', {}).get('short', '')
                if status not in ('FT', 'AET', 'PEN'):
                    continue

                goals = fixture.get('goals', {})
                h = goals.get('home')
                a = goals.get('away')
                if h is None or a is None:
                    continue
                h, a = int(h), int(a)
                actual_label = 'Победа хозяев' if h > a else ('Победа гостей' if a > h else 'Ничья')

                pred = self._predict_with_rudy_super(fixture)
                probs = pred.get('probabilities', {})
                label_map = {'home_win': 'Победа хозяев', 'draw': 'Ничья', 'away_win': 'Победа гостей'}
                predicted_label = max(probs, key=probs.get) if probs else None
                if predicted_label is None:
                    continue
                predicted_label = label_map.get(predicted_label, predicted_label)

                home_team = fixture.get('teams', {}).get('home', {}).get('name', '')
                away_team = fixture.get('teams', {}).get('away', {}).get('name', '')
                league_id = fixture.get('league', {}).get('id', 0)
                season = fixture.get('league', {}).get('season', current_season)

                self.cache.save_rudy_super_stat(
                    fid, league_id, season, match_date,
                    home_team, away_team, predicted_label, actual_label
                )
                processed_fixture_ids.add(fid)
                added += 1
            except Exception as e:
                logger.debug(f"Ошибка обработки матча {fid}: {e}")
                errors += 1
                continue

        summary = self.cache.get_rudy_super_stats_summary()
        return {'added': added, 'errors': errors, 'summary': summary}
