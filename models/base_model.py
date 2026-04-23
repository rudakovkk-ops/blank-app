"""
Базовый класс для всех моделей прогнозирования
"""
import pickle
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from config.settings import (
    MODELS_DIR,
    PREDICTION_POLICY_MIN_ACCURACY_GAIN,
    PREDICTION_POLICY_MAX_COVERAGE,
    PREDICTION_POLICY_CONFIDENCE_FLOOR,
    PREDICTION_POLICY_MARGIN_FLOOR,
    PREDICTION_POLICY_COVERAGE_PENALTY,
    CALIBRATION_MIN_BRIER_IMPROVEMENT,
    CALIBRATION_MAX_ECE_DEGRADATION,
)

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Базовый класс для всех моделей"""
    
    def __init__(self, name: str, model_type: str):
        """
        Инициализировать модель
        
        Args:
            name: Название модели
            model_type: Тип модели (e.g., 'logistic_regression')
        """
        self.name = name
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.training_date = None
        self.metrics = {}
        self.probability_calibrator = None
        self.is_probability_calibrated = False
        self.feature_columns = []
        self.active_feature_groups = []
        self.prediction_policy = {}
        
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> bool:
        """
        Обучить модель
        
        Args:
            X_train: Тренировочные признаки
            y_train: Целевая переменная (тренировка)
        
        Returns:
            True если успешно
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Сделать предсказание
        
        Args:
            X: Признаки для предсказания
        
        Returns:
            Массив с предсказаниями
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Получить вероятности классов
        
        Args:
            X: Признаки
        
        Returns:
            Массив с вероятностями
        """
        pass
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Оценить модель на тестовых данных
        
        Args:
            X_test: Тестовые признаки
            y_test: Реальные значения
            verbose: Выводить ли результаты
        
        Returns:
            Словарь с метриками
        """
        y_proba = self.get_calibrated_probabilities(X_test)
        y_pred = np.argmax(y_proba, axis=1)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }
        
        try:
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        except Exception:
            metrics['roc_auc'] = 0.0
        metrics['brier_score'] = self._multiclass_brier_score(y_test, y_proba)
        metrics['expected_calibration_error'] = self._expected_calibration_error(y_test, y_proba)
        metrics['reliability_curve'] = self._build_reliability_curve(y_test, y_proba)
        metrics['is_probability_calibrated'] = bool(getattr(self, 'is_probability_calibrated', False))
        
        preserved_metrics = {
            key: value
            for key, value in self.metrics.items()
            if key not in metrics
        }
        self.metrics = {**preserved_metrics, **metrics}
        
        if verbose:
            logger.info(f"Model: {self.name}")
            for metric_name, metric_value in metrics.items():
                if np.isscalar(metric_value):
                    logger.info(f"  {metric_name}: {float(metric_value):.4f}")
                else:
                    logger.info(f"  {metric_name}: {metric_value}")
        
        return metrics
    
    def save(self, filepath: Path = None) -> bool:
        """
        Сохранить модель в файл
        
        Args:
            filepath: Путь для сохранения (по умолчанию models/)
        
        Returns:
            True если успешно
        """
        try:
            if filepath is None:
                filepath = MODELS_DIR / f"{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    @staticmethod
    def load(filepath: Path) -> 'BaseModel':
        """
        Загрузить модель из файла
        
        Args:
            filepath: Путь к файлу модели
        
        Returns:
            Загруженная модель
        """
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Получить важность признаков (если поддерживается)
        
        Returns:
            Словарь с важностью признаков
        """
        return {}

    def set_feature_context(self, feature_columns: list[str], active_feature_groups: list[str] | tuple[str, ...] | None = None):
        """Запомнить колонки, на которых модель обучалась, и активные группы признаков."""
        self.feature_columns = list(feature_columns or [])
        self.active_feature_groups = list(active_feature_groups or [])

    def prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Выровнять вход под модельно-специфичный набор колонок."""
        if not self.feature_columns:
            return X

        prepared = X.copy()
        for column in self.feature_columns:
            if column not in prepared.columns:
                prepared[column] = 0.0
        return prepared[self.feature_columns]

    def calibrate_probabilities(self, X_calibration: pd.DataFrame, y_calibration: pd.Series) -> bool:
        """Обучить post-hoc calibrator на validation split по сырым вероятностям модели."""
        try:
            prepared_features = self.prepare_features(X_calibration)
            raw_probabilities = np.asarray(self.predict_proba(prepared_features), dtype=float)
            if raw_probabilities.size == 0:
                return False

            unique_classes = np.unique(y_calibration)
            if len(unique_classes) < 2:
                logger.warning("Skipping probability calibration for %s: not enough classes", self.name)
                return False

            clipped_probabilities = np.clip(raw_probabilities, 1e-6, 1.0)
            calibrator = LogisticRegression(
                max_iter=1000,
                multi_class='multinomial',
                random_state=42,
            )
            calibrator.fit(clipped_probabilities, y_calibration)
            calibrated_probabilities = self._normalize_probabilities(
                calibrator.predict_proba(clipped_probabilities)
            )

            raw_brier = self._multiclass_brier_score(y_calibration, raw_probabilities)
            calibrated_brier = self._multiclass_brier_score(y_calibration, calibrated_probabilities)
            raw_ece = self._expected_calibration_error(y_calibration, raw_probabilities)
            calibrated_ece = self._expected_calibration_error(y_calibration, calibrated_probabilities)

            brier_improvement = raw_brier - calibrated_brier
            ece_delta = calibrated_ece - raw_ece

            keep_calibrator = (
                brier_improvement >= CALIBRATION_MIN_BRIER_IMPROVEMENT
                and ece_delta <= CALIBRATION_MAX_ECE_DEGRADATION
            )

            self.probability_calibrator = calibrator if keep_calibrator else None
            self.is_probability_calibrated = keep_calibrator
            self.metrics['calibration_method'] = (
                'multinomial_logistic_calibration'
                if keep_calibrator else
                'none'
            )
            self.metrics['calibration_rows'] = int(len(X_calibration))
            self.metrics['calibration_brier_before'] = raw_brier
            self.metrics['calibration_brier_after'] = calibrated_brier if keep_calibrator else raw_brier
            self.metrics['calibration_brier_improvement'] = (
                raw_brier - calibrated_brier if keep_calibrator else 0.0
            )
            self.metrics['calibration_ece_before'] = raw_ece
            self.metrics['calibration_ece_after'] = calibrated_ece if keep_calibrator else raw_ece
            self.metrics['calibration_ece_delta'] = ece_delta if keep_calibrator else 0.0
            self.metrics['calibration_brier_required_improvement'] = CALIBRATION_MIN_BRIER_IMPROVEMENT
            self.metrics['calibration_max_ece_degradation'] = CALIBRATION_MAX_ECE_DEGRADATION
            self.metrics['calibration_selected'] = bool(keep_calibrator)
            return keep_calibrator
        except Exception as e:
            logger.error("Error calibrating probabilities for %s: %s", self.name, e)
            return False

    def optimize_prediction_policy(
        self,
        X_policy: pd.DataFrame,
        y_policy: pd.Series,
        min_coverage: float = 0.55,
    ) -> dict:
        """Подобрать selective prediction policy по confidence и margin."""
        probabilities = self.get_calibrated_probabilities(X_policy)
        predicted = np.argmax(probabilities, axis=1)
        confidence = np.max(probabilities, axis=1)
        sorted_probabilities = np.sort(probabilities, axis=1)
        margin = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
        baseline_accuracy = float(accuracy_score(y_policy, predicted))

        confidence_candidates = sorted({
            PREDICTION_POLICY_CONFIDENCE_FLOOR,
            max(PREDICTION_POLICY_CONFIDENCE_FLOOR, 0.60),
            max(PREDICTION_POLICY_CONFIDENCE_FLOOR, 0.65),
            max(PREDICTION_POLICY_CONFIDENCE_FLOOR, 0.70),
            max(PREDICTION_POLICY_CONFIDENCE_FLOOR, 0.75),
            *np.round(np.quantile(confidence, [0.1, 0.25, 0.5, 0.75, 0.9]), 3).tolist(),
        })
        confidence_candidates = [
            candidate
            for candidate in confidence_candidates
            if candidate >= PREDICTION_POLICY_CONFIDENCE_FLOOR
        ]
        margin_candidates = sorted({
            PREDICTION_POLICY_MARGIN_FLOOR,
            max(PREDICTION_POLICY_MARGIN_FLOOR, 0.12),
            max(PREDICTION_POLICY_MARGIN_FLOOR, 0.14),
            max(PREDICTION_POLICY_MARGIN_FLOOR, 0.16),
            max(PREDICTION_POLICY_MARGIN_FLOOR, 0.18),
            *np.round(np.quantile(margin, [0.1, 0.25, 0.5, 0.75, 0.9]), 3).tolist(),
        })
        margin_candidates = [
            candidate
            for candidate in margin_candidates
            if candidate >= PREDICTION_POLICY_MARGIN_FLOOR
        ]

        best_policy = {
            'confidence_threshold': float(PREDICTION_POLICY_CONFIDENCE_FLOOR),
            'margin_threshold': float(PREDICTION_POLICY_MARGIN_FLOOR),
            'coverage': 1.0,
            'selective_accuracy': baseline_accuracy,
            'baseline_accuracy': baseline_accuracy,
            'abstain_enabled': True,
        }
        best_score = baseline_accuracy

        y_policy_array = np.asarray(y_policy)
        for confidence_threshold in confidence_candidates:
            for margin_threshold in margin_candidates:
                mask = (confidence >= confidence_threshold) & (margin >= margin_threshold)
                coverage = float(np.mean(mask))
                if (
                    not np.any(mask)
                    or coverage < min_coverage
                    or coverage > PREDICTION_POLICY_MAX_COVERAGE
                ):
                    continue

                selective_accuracy = float(accuracy_score(y_policy_array[mask], predicted[mask]))
                score = selective_accuracy - PREDICTION_POLICY_COVERAGE_PENALTY * coverage
                if (
                    score > best_score + 1e-6
                    and selective_accuracy >= baseline_accuracy + PREDICTION_POLICY_MIN_ACCURACY_GAIN
                ):
                    best_score = score
                    best_policy = {
                        'confidence_threshold': float(confidence_threshold),
                        'margin_threshold': float(margin_threshold),
                        'coverage': coverage,
                        'selective_accuracy': selective_accuracy,
                        'baseline_accuracy': baseline_accuracy,
                        'abstain_enabled': True,
                    }

        # Fallback: если gain не достигнут, сохраняем conservative policy с фиксированными floor-порогами.
        if best_policy.get('abstain_enabled', False) and best_policy['coverage'] >= 1.0:
            conservative_mask = (
                (confidence >= PREDICTION_POLICY_CONFIDENCE_FLOOR)
                & (margin >= PREDICTION_POLICY_MARGIN_FLOOR)
            )
            conservative_coverage = float(np.mean(conservative_mask))
            if np.any(conservative_mask):
                conservative_accuracy = float(accuracy_score(y_policy_array[conservative_mask], predicted[conservative_mask]))
            else:
                conservative_accuracy = baseline_accuracy
            best_policy = {
                'confidence_threshold': float(PREDICTION_POLICY_CONFIDENCE_FLOOR),
                'margin_threshold': float(PREDICTION_POLICY_MARGIN_FLOOR),
                'coverage': conservative_coverage,
                'selective_accuracy': conservative_accuracy,
                'baseline_accuracy': baseline_accuracy,
                'abstain_enabled': True,
            }

        self.prediction_policy = best_policy
        self.metrics['prediction_policy'] = best_policy
        self.metrics['prediction_policy_confidence_threshold'] = best_policy['confidence_threshold']
        self.metrics['prediction_policy_margin_threshold'] = best_policy['margin_threshold']
        self.metrics['prediction_policy_coverage'] = best_policy['coverage']
        self.metrics['prediction_policy_selective_accuracy'] = best_policy['selective_accuracy']
        self.metrics['prediction_policy_baseline_accuracy'] = best_policy['baseline_accuracy']
        self.metrics['prediction_policy_enabled'] = best_policy['abstain_enabled']
        return best_policy

    def get_calibrated_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        """Вернуть calibrated probabilities, если calibrator обучен; иначе raw probabilities."""
        prepared_features = self.prepare_features(X)
        probabilities = self._normalize_probabilities(np.asarray(self.predict_proba(prepared_features), dtype=float))
        calibrator = getattr(self, 'probability_calibrator', None)
        if calibrator is None:
            return probabilities

        calibrated_probabilities = calibrator.predict_proba(np.clip(probabilities, 1e-6, 1.0))
        return self._normalize_probabilities(np.asarray(calibrated_probabilities, dtype=float))

    @staticmethod
    def _normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
        row_sums = probabilities.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return probabilities / row_sums

    @staticmethod
    def _multiclass_brier_score(y_true: pd.Series, probabilities: np.ndarray) -> float:
        y_array = np.asarray(y_true, dtype=int)
        probabilities = BaseModel._normalize_probabilities(np.asarray(probabilities, dtype=float))
        n_classes = probabilities.shape[1]
        one_hot = np.eye(n_classes)[y_array]
        return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))

    @staticmethod
    def _build_reliability_curve(y_true: pd.Series, probabilities: np.ndarray, n_bins: int = 10) -> list[dict]:
        y_array = np.asarray(y_true, dtype=int)
        probabilities = BaseModel._normalize_probabilities(np.asarray(probabilities, dtype=float))
        predicted_labels = np.argmax(probabilities, axis=1)
        confidence = np.max(probabilities, axis=1)
        correctness = (predicted_labels == y_array).astype(float)
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        curve = []

        for index in range(n_bins):
            left = bin_edges[index]
            right = bin_edges[index + 1]
            if index == n_bins - 1:
                mask = (confidence >= left) & (confidence <= right)
            else:
                mask = (confidence >= left) & (confidence < right)
            if not np.any(mask):
                continue
            curve.append({
                'bin_start': float(left),
                'bin_end': float(right),
                'mean_confidence': float(np.mean(confidence[mask])),
                'empirical_accuracy': float(np.mean(correctness[mask])),
                'sample_count': int(np.sum(mask)),
            })

        return curve

    @staticmethod
    def _expected_calibration_error(y_true: pd.Series, probabilities: np.ndarray, n_bins: int = 10) -> float:
        curve = BaseModel._build_reliability_curve(y_true, probabilities, n_bins=n_bins)
        total = sum(item['sample_count'] for item in curve)
        if total == 0:
            return 0.0
        return float(sum(
            abs(item['empirical_accuracy'] - item['mean_confidence']) * item['sample_count'] / total
            for item in curve
        ))
    
    def get_info(self) -> Dict[str, Any]:
        """Получить информацию о модели"""
        return {
            'name': self.name,
            'type': self.model_type,
            'is_trained': self.is_trained,
            'training_date': self.training_date,
            'metrics': self.metrics,
        }
