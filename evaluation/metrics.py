"""
Метрики для оценки моделей проектирования
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, classification_report
)

logger = logging.getLogger(__name__)


class ModelMetrics:
    """Класс для расчета метрик моделей"""
    
    @staticmethod
    def calculate_roi(
        predictions: np.ndarray,
        probabilities: np.ndarray,
        actual: np.ndarray,
        initial_bankroll: float = 1000,
        stake_size: float = 10
    ) -> Tuple[float, Dict]:
        """
        Рассчитать ROI (Return on Investment)
        
        Args:
            predictions: Предсказания модели
            probabilities: Вероятности предсказаний
            actual: Реальные результаты
            initial_bankroll: Начальный банк
            stake_size: Размер ставки
        
        Returns:
            Кортеж (ROI в %, детали)
        """
        bankroll = initial_bankroll
        bets = []
        
        for i, (pred, probs, true) in enumerate(zip(predictions, probabilities, actual)):
            # Выбираем букмекерский коэффициент (примерный расчет)
            confidence = np.max(probs)
            
            # Ставим если уверенность > 55%
            if confidence > 0.55:
                # Имплидовый коэффициент
                implied_odds = 1 / np.max(probs)
                
                # Ставим на предсказание
                if pred == true:
                    win = stake_size * implied_odds
                    bankroll += win
                    result = "W"
                else:
                    bankroll -= stake_size
                    result = "L"
            else:
                result = "N"  # No bet
                win = 0
            
            bets.append({
                'prediction': pred,
                'actual': true,
                'confidence': confidence,
                'result': result,
                'stake': stake_size if confidence > 0.55 else 0,
                'win': win if confidence > 0.55 else 0,
            })
        
        roi = ((bankroll - initial_bankroll) / initial_bankroll) * 100
        
        return roi, {
            'roi': roi,
            'final_bankroll': bankroll,
            'initial_bankroll': initial_bankroll,
            'profit': bankroll - initial_bankroll,
            'total_bets': sum(1 for b in bets if b['result'] != "N"),
            'winning_bets': sum(1 for b in bets if b['result'] == "W"),
            'losing_bets': sum(1 for b in bets if b['result'] == "L"),
            'bets_detail': bets
        }
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        Рассчитать Sharpe Ratio
        
        Args:
            returns: Массив дневных доходов
            risk_free_rate: Безрисковая ставка
        
        Returns:
            Sharpe Ratio
        """
        if len(returns) == 0:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualized (252 торговых дня в году)
        sharpe = ((mean_return - risk_free_rate) / std_return) * np.sqrt(252)
        return sharpe
    
    @staticmethod
    def calculate_max_drawdown(bankroll_history: list) -> float:
        """
        Рассчитать максимальную просадку
        
        Args:
            bankroll_history: История банкролла
        
        Returns:
            Максимальная просадка в %
        """
        if len(bankroll_history) == 0:
            return 0.0
        
        peak = bankroll_history[0]
        max_drawdown = 0.0
        
        for value in bankroll_history:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return max_drawdown * 100
    
    @staticmethod
    def calculate_win_rate(actual: np.ndarray, predictions: np.ndarray) -> float:
        """Рассчитать процент правильных подсказаний"""
        return np.mean(actual == predictions) * 100
    
    @staticmethod
    def calculate_precision_recall_f1(
        actual: np.ndarray,
        predictions: np.ndarray
    ) -> Dict[str, float]:
        """Рассчитать Precision, Recall, F1"""
        return {
            'precision': precision_score(actual, predictions, average='weighted', zero_division=0),
            'recall': recall_score(actual, predictions, average='weighted', zero_division=0),
            'f1': f1_score(actual, predictions, average='weighted', zero_division=0),
        }
    
    @staticmethod
    def calculate_all_metrics(
        actual: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        initial_bankroll: float = 1000
    ) -> Dict:
        """Рассчитать все метрики"""
        metrics = {
            'accuracy': accuracy_score(actual, predictions),
        }
        
        # Precision, Recall, F1
        pr_metrics = ModelMetrics.calculate_precision_recall_f1(actual, predictions)
        metrics.update(pr_metrics)
        
        # ROC-AUC
        try:
            metrics['roc_auc'] = roc_auc_score(actual, probabilities, multi_class='ovr', average='weighted')
        except:
            metrics['roc_auc'] = 0.0
        
        # Log Loss
        try:
            metrics['logloss'] = log_loss(actual, probabilities)
        except:
            metrics['logloss'] = 0.0
        
        # ROI
        roi, roi_details = ModelMetrics.calculate_roi(
            predictions, probabilities, actual,
            initial_bankroll=initial_bankroll
        )
        metrics['roi'] = roi
        metrics['roi_details'] = roi_details
        
        return metrics
    
    @staticmethod
    def get_classification_report(
        actual: np.ndarray,
        predictions: np.ndarray,
        target_names: list = None
    ) -> str:
        """Получить подробный classification report"""
        if target_names is None:
            target_names = ['Home Win', 'Draw', 'Away Win']
        
        return classification_report(actual, predictions, target_names=target_names)
