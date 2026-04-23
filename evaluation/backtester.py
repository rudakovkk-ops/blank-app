"""
Бэктестирование стратегий прогнозирования
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class Backtester:
    """Симулятор ставок на исторических данных"""
    
    def __init__(self, initial_bankroll: float = 1000, kelly_fraction: float = 0.25):
        """
        Инициализировать бэктестер
        
        Args:
            initial_bankroll: Начальный банк
            kelly_fraction: Доля от Kelly Criterion для ставок
        """
        self.initial_bankroll = initial_bankroll
        self.kelly_fraction = kelly_fraction
        self.trades = []
    
    def calculate_kelly_stake(
        self,
        probability_of_win: float,
        odds: float
    ) -> float:
        """
        Рассчитать размер ставки по Kelly Criterion
        
        Args:
            probability_of_win: Вероятность выигрыша (0-1)
            odds: Букмекерские коэффициенты
        
        Returns:
            Рекомендуемый размер ставки в процентах от банка
        """
        if odds <= 1:
            return 0
        
        # Kelly formula: f* = (p*o - q) / (o - 1)
        # где p = вероятность выигрыша, q = вероятность проигрыша, o = odds
        q = 1 - probability_of_win
        numerator = probability_of_win * odds - q
        denominator = odds - 1
        
        if denominator <= 0:
            return 0
        
        kelly_stake = numerator / denominator
        kelly_stake = max(0, min(kelly_stake, 0.5))  # Ограничить [0, 50%]
        
        # Применить фракцию Kelly
        return kelly_stake * self.kelly_fraction
    
    def backtest(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        actual: np.ndarray,
        odds_column: str = 'implied_odds',
        probability_threshold: float = 0.55,
        use_kelly: bool = True,
        fixed_stake_fraction: float | None = None,
        odds_values: np.ndarray | None = None,
        bet_mask: np.ndarray | None = None,
    ) -> Dict:
        """
        Выполнить бэктест стратегии
        
        Args:
            df: DataFrame с матчами
            predictions: Предсказания модели
            probabilities: Вероятности предсказаний
            actual: Реальные результаты
            odds_column: Столбец с коэффициентами
            probability_threshold: Минимальная вероятность для ставки
            use_kelly: Использовать Kelly Criterion для размера ставки
        
        Returns:
            Словарь с результатами бэктеста
        """
        self.trades = []
        bankroll = self.initial_bankroll
        bankroll_history = [bankroll]
        daily_returns = []
        
        for i, (pred, probs, true) in enumerate(zip(predictions, probabilities, actual)):
            confidence = np.max(probs)

            should_bet = confidence >= probability_threshold
            if bet_mask is not None:
                should_bet = bool(should_bet and bet_mask[i])

            if not should_bet:
                continue  # Не ставим если уверенность низкая

            # Получить коэффициент
            if odds_values is not None:
                odds = float(odds_values[i]) if np.isfinite(odds_values[i]) else np.nan
            elif odds_column in df.columns:
                odds = df.iloc[i].get(odds_column, 1 / confidence)
            else:
                odds = 1 / confidence  # Примерный расчет

            if not np.isfinite(odds) or odds <= 1:
                continue
            
            # Рассчитать размер ставки
            if use_kelly:
                stake_fraction = self.calculate_kelly_stake(confidence, odds)
                stake = bankroll * stake_fraction
            elif fixed_stake_fraction is not None:
                stake = bankroll * fixed_stake_fraction
            else:
                stake = bankroll * 0.02  # 2% от банка

            if stake <= 0:
                continue
            
            # Проверить результат
            win_amount = 0.0
            if pred == true:
                win_amount = stake * (odds - 1)
                bankroll += win_amount
                result = "W"
            else:
                bankroll -= stake
                result = "L"
            
            bankroll_history.append(bankroll)
            daily_returns.append(bankroll - bankroll_history[-2])
            
            self.trades.append({
                'date': df.iloc[i].get('date', None),
                'home_team': df.iloc[i].get('home_team', ''),
                'away_team': df.iloc[i].get('away_team', ''),
                'prediction': pred,
                'actual': true,
                'confidence': confidence,
                'odds': odds,
                'stake': stake,
                'result': result,
                'profit': win_amount if result == 'W' else -stake,
                'bankroll': bankroll,
            })
        
        # Расчет метрик
        total_profit = bankroll - self.initial_bankroll
        total_return = (total_profit / self.initial_bankroll) * 100
        
        win_trades = sum(1 for t in self.trades if t['result'] == 'W')
        loss_trades = sum(1 for t in self.trades if t['result'] == 'L')
        win_rate = (win_trades / (win_trades + loss_trades) * 100
                   if (win_trades + loss_trades) > 0 else 0)
        
        # Sharpe Ratio
        if len(daily_returns) > 0:
            mean_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            sharpe_ratio = ((mean_return / std_return) * np.sqrt(252)
                           if std_return > 0 else 0)
        else:
            sharpe_ratio = 0
        
        # Max drawdown
        max_dd = self._calculate_max_drawdown(bankroll_history)
        
        return {
            'initial_bankroll': self.initial_bankroll,
            'final_bankroll': bankroll,
            'total_profit': total_profit,
            'total_return': total_return,
            'total_trades': len(self.trades),
            'winning_trades': win_trades,
            'losing_trades': loss_trades,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'bankroll_history': bankroll_history,
            'daily_returns': daily_returns,
            'trades': self.trades,
        }
    
    @staticmethod
    def _calculate_max_drawdown(bankroll_history: list) -> float:
        """Рассчитать максимальную просадку"""
        if len(bankroll_history) == 0:
            return 0.0
        
        peak = bankroll_history[0]
        max_dd = 0.0
        
        for value in bankroll_history:
            if value > peak:
                peak = value
            
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd * 100
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """Получить все ставки в виде DataFrame"""
        return pd.DataFrame(self.trades)
    
    def get_summary(self) -> Dict:
        """Получить краткую сводку по всем ставкам"""
        if len(self.trades) == 0:
            return {}
        
        df = self.get_trades_dataframe()
        
        return {
            'total_trades': len(self.trades),
            'winning_trades': (df['result'] == 'W').sum(),
            'losing_trades': (df['result'] == 'L').sum(),
            'win_rate': ((df['result'] == 'W').sum() / len(self.trades) * 100),
            'avg_profit_per_trade': df['profit'].mean(),
            'total_profit': df['profit'].sum(),
            'min_profit': df['profit'].min(),
            'max_profit': df['profit'].max(),
        }
