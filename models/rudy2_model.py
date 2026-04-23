"""Rudy2: rule-based модель на базе последних 4/4/4 матчей."""

from models.rudy_model import RudyModel


class Rudy2Model(RudyModel):
    """Модель Rudy2: 4 домашних, 4 гостевых и 4 очных матча."""

    def __init__(self):
        super().__init__(home_window=4, away_window=4, h2h_window=4)
        self.name = "Rudy2"
        self.model_type = "rudy2"
