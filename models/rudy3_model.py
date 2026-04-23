"""Rudy3: rule-based модель на базе последних 6/6/6 матчей."""

from models.rudy_model import RudyModel


class Rudy3Model(RudyModel):
    """Модель Rudy3: 6 домашних, 6 гостевых и 6 очных матчей."""

    def __init__(self):
        super().__init__(home_window=6, away_window=6, h2h_window=6)
        self.name = "Rudy3"
        self.model_type = "rudy3"
