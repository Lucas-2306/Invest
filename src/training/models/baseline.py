from dataclasses import dataclass
from sklearn.linear_model import Ridge

@dataclass
class RidgeBaseline:
    alpha: float = 1.0

    def build(self):
        return Ridge(alpha=self.alpha, random_state=42)
