import numpy as np


class EarlyStopping:
    def __init__(self, patience: int = 10, delta: int = 0) -> None:

        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = np.Inf

    def __call__(self, score: float) -> bool:

        if self.best_score is None:
            self.best_score = score
            return False

        elif score > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            else:
                return False

        else:
            self.best_score = score
            self.counter = 0
            return False
