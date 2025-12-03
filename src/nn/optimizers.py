# src/nn/optimizers.py

from __future__ import annotations
import numpy as np
from typing import Dict


class Optimizer:
    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        raise NotImplementedError


# ----------------------------------------------------------
# Stochastic Gradient Descent (SGD)
# with optional momentum
# ----------------------------------------------------------
# src/nn/optimizers.py

class SGD(Optimizer):
    def __init__(self, lr: float = 0.01, momentum: float = 0.0):
        self.lr = lr
        self.momentum = momentum
        self.velocities = {}   # dictionary: (layer_id, param_name) → velocity

    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray], layer_id: int = 0):
        for name, p in params.items():
            if p is None:
                continue

            grad = grads[name]
            key = (layer_id, name)   # <-- UNIQUE KEY

            if self.momentum > 0:
                if key not in self.velocities:
                    self.velocities[key] = np.zeros_like(p)

                # momentum update
                self.velocities[key] = (
                    self.momentum * self.velocities[key] - self.lr * grad
                )
                params[name] += self.velocities[key]

            else:
                params[name] -= self.lr * grad
