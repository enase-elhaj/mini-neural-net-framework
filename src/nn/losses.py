# src/nn/losses.py

from __future__ import annotations
import numpy as np


class Loss:
    """Base class for all losses."""

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        raise NotImplementedError

    def backward(self) -> np.ndarray:
        raise NotImplementedError


# ----------------------------------------------------------
# Mean Squared Error
# ----------------------------------------------------------
class MeanSquaredError(Loss):
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true) ** 2)

    def backward(self) -> np.ndarray:
        # dL/dy_pred
        return 2 * (self.y_pred - self.y_true) / self.y_pred.shape[0]


# ----------------------------------------------------------
# Categorical Cross Entropy
# y_true must be one-hot encoded
# ----------------------------------------------------------
class CategoricalCrossEntropy(Loss):
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = y_pred
        self.y_true = y_true

        # avoid log(0)
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1.0 - eps)

        # mean cross entropy
        sample_losses = -np.sum(y_true * np.log(y_pred), axis=1)
        return np.mean(sample_losses)

    def backward(self) -> np.ndarray:
        eps = 1e-12
        y_pred = np.clip(self.y_pred, eps, 1.0 - eps)
        return -(self.y_true / y_pred) / self.y_pred.shape[0]


# ----------------------------------------------------------
# Combined Softmax + CCE
# Efficient gradient: (y_pred - y_true) / batch_size
# ----------------------------------------------------------
class SoftmaxCrossEntropy(Loss):
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = y_pred
        self.y_true = y_true

        # numerical stability
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1.0 - eps)

        sample_losses = -np.sum(y_true * np.log(y_pred), axis=1)
        return np.mean(sample_losses)

    def backward(self) -> np.ndarray:
        # simplified gradient
        batch_size = self.y_pred.shape[0]
        return (self.y_pred - self.y_true) / batch_size
