# src/nn/activations.py

from __future__ import annotations
import numpy as np
from .layers import Layer


class ReLU(Layer):
    """
    Rectified Linear Unit activation: f(x) = max(0, x)
    """

    def __init__(self) -> None:
        super().__init__()
        self._mask: np.ndarray | None = None

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self._mask = inputs > 0
        return np.where(self._mask, inputs, 0.0)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._mask is None:
            raise RuntimeError("Must call forward() before backward().")
        # Gradient passes only where inputs were positive
        return grad_output * self._mask.astype(grad_output.dtype)


class Softmax(Layer):
    """
    Softmax activation, typically used in the final layer for classification.
    """

    def __init__(self) -> None:
        super().__init__()
        self._outputs: np.ndarray | None = None

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        # Numerical stability: subtract max per row
        shifted = inputs - inputs.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        self._outputs = probs
        return probs

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Generic softmax derivative using the full Jacobian.
        For classification with cross-entropy, we will later use a simpler
        combined formula: grad = (y_pred - y_true) / batch_size.
        """
        if self._outputs is None:
            raise RuntimeError("Must call forward() before backward().")

        batch_size, n_classes = grad_output.shape
        grad_inputs = np.empty_like(grad_output)

        for i in range(batch_size):
            y = self._outputs[i].reshape(-1, 1)  # column vector
            # Jacobian of softmax for one sample
            J = np.diagflat(y) - y @ y.T
            grad_inputs[i] = J @ grad_output[i]

        return grad_inputs
