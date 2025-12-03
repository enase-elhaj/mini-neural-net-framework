# src/nn/layers.py

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict
import numpy as np


class Layer(ABC):
    """
    Base class for all layers.
    Each layer implements forward() and backward() and stores
    its parameters and gradients in dictionaries.
    """

    def __init__(self) -> None:
        # Trainable parameters (e.g., weights, biases)
        self.params: Dict[str, np.ndarray] = {}
        # Gradients of the parameters computed during backpropagation
        self.grads: Dict[str, np.ndarray] = {}

    @abstractmethod
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Compute the output of the layer given inputs.
        """
        raise NotImplementedError

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Given gradient of the loss w.r.t. the layer's output,
        compute gradient w.r.t. the inputs and store parameter grads.
        """
        raise NotImplementedError


class Dense(Layer):
    """
    Fully connected (linear) layer:
        y = xW + b
    """

    def __init__(
        self,
        n_inputs: int,
        n_units: int,
        weight_scale: float = 0.01,
        use_bias: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        rng = np.random.default_rng(seed)

        # He or Xavier init could be used; here a simple scaled normal
        self.params["W"] = weight_scale * rng.standard_normal((n_inputs, n_units))
        if use_bias:
            self.params["b"] = np.zeros((1, n_units))
        else:
            self.params["b"] = None

        # Cache for backprop
        self._inputs: np.ndarray | None = None

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        inputs: shape (batch_size, n_inputs)
        returns: shape (batch_size, n_units)
        """
        self._inputs = inputs  # Cache for backward

        W = self.params["W"]
        b = self.params["b"]

        outputs = inputs @ W
        if b is not None:
            outputs = outputs + b
        return outputs

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        grad_output: dL/dY, shape (batch_size, n_units)
        returns: dL/dX, shape (batch_size, n_inputs)
        """
        if self._inputs is None:
            raise RuntimeError("Must call forward() before backward().")

        X = self._inputs
        W = self.params["W"]
        batch_size = X.shape[0]

        # Gradients of parameters (mean over batch for stability)
        self.grads["W"] = X.T @ grad_output / batch_size

        if self.params["b"] is not None:
            self.grads["b"] = grad_output.mean(axis=0, keepdims=True)
        else:
            self.grads["b"] = None

        # Gradient w.r.t inputs
        grad_inputs = grad_output @ W.T
        return grad_inputs
