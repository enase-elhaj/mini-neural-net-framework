# src/nn/model.py

from __future__ import annotations
import numpy as np
from typing import List, Tuple

from .layers import Layer
from .losses import Loss
from .optimizers import Optimizer


class Model:
    def __init__(self, layers: List[Layer]):
        self.layers = layers
        self.loss_fn: Loss | None = None
        self.optimizer: Optimizer | None = None

    # -----------------------------------------
    def compile(self, loss: Loss, optimizer: Optimizer):
        self.loss_fn = loss
        self.optimizer = optimizer

    # -----------------------------------------
    def forward(self, X: np.ndarray) -> np.ndarray:
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    # -----------------------------------------
    def backward(self, grad: np.ndarray):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    # -----------------------------------------
    def update_weights(self):
        for i, layer in enumerate(self.layers):
            if layer.params:
                self.optimizer.update(layer.params, layer.grads, layer_id=i)

    # -----------------------------------------
    def train_step(self, X_batch: np.ndarray, y_batch: np.ndarray) -> float:
        # Forward
        y_pred = self.forward(X_batch)

        # Loss
        loss = self.loss_fn.forward(y_pred, y_batch)

        # Backward
        grad = self.loss_fn.backward()
        self.backward(grad)

        # Weight update
        self.update_weights()

        return loss

    # -----------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 20,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> list[float]:

        n_samples = X.shape[0]
        losses = []

        for epoch in range(epochs):
            # shuffle
            indices = np.random.permutation(n_samples)
            X = X[indices]
            y = y[indices]

            batch_losses = []

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X[start:end]
                y_batch = y[start:end]

                loss = self.train_step(X_batch, y_batch)
                batch_losses.append(loss)

            epoch_loss = np.mean(batch_losses)
            losses.append(epoch_loss)

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} — Loss: {epoch_loss:.4f}")

        return losses

    # -----------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)
