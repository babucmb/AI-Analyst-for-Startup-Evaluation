from __future__ import annotations
import os
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
	"founder_exp_years",
	"funding_stage",
	"employee_count",
	"traction_score",
]


class SuccessModel:
	def __init__(self) -> None:
		self.weights: np.ndarray | None = None
		self.bias: float = 0.0

	def fit(self, df: pd.DataFrame, y: np.ndarray, lr: float = 0.05, epochs: int = 800) -> None:
		X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
		X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
		y = y.astype(np.float32)
		n, d = X.shape
		w = np.zeros(d, dtype=np.float32)
		b = 0.0
		for _ in range(epochs):
			z = X @ w + b
			p = 1.0 / (1.0 + np.exp(-z))
			grad_w = (X.T @ (p - y)) / n
			grad_b = float(np.mean(p - y))
			w -= lr * grad_w
			b -= lr * grad_b
		self.weights = w
		self.bias = b

	def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
		if self.weights is None:
			raise RuntimeError("Model not trained or loaded")
		X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
		X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
		z = X @ self.weights + self.bias
		p = 1.0 / (1.0 + np.exp(-z))
		return p

	def save(self, path: str) -> None:
		params = {"weights": self.weights, "bias": self.bias}
		np.savez(path, **params)

	def load(self, path: str) -> None:
		obj = np.load(path)
		self.weights = obj["weights"]
		self.bias = float(obj["bias"]) 