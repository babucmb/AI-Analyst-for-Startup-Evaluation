from __future__ import annotations
import os
import numpy as np
import pandas as pd

from src.ml.model import SuccessModel, FEATURE_COLUMNS


def generate_synthetic(n: int = 400, seed: int = 42) -> pd.DataFrame:
	rng = np.random.default_rng(seed)
	df = pd.DataFrame({
		"startup_id": [f"s{i}" for i in range(n)],
		"founder_exp_years": rng.normal(5, 3, n).clip(0, 20),
		"funding_stage": rng.integers(0, 4, n),
		"employee_count": rng.normal(8, 10, n).clip(1, 200),
		"traction_score": rng.uniform(0, 1, n),
	})
	logit = (
		0.15*df["founder_exp_years"] + 0.3*df["funding_stage"] + 0.02*df["employee_count"] + 1.5*df["traction_score"] - 2.5
	)
	prob = 1/(1+np.exp(-logit))
	y = (rng.uniform(0,1,n) < prob).astype(int)
	df["label"] = y
	return df


def main() -> None:
	models_dir = os.getenv("MODELS_DIR", "models")
	os.makedirs(models_dir, exist_ok=True)

	df = generate_synthetic()
	y = df["label"].values

	model = SuccessModel()
	model.fit(df, y)

	model_path = os.path.join(models_dir, "success_model.npz")
	model.save(model_path)
	print(f"Saved model to {model_path}")


if __name__ == "__main__":
	main() 