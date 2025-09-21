from __future__ import annotations
from typing import Dict, Any
import pandas as pd


CATEGORIES_STAGE = ["idea", "pre-seed", "seed", "series-a", "series-b+"]


def compute_traction_score(metrics: Dict[str, Any]) -> float:
	users = float(metrics.get("monthly_active_users", 0) or 0)
	rev = float(metrics.get("mrr", 0) or 0)
	growth = float(metrics.get("qoq_growth", 0) or 0)
	score = 0.0
	if users > 0:
		score += min(0.4, 0.4 * (users / 10000.0))
	if rev > 0:
		score += min(0.4, 0.4 * (rev / 100000.0))
	score += max(0.0, min(0.2, growth / 100.0 * 0.2))
	return round(score, 4)


def build_features(startup_id: str, meta: Dict[str, Any]) -> pd.DataFrame:
	founder_exp_years = float(meta.get("founder_exp_years", 0) or 0)
	stage = str(meta.get("funding_stage", "idea")).lower()
	employee_count = float(meta.get("employee_count", 1) or 1)
	metrics = meta.get("metrics", {}) or {}
	traction_score = compute_traction_score(metrics)

	stage_idx = CATEGORIES_STAGE.index(stage) if stage in CATEGORIES_STAGE else 0

	row = {
		"startup_id": startup_id,
		"founder_exp_years": founder_exp_years,
		"funding_stage": stage_idx,
		"employee_count": employee_count,
		"traction_score": traction_score,
	}
	return pd.DataFrame([row]) 