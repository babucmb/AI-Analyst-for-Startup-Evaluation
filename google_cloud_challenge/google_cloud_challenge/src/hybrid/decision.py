from __future__ import annotations
from typing import Dict, Any


def combine_scores(ml_score: float, llm_score: float, alpha: float, beta: float) -> float:
	alpha = float(alpha)
	beta = float(beta)
	if alpha + beta == 0:
		return 0.0
	return round(alpha * ml_score + beta * llm_score, 4)


def build_output(summary: str, opportunities, risks, ml_score: float, final_score: float):
	return {
		"summary": summary,
		"opportunities": opportunities,
		"risks": risks,
		"ml_score": float(ml_score),
		"final_score": float(final_score),
	} 