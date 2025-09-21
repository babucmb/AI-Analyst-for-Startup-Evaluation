from __future__ import annotations
from typing import List, Dict, Any

RESPONSE_SCHEMA = {
	"type": "object",
	"properties": {
		"summary": {"type": "string"},
		"opportunities": {
			"type": "array",
			"items": {
				"type": "object",
				"properties": {
					"point": {"type": "string"},
					"evidence": {"type": "string"}
				},
				"required": ["point", "evidence"]
			}
		},
		"risks": {
			"type": "array",
			"items": {
				"type": "object",
				"properties": {
					"point": {"type": "string"},
					"evidence": {"type": "string"}
				},
				"required": ["point", "evidence"]
			}
		},
		"ml_score": {"type": "number", "minimum": 0, "maximum": 1},
		"final_score": {"type": "number", "minimum": 0, "maximum": 1}
	},
	"required": ["summary", "opportunities", "risks", "ml_score", "final_score"]
}


def validate_response(payload: Dict[str, Any]) -> bool:
	try:
		# Minimal structural checks to avoid heavy deps
		assert isinstance(payload.get("summary"), str)
		assert isinstance(payload.get("opportunities"), list)
		assert isinstance(payload.get("risks"), list)
		assert isinstance(payload.get("ml_score"), (int, float))
		assert isinstance(payload.get("final_score"), (int, float))
		for item in payload.get("opportunities", []):
			assert isinstance(item.get("point"), str)
			assert isinstance(item.get("evidence"), str)
		for item in payload.get("risks", []):
			assert isinstance(item.get("point"), str)
			assert isinstance(item.get("evidence"), str)
		return True
	except Exception:
		return False 