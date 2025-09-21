from __future__ import annotations
import os
import json
from typing import List, Dict, Any

from src.llm.schema import validate_response

try:
	from openai import OpenAI
except Exception:
	OpenAI = None  # type: ignore


SYSTEM_PROMPT = (
	"You are an investment analyst. Use ONLY the provided evidence chunks. "
	"Return strict JSON matching the schema. Include evidence chunk IDs for each point. "
	"Do not invent facts beyond evidence."
)


class EvidenceLinkedLLM:
	def __init__(self) -> None:
		self.client = None
		if os.getenv("OPENAI_API_KEY") and OpenAI is not None:
			self.client = OpenAI()

	def generate(self, query: str, evidence: List[Dict], ml_score: float, alpha: float, beta: float) -> Dict[str, Any]:
		if self.client is None:
			# Fallback deterministic template result
			opps = [{"point": "Experienced founders", "evidence": evidence[0]["chunk_id"]}] if evidence else []
			risks = [{"point": "Limited traction mentioned", "evidence": evidence[-1]["chunk_id"]}] if evidence else []
			llm_score = 0.65
			final_score = round(alpha * ml_score + beta * llm_score, 4)
			payload = {
				"summary": "Template summary based on retrieved evidence.",
				"opportunities": opps,
				"risks": risks,
				"ml_score": float(ml_score),
				"final_score": float(final_score)
			}
			return payload

		messages = [
			{"role": "system", "content": SYSTEM_PROMPT},
			{"role": "user", "content": json.dumps({
				"query": query,
				"evidence": [{"chunk_id": e["chunk_id"], "text": e["text"]} for e in evidence],
				"ml_score": ml_score,
				"alpha": alpha,
				"beta": beta,
			}, ensure_ascii=False)}
		]

		resp = self.client.chat.completions.create(
			model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
			messages=messages,
			response_format={"type": "json_object"},
			temperature=0.2,
		)
		text = resp.choices[0].message.content
		payload = json.loads(text)
		if not validate_response(payload):
			raise ValueError("LLM response failed schema validation")
		return payload 