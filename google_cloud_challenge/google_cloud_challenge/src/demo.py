from __future__ import annotations
import os
import json
import numpy as np

from src.ingestion.pdf_parser import parse_pdfs_to_chunks
from src.kb.embedder import TextEmbedder
from src.kb.vector_store import NumpyVectorDB
from src.kb.retriever import EvidenceRetriever
from src.ml.features import build_features
from src.ml.model import SuccessModel
from src.llm.generator import EvidenceLinkedLLM
from src.hybrid.decision import combine_scores, build_output


def ensure_sample_pdf(dir_path: str) -> None:
	os.makedirs(dir_path, exist_ok=True)
	# No-op: we avoid creating PDFs to stay pure-Python on Py 3.13
	return


def run_demo(startup_id: str = "demo_startup") -> None:
	data_dir = os.getenv("DATA_DIR", "data/samples")
	ensure_sample_pdf(data_dir)
	pdfs = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
	chunks = parse_pdfs_to_chunks(startup_id, pdfs)
	chunk_dicts = [c.__dict__ for c in chunks]

	embedder = TextEmbedder()
	dim = 384
	store = NumpyVectorDB(dim)
	retriever = EvidenceRetriever(embedder, store)
	retriever.add_chunks(chunk_dicts)

	evidence_lists = retriever.retrieve(["What are key risks and opportunities?"], top_k=5)
	evidence = evidence_lists[0]

	meta = {
		"founder_exp_years": 6,
		"funding_stage": "seed",
		"employee_count": 10,
		"metrics": {"monthly_active_users": 2000, "mrr": 5000, "qoq_growth": 20}
	}
	df = build_features(startup_id, meta)
	model = SuccessModel()
	model_path = os.path.join(os.getenv("MODELS_DIR", "models"), "success_model.npz")
	if os.path.exists(model_path):
		model.load(model_path)
	else:
		model.fit(df, np.array([1], dtype=float))
	ml_score = float(model.predict_proba(df)[0])

	alpha = float(os.getenv("ALPHA", 0.6))
	beta = float(os.getenv("BETA", 0.4))

	llm = EvidenceLinkedLLM()
	payload = llm.generate("Analyze startup investment profile", evidence, ml_score, alpha, beta)

	print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
	run_demo() 