import os
import io
import json
import tempfile
import streamlit as st
import plotly.graph_objects as go

from src.ingestion.pdf_parser import parse_pdfs_to_chunks
from src.kb.embedder import TextEmbedder
from src.kb.vector_store import NumpyVectorDB
from src.kb.retriever import EvidenceRetriever
from src.llm.generator import EvidenceLinkedLLM
from src.llm.schema import validate_response
from src.ml.features import build_features
from src.ml.model import SuccessModel


def _ensure_session():
	if "analysis_result" not in st.session_state:
		st.session_state["analysis_result"] = None
	if "evidence_chunks" not in st.session_state:
		st.session_state["evidence_chunks"] = []
	if "retriever_cache" not in st.session_state:
		st.session_state["retriever_cache"] = None


def _save_uploads(files) -> list[str]:
	if not files:
		return []
	tmp_dir = os.path.join("artifacts", "upload_cache")
	os.makedirs(tmp_dir, exist_ok=True)
	paths: list[str] = []
	for f in files:
		data = f.read()
		path = os.path.join(tmp_dir, f.name)
		with open(path, "wb") as out:
			out.write(data)
		paths.append(path)
	return paths


def _build_retriever(chunks_dicts):
	embedder = TextEmbedder()
	store = NumpyVectorDB(dim=384)
	retriever = EvidenceRetriever(embedder, store)
	retriever.add_chunks(chunks_dicts)
	return retriever


def _analyze_pipeline(startup_id: str, pdf_paths: list[str], alpha: float, beta: float):
	# Ingestion
	chunks = parse_pdfs_to_chunks(startup_id, pdf_paths)
	chunk_dicts = [c.__dict__ for c in chunks]

	# Retriever
	retriever = _build_retriever(chunk_dicts)
	evidence_lists = retriever.retrieve(["Identify key risks and opportunities for this startup"], top_k=8)
	evidence = evidence_lists[0] if evidence_lists else []

	# ML features and score
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
		model.fit(df, df.assign(label=1)["label"].to_numpy())
	ml_score = float(model.predict_proba(df)[0])

	# LLM JSON with evidence linking
	llm = EvidenceLinkedLLM()
	payload = llm.generate("Provide a one-pager summary with risks and opportunities.", evidence, ml_score, alpha, beta)
	valid = validate_response(payload)
	return payload, valid, chunk_dicts, retriever


def _radar_from_payload(payload: dict):
	# Derive simple radar: count-based normalized scores
	labels = ["Team", "Market", "Product", "Traction", "Moat"]
	opps = [0, 0, 0, 0, 0]
	risks = [0, 0, 0, 0, 0]
	for item in payload.get("opportunities", []):
		text = (item.get("point") or "").lower()
		if any(k in text for k in ["team", "founder", "hiring"]):
			opps[0] += 1
		elif any(k in text for k in ["market", "demand", "gtm", "sales"]):
			opps[1] += 1
		elif any(k in text for k in ["product", "tech", "roadmap", "ip"]):
			opps[2] += 1
		elif any(k in text for k in ["traction", "growth", "mrr", "mau"]):
			opps[3] += 1
		else:
			opps[4] += 1
	for item in payload.get("risks", []):
		text = (item.get("point") or "").lower()
		if any(k in text for k in ["team", "founder", "hiring"]):
			risks[0] += 1
		elif any(k in text for k in ["market", "demand", "gtm", "sales"]):
			risks[1] += 1
		elif any(k in text for k in ["product", "tech", "roadmap", "ip"]):
			risks[2] += 1
		elif any(k in text for k in ["traction", "growth", "mrr", "mau"]):
			risks[3] += 1
		else:
			risks[4] += 1
	# normalize to 0..1
	max_v = max(1, max(opps + risks))
	opps = [v / max_v for v in opps]
	risks = [v / max_v for v in risks]
	return labels, opps, risks


def render_one_pager():
	_ensure_session()
	st.subheader("One-Pager Summary")
	uploaded = st.file_uploader("Upload founder PDF(s)", type=["pdf"], accept_multiple_files=True)
	col1, col2 = st.columns([2,1])
	with col1:
		startup_id = st.text_input("Startup ID", value="demo_startup")
		alpha = st.slider("Alpha (ML weight)", 0.0, 1.0, float(os.getenv("ALPHA", 0.6)))
		beta = st.slider("Beta (LLM weight)", 0.0, 1.0, float(os.getenv("BETA", 0.4)))
	with col2:
		analyze = st.button("Analyze")

	if analyze:
		paths = _save_uploads(uploaded)
		payload, is_valid, chunks, retriever = _analyze_pipeline(startup_id, paths, alpha, beta)
		st.session_state["analysis_result"] = payload
		st.session_state["evidence_chunks"] = chunks
		st.session_state["retriever_cache"] = retriever

		if not is_valid:
			st.error("LLM output failed schema validation. Showing raw payload.")
		st.json(payload)

		# Evidence table
		if chunks:
			with st.expander("Evidence Chunks"):
				rows = [{"chunk_id": c["chunk_id"], "source": c["source"], "page_no": c["page_no"], "text": c["text"][:200] + ("..." if len(c["text"])>200 else "")} for c in chunks]
				st.dataframe(rows, use_container_width=True)


def render_radar_tab():
	_ensure_session()
	st.subheader("Risk vs Opportunity Radar")
	payload = st.session_state.get("analysis_result")
	if not payload:
		st.info("Run Analyze in the One-Pager tab first.")
		return
	labels, opps, risks = _radar_from_payload(payload)
	fig = go.Figure()
	fig.add_trace(go.Scatterpolar(r=opps, theta=labels, fill='toself', name='Opportunities'))
	fig.add_trace(go.Scatterpolar(r=risks, theta=labels, fill='toself', name='Risks'))
	fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True)
	st.plotly_chart(fig, use_container_width=True)


def render_qa_tab():
	_ensure_session()
	st.subheader("Interactive Analyst Q&A")
	query = st.text_input("Ask a question about the startup and materials")
	ask = st.button("Ask")
	if ask and query:
		chunks = st.session_state.get("evidence_chunks") or []
		if not chunks:
			st.warning("No evidence loaded. Upload PDFs and run Analyze first.")
			return
		retriever = _build_retriever(chunks)
		results = retriever.retrieve([query], top_k=6)[0]
		llm = EvidenceLinkedLLM()
		# Use previous ML score if available
		payload = st.session_state.get("analysis_result") or {"ml_score": 0.5}
		answer = llm.generate(query, results, float(payload.get("ml_score", 0.5)), float(os.getenv("ALPHA", 0.6)), float(os.getenv("BETA", 0.4)))
		st.json(answer)
		with st.expander("Evidence used"):
			rows = [{"chunk_id": r["chunk_id"], "source": r["source"], "page_no": r["page_no"], "text": r["text"][:200] + ("..." if len(r["text"])>200 else "")} for r in results]
			st.dataframe(rows, use_container_width=True) 