# AI-Analyst-for-Startup-Evaluation

Hybrid LLM + ML system to analyze startup founder materials and public data, producing explainable investment insights with traceable evidence and an interactive Streamlit UI.

## Features
- Data ingestion: PDF parsing (PyMuPDF/pdfplumber) with OCR fallback
- Knowledge base: chunking, embeddings (SentenceTransformers), FAISS vector DB
- RAG: retrieve evidence chunks; LLM outputs strict JSON with evidence links
- ML scoring: engineered features + Logistic Regression or LightGBM
- Hybrid decision: final_score = α * ml_score + β * llm_score
- UI: Streamlit tabs (One-Pager, Radar Chart, Q&A)

## Quickstart
1. Create environment and install deps
```
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Set environment variables in `.env`
```
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ALPHA=0.6
BETA=0.4
```

3. Run Streamlit app
```
streamlit run src/app.py
```

## Repo Structure
```
src/
  ingestion/
    pdf_parser.py
  kb/
    embedder.py
    vector_store.py
    retriever.py
  llm/
    schema.py
    generator.py
  ml/
    features.py
    train.py
    model.py
  hybrid/
    decision.py
  ui/
    components.py
  app.py
  demo.py
data/
  samples/
models/
artifacts/
```

## Data Structures
- Vector DB Entry: `{startup_id, chunk_id, text, embedding, source, page_no}`
- ML Feature Table: `{startup_id, founder_exp_years, funding_stage, employee_count, traction_score, ...}`
- LLM Output JSON:
```
{
  "summary": "...",
  "opportunities": [{"point": "...", "evidence": "chunk_12"}],
  "risks": [{"point": "...", "evidence": "chunk_23"}],
  "ml_score": 0.72,
  "final_score": 0.68
}
```

## Notes
- Strict evidence linking is enforced; chunks are included by id in outputs.
- Models are lightweight (SentenceTransformers + Logistic Regression by default).

- Works offline with FAISS; can swap to Pinecone with minimal changes.

