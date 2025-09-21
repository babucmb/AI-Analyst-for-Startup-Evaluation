from __future__ import annotations
from typing import List, Dict
import numpy as np

from src.kb.embedder import TextEmbedder
from src.kb.vector_store import NumpyVectorDB, VectorEntry


class EvidenceRetriever:
	def __init__(self, embedder: TextEmbedder, store: NumpyVectorDB) -> None:
		self.embedder = embedder
		self.store = store

	def add_chunks(self, chunks: List[Dict]) -> None:
		if not chunks:
			return
		texts = [c["text"] for c in chunks]
		vecs = self.embedder.embed(texts)
		if vecs.shape[0] == 0:
			return
		entries: List[VectorEntry] = []
		for c, v in zip(chunks, vecs):
			entries.append(VectorEntry(
				startup_id=c["startup_id"],
				chunk_id=c["chunk_id"],
				text=c["text"],
				source=c["source"],
				page_no=c["page_no"],
				embedding=v,
			))
		self.store.add_entries(entries)

	def retrieve(self, questions: List[str], top_k: int = 5) -> List[List[Dict]]:
		if not questions:
			return [[]]
		q_vecs = self.embedder.embed(questions)
		if q_vecs.shape[0] == 0:
			return [[] for _ in questions]
		results = self.store.search(q_vecs, top_k=top_k)
		bundles: List[List[Dict]] = []
		for row in results:
			bundles.append([
				{
					"chunk_id": e.chunk_id,
					"text": e.text,
					"source": e.source,
					"page_no": e.page_no,
					"score": score,
				}
				for e, score in row
			])
		return bundles