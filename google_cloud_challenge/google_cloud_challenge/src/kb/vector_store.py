from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class VectorEntry:
	startup_id: str
	chunk_id: str
	text: str
	source: str
	page_no: int
	embedding: np.ndarray


class NumpyVectorDB:
	def __init__(self, dim: int) -> None:
		self.dim = dim
		self._matrix: np.ndarray | None = None
		self.entries: List[VectorEntry] = []

	def add_entries(self, entries: List[VectorEntry]) -> None:
		if not entries:
			return
		vecs = np.vstack([e.embedding for e in entries]).astype("float32")
		norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
		vecs = vecs / norms
		self._matrix = vecs if self._matrix is None else np.vstack([self._matrix, vecs])
		self.entries.extend(entries)

	def search(self, query_vecs: np.ndarray, top_k: int = 5) -> List[List[Tuple[VectorEntry, float]]]:
		if self._matrix is None or len(self.entries) == 0:
			return [[] for _ in range(len(query_vecs))]
		Q = query_vecs.astype("float32")
		Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
		scores = Q @ self._matrix.T
		results: List[List[Tuple[VectorEntry, float]]] = []
		for row in scores:
			idxs = np.argsort(-row)[:top_k]
			items = [(self.entries[i], float(row[i])) for i in idxs]
			results.append(items)
		return results 