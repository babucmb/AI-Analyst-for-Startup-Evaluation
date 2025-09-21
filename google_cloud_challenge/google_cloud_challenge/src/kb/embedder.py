from __future__ import annotations
import os
import hashlib
from typing import List

import numpy as np

try:
	from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
	SentenceTransformer = None  # type: ignore


def _hash_embed(texts: List[str], dim: int = 384) -> np.ndarray:
	if not texts:
		return np.zeros((0, dim), dtype=np.float32)
	vecs = []
	for t in texts:
		acc = np.zeros(dim, dtype=np.float32)
		for tok in t.split():
			h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16)
			rng = np.random.default_rng(h % (2**32))
			acc += rng.standard_normal(dim).astype(np.float32)
		norm = np.linalg.norm(acc) + 1e-8
		vecs.append(acc / norm)
	return np.vstack(vecs)


class TextEmbedder:
	def __init__(self, model_name: str | None = None, dim: int = 384) -> None:
		self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
		self.dim = dim
		self._st_model = None
		if SentenceTransformer is not None and self.model_name.startswith("sentence-transformers/"):
			try:
				self._st_model = SentenceTransformer(self.model_name)
			except Exception:
				self._st_model = None

	def embed(self, texts: List[str]) -> np.ndarray:
		if not texts:
			return np.zeros((0, self.dim), dtype=np.float32)
		if self._st_model is not None:
			try:
				emb = self._st_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
				return emb.astype("float32")
			except Exception:
				pass
		return _hash_embed(texts, dim=self.dim).astype("float32") 