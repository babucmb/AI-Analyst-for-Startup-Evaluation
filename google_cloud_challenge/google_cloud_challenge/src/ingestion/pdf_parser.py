from __future__ import annotations
import io
import os
from dataclasses import dataclass
from typing import List, Dict, Iterable

import pdfplumber


@dataclass
class ParsedChunk:
	startup_id: str
	chunk_id: str
	text: str
	source: str
	page_no: int


def extract_text_pdfplumber(pdf_path: str) -> List[str]:
	texts: List[str] = []
	with pdfplumber.open(pdf_path) as pdf:
		for page in pdf.pages:
			texts.append(page.extract_text() or "")
	return texts


# Removed PyMuPDF path to ensure Python 3.13 compatibility


# OCR removed for minimal, dependency-light run on Python 3.13


# OCR fallback removed


def fallback_text_extraction(pdf_path: str) -> List[str]:
	texts = extract_text_pdfplumber(pdf_path)
	return texts


def chunk_text(text: str, max_tokens: int = 500, overlap: int = 50) -> List[str]:
	words = text.split()
	chunks: List[str] = []
	start = 0
	while start < len(words):
		end = min(len(words), start + max_tokens)
		chunk = " ".join(words[start:end]).strip()
		if chunk:
			chunks.append(chunk)
		start = end - overlap
		if start < 0:
			start = 0
	return chunks


def parse_pdfs_to_chunks(startup_id: str, pdf_paths: Iterable[str]) -> List[ParsedChunk]:
	chunks: List[ParsedChunk] = []
	for pdf_path in pdf_paths:
		pages = fallback_text_extraction(pdf_path)
		for i, text in enumerate(pages, start=1):
			for j, c in enumerate(chunk_text(text), start=1):
				chunk_id = f"{startup_id}_{os.path.basename(pdf_path)}_p{i}_c{j}"
				chunks.append(ParsedChunk(
					startup_id=startup_id,
					chunk_id=chunk_id,
					text=c,
					source=os.path.basename(pdf_path),
					page_no=i,
				))
	return chunks 