from __future__ import annotations

import hashlib
import json
import os
import re
from typing import List, Tuple

from .types import Chunk


def _read_pdf_pages(pdf_path: str) -> List[str]:
    """
    Returns list of per-page text.
    Uses pypdf if available, else PyPDF2.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = None
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(pdf_path)
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore
            reader = PdfReader(pdf_path)
        except Exception as e:
            raise ImportError(
                "Need a PDF reader. Install one:\n"
                "  pip install pypdf\n"
                "(fallback supported: PyPDF2)"
            ) from e

    pages: List[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return pages


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_id(i: int) -> str:
    return f"c{i:06d}"


def chunk_pdf(
    pdf_path: str,
    chunk_size: int = 1400,
    overlap: int = 200,
) -> List[Chunk]:
    """
    Character-based chunking (checkpoint-friendly and stable).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    pages = _read_pdf_pages(pdf_path)
    pages = [_clean_text(p) for p in pages]

    chunks: List[Chunk] = []
    idx = 0

    for page_num, page_text in enumerate(pages, start=1):
        if not page_text:
            continue

        start = 0
        n = len(page_text)
        while start < n:
            end = min(start + chunk_size, n)
            slice_text = page_text[start:end].strip()
            if slice_text:
                chunks.append(
                    Chunk(
                        chunk_id=_chunk_id(idx),
                        text=slice_text,
                        meta={
                            "source": os.path.basename(pdf_path),
                            "page": page_num,
                            "char_start": start,
                            "char_end": end,
                        },
                    )
                )
                idx += 1

            if end == n:
                break
            start = max(0, end - overlap)

    return chunks


def _hash_file_and_params(pdf_path: str, chunk_size: int, overlap: int) -> str:
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    h.update(f"|chunk_size={chunk_size}|overlap={overlap}".encode("utf-8"))
    return h.hexdigest()[:16]


def save_chunks_jsonl(chunks: List[Chunk], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"chunk_id": c.chunk_id, "text": c.text, "meta": c.meta}, ensure_ascii=False))
            f.write("\n")


def load_chunks_jsonl(path: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            chunks.append(Chunk(chunk_id=obj["chunk_id"], text=obj["text"], meta=obj.get("meta", {})))
    return chunks


def load_or_build_chunks(
    pdf_path: str,
    chunk_size: int = 1400,
    overlap: int = 200,
    cache_dir: str = "data/cache",
    force_rebuild: bool = False,
) -> Tuple[List[Chunk], str]:
    """
    Returns (chunks, cache_path)
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = _hash_file_and_params(pdf_path, chunk_size, overlap)
    cache_path = os.path.join(cache_dir, f"chunks_{key}.jsonl")

    if (not force_rebuild) and os.path.exists(cache_path):
        return load_chunks_jsonl(cache_path), cache_path

    chunks = chunk_pdf(pdf_path, chunk_size=chunk_size, overlap=overlap)
    save_chunks_jsonl(chunks, cache_path)
    return chunks, cache_path