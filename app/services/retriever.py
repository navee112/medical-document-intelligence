import re
from typing import Iterable

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, TOP_K
from app.utils.chunking import chunk_text as split_text

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    return split_text(text, chunk_size=chunk_size, overlap=overlap)


def load_chunks(
    documents: Iterable[str],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    chunks: list[str] = []
    for doc in documents:
        if doc and doc.strip():
            chunks.extend(chunk_text(doc, chunk_size=chunk_size, overlap=overlap))
    return chunks


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _keyword_score(question: str, chunk: str) -> float:
    q_tokens = _tokenize(question)
    c_tokens = _tokenize(chunk)
    if not q_tokens or not c_tokens:
        return 0.0
    overlap = q_tokens.intersection(c_tokens)
    return len(overlap) / len(q_tokens)


def _tfidf_rank(question: str, chunks: list[str], top_k: int) -> list[int]:
    if not SKLEARN_AVAILABLE or not chunks:
        return []
    try:
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(chunks + [question])
        question_vector = matrix[-1]
        chunk_vectors = matrix[:-1]
        scores = cosine_similarity(question_vector, chunk_vectors).flatten()
        ranked = sorted(enumerate(scores.tolist()), key=lambda pair: pair[1], reverse=True)
        return [idx for idx, _ in ranked[:top_k]]
    except Exception:
        return []


def retrieve_context(question: str, documents: list[str], top_k: int = TOP_K) -> list[str]:
    if not documents:
        return []

    chunks = load_chunks(documents)
    if not chunks:
        return []

    limit = max(1, min(top_k, len(chunks)))
    ranked_indices = _tfidf_rank(question, chunks, limit)
    if not ranked_indices:
        ranked = sorted(
            enumerate(chunks),
            key=lambda pair: _keyword_score(question, pair[1]),
            reverse=True,
        )
        ranked_indices = [idx for idx, _ in ranked[:limit]]

    return [chunks[idx] for idx in ranked_indices]

