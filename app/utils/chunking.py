from app.config import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    size = max(1, chunk_size)
    safe_overlap = max(0, min(overlap, size - 1))
    step = size - safe_overlap

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + size, len(cleaned))
        if end < len(cleaned):
            split_at = cleaned.rfind(" ", start, end)
            if split_at > start:
                end = split_at

        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(cleaned):
            break
        start += step

    return chunks

