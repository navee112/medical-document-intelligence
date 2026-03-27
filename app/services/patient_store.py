import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

from fastapi import UploadFile

from app.config import DATA_ROOT
from app.schemas import DocumentInput, PatientDocument, PatientSummary, ReportRequest

PATIENT_ID_SANITIZER = re.compile(r"[^A-Za-z0-9_-]+")


def _data_root() -> Path:
    root = Path(DATA_ROOT).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def normalize_patient_id(patient_id: str) -> str:
    cleaned = PATIENT_ID_SANITIZER.sub("_", patient_id.strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise ValueError("Invalid patient id.")
    return cleaned


def _patient_path(patient_id: str) -> tuple[str, Path]:
    normalized = normalize_patient_id(patient_id)
    return normalized, _data_root() / normalized


def create_patient(patient_id: str) -> PatientSummary:
    normalized, patient_path = _patient_path(patient_id)
    patient_path.mkdir(parents=True, exist_ok=True)
    return PatientSummary(patient_id=normalized, documents_count=_count_documents(patient_path))


def _count_documents(path: Path) -> int:
    return sum(1 for entry in path.iterdir() if entry.is_file())


def list_patients() -> list[PatientSummary]:
    root = _data_root()
    summaries: list[PatientSummary] = []
    for entry in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if entry.is_dir():
            summaries.append(
                PatientSummary(
                    patient_id=entry.name,
                    documents_count=_count_documents(entry),
                )
            )
    return summaries


def list_documents(patient_id: str) -> list[PatientDocument]:
    _, patient_path = _patient_path(patient_id)
    if not patient_path.exists():
        return []

    documents: list[PatientDocument] = []
    for entry in sorted(patient_path.iterdir(), key=lambda item: item.name.lower()):
        if not entry.is_file():
            continue
        stat = entry.stat()
        modified = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
        documents.append(
            PatientDocument(
                filename=entry.name,
                path=str(entry),
                size_bytes=stat.st_size,
                modified_at=modified,
            )
        )
    return documents


def _next_available_path(patient_dir: Path, filename: str) -> Path:
    candidate = patient_dir / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    index = 1
    while True:
        new_candidate = patient_dir / f"{stem}_{index}{suffix}"
        if not new_candidate.exists():
            return new_candidate
        index += 1


def save_uploads(patient_id: str, files: list[UploadFile]) -> list[str]:
    _, patient_dir = _patient_path(patient_id)
    patient_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    for upload in files:
        filename = Path(upload.filename or "").name
        if not filename:
            continue
        destination = _next_available_path(patient_dir, filename)
        with destination.open("wb") as output:
            shutil.copyfileobj(upload.file, output)
        saved.append(str(destination))

    return saved


def build_report_request_for_patient(
    patient_id: str,
    objective: str,
    top_k_chunks: int,
) -> ReportRequest:
    normalized, patient_path = _patient_path(patient_id)
    if not patient_path.exists():
        raise FileNotFoundError(f"Patient folder not found: {normalized}")

    documents: list[DocumentInput] = []
    for entry in sorted(patient_path.iterdir(), key=lambda item: item.name.lower()):
        if entry.is_file():
            documents.append(
                DocumentInput(
                    document_id=f"{normalized} - {entry.stem}",
                    path=str(entry),
                )
            )

    if not documents:
        raise FileNotFoundError(f"No documents found for patient: {normalized}")

    return ReportRequest(
        objective=objective,
        documents=documents,
        top_k_chunks=top_k_chunks,
    )

