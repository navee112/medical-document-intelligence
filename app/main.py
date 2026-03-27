from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import APP_NAME, APP_VERSION
from app.schemas import (
    AnswerResponse,
    CreatePatientRequest,
    ExtractRequest,
    ExtractResponse,
    PatientDocument,
    PatientReportRequest,
    PatientSummary,
    QuestionRequest,
    ReportRequest,
    ReportResponse,
    UploadResponse,
)
from app.services.llm import get_llm_status
from app.services.router import handle_extraction, handle_question, handle_report
from app.services.patient_store import (
    build_report_request_for_patient,
    create_patient,
    list_documents,
    list_patients,
    normalize_patient_id,
    save_uploads,
)

app = FastAPI(title=APP_NAME, version=APP_VERSION)
STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": APP_NAME,
        "status": "ok",
        "dashboard": "/dashboard",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "service": APP_NAME,
        "version": APP_VERSION,
        "llm": get_llm_status(),
    }


@app.get("/api/health")
def health_alias() -> dict[str, object]:
    return health()


@app.get("/dashboard")
def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "dashboard.html")


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest) -> AnswerResponse:
    return handle_question(request)


@app.post("/extract", response_model=ExtractResponse)
def extract(request: ExtractRequest) -> ExtractResponse:
    return handle_extraction(request)


@app.post("/report", response_model=ReportResponse)
def report(request: ReportRequest) -> ReportResponse:
    return handle_report(request)


@app.get("/patients", response_model=list[PatientSummary])
def patients() -> list[PatientSummary]:
    return list_patients()


@app.post("/patients", response_model=PatientSummary)
def patient_create(request: CreatePatientRequest) -> PatientSummary:
    try:
        return create_patient(request.patient_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/patients/{patient_id}/documents", response_model=list[PatientDocument])
def patient_documents(patient_id: str) -> list[PatientDocument]:
    try:
        normalized = normalize_patient_id(patient_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return list_documents(normalized)


@app.post("/patients/{patient_id}/upload", response_model=UploadResponse)
def patient_upload(patient_id: str, files: list[UploadFile] = File(...)) -> UploadResponse:
    try:
        normalized = normalize_patient_id(patient_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    saved_files = save_uploads(normalized, files)
    return UploadResponse(patient_id=normalized, saved_files=saved_files)


@app.post("/patients/{patient_id}/report", response_model=ReportResponse)
def patient_report(patient_id: str, request: PatientReportRequest) -> ReportResponse:
    try:
        report_request = build_report_request_for_patient(
            patient_id=patient_id,
            objective=request.objective,
            top_k_chunks=request.top_k_chunks,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return handle_report(report_request)
