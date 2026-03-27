from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class QuestionRequest(BaseModel):
    question: str
    documents: Optional[List[str]] = None


class AnswerResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    grounded: bool = True


class ExtractRequest(BaseModel):
    text: str


class ExtractResponse(BaseModel):
    document_type: Optional[str] = None
    extracted_fields: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


class DocumentInput(BaseModel):
    document_id: Optional[str] = None
    path: Optional[str] = None
    text: Optional[str] = None

    @model_validator(mode="after")
    def validate_source(self) -> "DocumentInput":
        if not (self.path or self.text):
            raise ValueError("Each document must include either `path` or `text`.")

        if not self.document_id:
            if self.path:
                self.document_id = Path(self.path).stem
            else:
                self.document_id = "document"
        return self


class Finding(BaseModel):
    statement: str
    citations: List[str] = Field(default_factory=list)


class CrossDocumentInsight(BaseModel):
    statement: str
    citations: List[str] = Field(default_factory=list)


class ReportRequest(BaseModel):
    objective: str = (
        "Produce an executive intelligence report with grounded citations. "
        "Prioritize clinical abnormalities, trends over time, and follow-up risks. "
        "Ignore legal disclaimers and administrative boilerplate."
    )
    documents: List[DocumentInput]
    top_k_chunks: int = 12


class ReportResponse(BaseModel):
    executive_summary: str
    key_findings: List[Finding] = Field(default_factory=list)
    cross_document_insights: List[CrossDocumentInsight] = Field(default_factory=list)
    sources_used: List[str] = Field(default_factory=list)
    grounded: bool = True
    warnings: List[str] = Field(default_factory=list)


class CreatePatientRequest(BaseModel):
    patient_id: str


class PatientSummary(BaseModel):
    patient_id: str
    documents_count: int = 0


class PatientDocument(BaseModel):
    filename: str
    path: str
    size_bytes: int
    modified_at: str


class UploadResponse(BaseModel):
    patient_id: str
    saved_files: List[str] = Field(default_factory=list)


class PatientReportRequest(BaseModel):
    objective: str = (
        "Produce an executive intelligence report with grounded citations. "
        "Prioritize clinical abnormalities, trends over time, and follow-up risks. "
        "Ignore legal disclaimers and administrative boilerplate."
    )
    top_k_chunks: int = 12
