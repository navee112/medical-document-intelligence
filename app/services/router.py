from app.config import TOP_K
from app.prompts import GROUNDED_QA_PROMPT
from app.schemas import (
    AnswerResponse,
    ExtractRequest,
    ExtractResponse,
    QuestionRequest,
    ReportRequest,
    ReportResponse,
)
from app.services.extractor import extract_medical_fields
from app.services.llm import call_llm
from app.services.retriever import retrieve_context
from app.services.synthesizer import generate_executive_report


def handle_question(request: QuestionRequest) -> AnswerResponse:
    documents = request.documents or []
    context_chunks = retrieve_context(request.question, documents, top_k=TOP_K)

    if not context_chunks:
        return AnswerResponse(
            answer="Insufficient evidence in provided documents.",
            sources=[],
            grounded=False,
        )

    formatted_context = "\n\n".join(
        f"[chunk_{idx + 1}] {chunk}" for idx, chunk in enumerate(context_chunks)
    )
    prompt = GROUNDED_QA_PROMPT.format(context=formatted_context, question=request.question)
    answer = call_llm(prompt)
    sources = [f"chunk_{idx + 1}" for idx in range(len(context_chunks))]

    grounded = "insufficient evidence in provided documents" not in answer.lower()
    return AnswerResponse(answer=answer, sources=sources, grounded=grounded)


def handle_extraction(request: ExtractRequest) -> ExtractResponse:
    payload = extract_medical_fields(request.text)
    return ExtractResponse(
        document_type=payload.get("document_type"),
        extracted_fields=payload.get("extracted_fields", {}),
        warnings=payload.get("warnings", []),
    )


def handle_report(request: ReportRequest) -> ReportResponse:
    return generate_executive_report(request)
