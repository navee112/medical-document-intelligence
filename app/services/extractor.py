import re
from typing import Any

from app.prompts import STRUCTURED_EXTRACTION_PROMPT
from app.services.llm import call_llm_json

LAB_VALUE_PATTERN = re.compile(
    r"([A-Za-z][A-Za-z0-9 ()/%-]{1,40})[:=]\s*([-+]?\d+(?:\.\d+)?)\s*([A-Za-z/%]+)?"
)


def _default_output() -> dict[str, Any]:
    return {
        "document_type": "unknown",
        "extracted_fields": {
            "patient_name": None,
            "report_date": None,
            "test_items": [],
            "abnormal_values": [],
            "diagnosis": None,
            "medication": [],
            "follow_up": None,
            "clinical_notes": None,
        },
        "warnings": [],
    }


def _first_group(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def _infer_document_type(text: str) -> str:
    lowered = text.lower()
    if "discharge" in lowered:
        return "discharge_summary"
    if "lab" in lowered or "hemoglobin" in lowered or "creatinine" in lowered:
        return "lab_report"
    if "clinical note" in lowered or "hpi" in lowered:
        return "clinical_note"
    return "unknown"


def _extract_test_items(text: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for match in LAB_VALUE_PATTERN.finditer(text):
        name, value, unit = match.groups()
        items.append(
            {
                "name": name.strip(),
                "value": value,
                "unit": unit or None,
            }
        )
    return items


def _extract_abnormal_lines(text: str) -> list[str]:
    keywords = ("high", "low", "elevated", "decreased", "abnormal", "(h)", "(l)")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return [line for line in lines if any(keyword in line.lower() for keyword in keywords)]


def _rule_based_extract(text: str) -> dict[str, Any]:
    output = _default_output()
    fields = output["extracted_fields"]

    fields["patient_name"] = _first_group(r"(?:patient\s*name|name)\s*[:\-]\s*([^\n]+)", text)
    fields["report_date"] = _first_group(
        r"(?:report\s*date|date)\s*[:\-]\s*([^\n]+)",
        text,
    )
    fields["diagnosis"] = _first_group(r"(?:diagnosis|impression)\s*[:\-]\s*([^\n]+)", text)
    fields["follow_up"] = _first_group(r"(?:follow[- ]?up)\s*[:\-]\s*([^\n]+)", text)

    medication_lines = [
        line.strip("- ").strip()
        for line in text.splitlines()
        if "medication" in line.lower() or "rx:" in line.lower()
    ]
    fields["medication"] = medication_lines
    fields["test_items"] = _extract_test_items(text)
    fields["abnormal_values"] = _extract_abnormal_lines(text)
    fields["clinical_notes"] = text.strip()[:500] or None
    output["document_type"] = _infer_document_type(text)
    output["warnings"].append("Rule-based fallback used; validate extracted fields.")
    return output


def _merge_llm_output(base_output: dict[str, Any], llm_payload: dict[str, Any]) -> dict[str, Any]:
    merged = _default_output()
    merged["document_type"] = base_output.get("document_type")
    merged["extracted_fields"].update(base_output.get("extracted_fields", {}))
    merged["warnings"] = list(base_output.get("warnings", []))

    if not llm_payload:
        return merged

    if isinstance(llm_payload.get("document_type"), str):
        merged["document_type"] = llm_payload["document_type"]

    llm_fields = llm_payload.get("extracted_fields")
    if isinstance(llm_fields, dict):
        merged["extracted_fields"].update(llm_fields)
    elif isinstance(llm_payload, dict):
        merged["extracted_fields"].update(
            {
                key: value
                for key, value in llm_payload.items()
                if key not in {"document_type", "warnings", "extracted_fields"}
            }
        )

    llm_warnings = llm_payload.get("warnings")
    if isinstance(llm_warnings, list):
        merged["warnings"].extend(str(item) for item in llm_warnings)
    elif llm_payload:
        merged["warnings"].append("LLM enrichment applied.")

    return merged


def extract_medical_fields(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if not cleaned:
        output = _default_output()
        output["warnings"].append("Empty text input.")
        return output

    base = _rule_based_extract(cleaned)
    llm_prompt = STRUCTURED_EXTRACTION_PROMPT.format(text=cleaned)
    llm_output = call_llm_json(llm_prompt)
    return _merge_llm_output(base, llm_output)

