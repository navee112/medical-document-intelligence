import re
from datetime import date
from itertools import combinations
from typing import Any

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, REPORT_TOP_K_CHUNKS
from app.prompts import EXECUTIVE_SYNTHESIS_PROMPT
from app.schemas import (
    CrossDocumentInsight,
    DocumentInput,
    Finding,
    ReportRequest,
    ReportResponse,
)
from app.services.llm import call_llm_json
from app.services.parser import parse_document_sections
from app.utils.chunking import chunk_text

COMMON_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}

NOISE_PHRASES = {
    "further testing at additional cost",
    "copyright",
    "all rights reserved",
    "patient copy",
    "this report is not",
    "for clinical correlation",
}

MEASUREMENT_HINTS = {
    "haemoglobin",
    "hemoglobin",
    "platelet",
    "wbc",
    "rbc",
    "neutrophil",
    "lymphocyte",
    "monocyte",
    "eosinophil",
    "basophil",
    "creatinine",
    "urea",
    "glucose",
    "hba1c",
    "cholesterol",
    "triglyceride",
    "bilirubin",
    "albumin",
    "globulin",
    "alt",
    "ast",
    "alp",
    "esr",
    "inr",
    "coagulation",
    "cd4",
    "cd8",
}

MEASUREMENT_BLACKLIST = {
    "page",
    "table",
    "range",
    "reference",
    "result available",
    "sample",
    "lab report",
    "address",
    "telephone",
}

MEASUREMENT_PATTERN = re.compile(
    r"(?P<name>[A-Za-z][A-Za-z0-9 /()%+-]{1,40}?)\s*(?:[:=]\s*|\s+)"
    r"(?P<value>-?\d+(?:\.\d+)?)\s*"
    r"(?P<unit>%|[A-Za-z0-9xX\^/._ -]{0,24})\s*"
    r"(?:\((?P<range>[^)]{1,32})\))?",
    re.IGNORECASE,
)

RANGE_PATTERN = re.compile(r"(?P<low>-?\d+(?:\.\d+)?)\s*-\s*(?P<high>-?\d+(?:\.\d+)?)")
UPPER_BOUND_PATTERN = re.compile(r"<\s*(?P<high>-?\d+(?:\.\d+)?)")
LOWER_BOUND_PATTERN = re.compile(r">\s*(?P<low>-?\d+(?:\.\d+)?)")
DATE_PATTERN = re.compile(r"(20\d{2})[-_/]?(0[1-9]|1[0-2])[-_/]?([0-2]\d|3[01])")


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in COMMON_STOPWORDS and len(token) > 2
    }


def _sanitize_text(text: str) -> str:
    ascii_text = text.encode("ascii", "ignore").decode("ascii")
    ascii_text = re.sub(r"\s+", " ", ascii_text)
    return ascii_text.strip()


def _clean_statement(text: str) -> str:
    statement = _sanitize_text(text)
    statement = statement.strip("-:;,. ")
    statement = re.sub(r"^[^A-Za-z0-9]+", "", statement)
    if statement and statement[-1] not in ".!?":
        statement = f"{statement}."
    return statement


def _is_noisy_text(text: str) -> bool:
    cleaned = _sanitize_text(text)
    if not cleaned:
        return True

    lowered = cleaned.lower()
    if any(phrase in lowered for phrase in NOISE_PHRASES):
        return True

    if len(cleaned) < 25:
        return True

    alpha = sum(1 for ch in cleaned if ch.isalpha())
    punct = sum(1 for ch in cleaned if not ch.isalnum() and not ch.isspace())
    ratio = alpha / max(len(cleaned), 1)
    punct_ratio = punct / max(len(cleaned), 1)

    if ratio < 0.40:
        return True
    if punct_ratio > 0.22:
        return True
    return False


def _extract_doc_date(document_id: str) -> date | None:
    match = DATE_PATTERN.search(document_id)
    if not match:
        return None

    year, month, day = map(int, match.groups())
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _parse_reference_range(raw_range: str) -> tuple[float | None, float | None]:
    if not raw_range:
        return None, None

    candidate = raw_range.replace(" ", "")
    range_match = RANGE_PATTERN.search(candidate)
    if range_match:
        return float(range_match.group("low")), float(range_match.group("high"))

    upper_match = UPPER_BOUND_PATTERN.search(candidate)
    if upper_match:
        return None, float(upper_match.group("high"))

    lower_match = LOWER_BOUND_PATTERN.search(candidate)
    if lower_match:
        return float(lower_match.group("low")), None

    return None, None


def _derive_status(
    value: float,
    low: float | None,
    high: float | None,
    local_text: str,
) -> str:
    if low is not None and value < low:
        return "low"
    if high is not None and value > high:
        return "high"

    lowered = local_text.lower()
    if "(h)" in lowered or " elevated" in lowered or " high" in lowered:
        return "high"
    if "(l)" in lowered or " decreased" in lowered or " low" in lowered:
        return "low"

    if low is not None or high is not None:
        return "normal"
    return "unknown"



def _is_reliable_measurement_label(name: str) -> bool:
    lowered = name.lower()
    if "negative" in lowered or "positive" in lowered:
        return False

    tokens = [token for token in re.split(r"\s+", name.strip()) if token]
    if len(tokens) > 5:
        return False

    hint_matches = [hint for hint in MEASUREMENT_HINTS if hint in lowered]
    if len(hint_matches) > 1:
        return False

    if len(tokens) > 3 and not hint_matches:
        return False

    return True


def _is_measurement_candidate(name: str, unit: str, raw_range: str) -> bool:
    lowered = name.lower()
    if any(blocked in lowered for blocked in MEASUREMENT_BLACKLIST):
        return False

    # Keep fallback reports clean by prioritizing known clinical analytes.
    has_hint = any(hint in lowered for hint in MEASUREMENT_HINTS)
    if not has_hint:
        return False

    has_unit_signal = unit == "%" or "/" in unit or "x" in unit.lower()
    has_range = bool(raw_range)

    return has_hint and (has_unit_signal or has_range)


def _normalize_measurement_name(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", name).strip()
    tokens = cleaned.split()
    if len(tokens) <= 2:
        return cleaned

    lower_tokens = [token.lower() for token in tokens]
    best_index = -1
    best_hint_length = -1

    for index, token in enumerate(lower_tokens):
        for hint in MEASUREMENT_HINTS:
            if hint in token and len(hint) > best_hint_length:
                best_index = index
                best_hint_length = len(hint)

    if best_index < 0:
        return cleaned

    end_index = min(len(tokens), best_index + 3)
    normalized = " ".join(tokens[best_index:end_index]).strip()
    return normalized or cleaned

def _extract_measurements(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    measurements: list[dict[str, Any]] = []
    seen: set[tuple[str, str, float]] = set()

    for chunk in chunks:
        text = _sanitize_text(chunk["text"])
        document_id = str(chunk["document_id"])
        doc_date = _extract_doc_date(document_id)

        for match in MEASUREMENT_PATTERN.finditer(text):
            raw_name = re.sub(r"\s+", " ", match.group("name")).strip(" :-")
            if len(raw_name) < 3:
                continue

            normalized_name = _normalize_measurement_name(raw_name)
            if not _is_reliable_measurement_label(normalized_name):
                continue

            unit = (match.group("unit") or "").strip()
            raw_range = (match.group("range") or "").strip()
            if not _is_measurement_candidate(normalized_name, unit, raw_range):
                continue

            try:
                value = float(match.group("value"))
            except (TypeError, ValueError):
                continue

            ref_low, ref_high = _parse_reference_range(raw_range)
            local_window = text[match.start() : match.end() + 16]
            status = _derive_status(value, ref_low, ref_high, local_window)

            canonical = re.sub(r"[^a-z0-9]+", " ", normalized_name.lower()).strip()
            dedupe_key = (canonical, str(chunk["source"]), round(value, 4))
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            measurements.append(
                {
                    "name": normalized_name,
                    "canonical": canonical,
                    "value": value,
                    "unit": unit,
                    "ref_low": ref_low,
                    "ref_high": ref_high,
                    "status": status,
                    "source": str(chunk["source"]),
                    "document_id": document_id,
                    "doc_date": doc_date,
                }
            )

    return measurements

def _is_informative_chunk(text: str) -> bool:
    cleaned = _sanitize_text(text)
    if len(cleaned) < 60:
        return False
    if _is_noisy_text(cleaned):
        return False
    return True


def _load_sections(document: DocumentInput) -> tuple[list[dict[str, Any]], list[str]]:
    doc_id = document.document_id or "document"
    warnings: list[str] = []

    if document.text:
        text = document.text.strip()
        if not text:
            warnings.append(f"{doc_id}: empty text content.")
            return [], warnings
        return [{"document_id": doc_id, "page": 1, "text": text, "kind": "text"}], warnings

    if not document.path:
        warnings.append(f"{doc_id}: missing both text and path.")
        return [], warnings

    try:
        raw_sections = parse_document_sections(document.path)
    except Exception as exc:
        warnings.append(f"{doc_id}: failed to parse {document.path} ({exc}).")
        return [], warnings

    sections: list[dict[str, Any]] = []
    for section in raw_sections:
        text = str(section.get("text", "")).strip()
        if text:
            sections.append(
                {
                    "document_id": doc_id,
                    "page": int(section.get("page", 1)),
                    "kind": str(section.get("kind", "text")),
                    "text": text,
                }
            )

    if not sections:
        warnings.append(f"{doc_id}: no extractable content found.")
    return sections, warnings


def _build_chunks(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for section in sections:
        pieces = chunk_text(
            text=section["text"],
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )
        if not pieces:
            continue

        for index, piece in enumerate(pieces, start=1):
            source = f'{section["document_id"]}, Page {section["page"]}, Chunk {index}'
            chunks.append(
                {
                    "chunk_id": f'{section["document_id"]}-p{section["page"]}-c{index}',
                    "document_id": section["document_id"],
                    "page": section["page"],
                    "kind": section["kind"],
                    "source": source,
                    "text": piece,
                }
            )
    return chunks


def _rank_chunks(chunks: list[dict[str, Any]], objective: str, top_k: int) -> list[dict[str, Any]]:
    if not chunks:
        return []

    objective_tokens = _tokenize(objective)
    scored: list[tuple[float, dict[str, Any]]] = []
    for chunk in chunks:
        chunk_tokens = _tokenize(chunk["text"])
        overlap_score = 0.0
        if objective_tokens and chunk_tokens:
            overlap_score = len(objective_tokens.intersection(chunk_tokens)) / len(objective_tokens)

        clinical_signal = sum(1 for hint in MEASUREMENT_HINTS if hint in chunk["text"].lower()) * 0.03
        quality_penalty = 0.12 if not _is_informative_chunk(chunk["text"]) else 0.0
        score = overlap_score + clinical_signal + min(len(chunk["text"]), 1000) / 12000 - quality_penalty
        scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    limit = max(1, min(top_k, len(scored)))

    selected: list[dict[str, Any]] = []
    seen_docs: set[str] = set()
    for _, chunk in scored:
        if chunk["document_id"] not in seen_docs:
            selected.append(chunk)
            seen_docs.add(chunk["document_id"])
        if len(selected) >= limit:
            break

    for _, chunk in scored:
        if len(selected) >= limit:
            break
        if chunk not in selected:
            selected.append(chunk)

    return selected[:limit]

def _build_context(chunks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for chunk in chunks:
        cleaned_text = _sanitize_text(chunk["text"])[:1500]
        parts.append(
            "\n".join(
                [
                    f'Source: {chunk["source"]}',
                    f'Kind: {chunk["kind"]}',
                    f"Text: {cleaned_text}",
                ]
            )
        )
    return "\n\n---\n\n".join(parts)


def _best_citations_for_statement(
    statement: str,
    ranked_chunks: list[dict[str, Any]],
    valid_sources: set[str],
    limit: int = 2,
) -> list[str]:
    if not ranked_chunks:
        return []

    statement_tokens = _tokenize(statement)
    scored: list[tuple[float, str]] = []
    for chunk in ranked_chunks:
        source = str(chunk["source"])
        if source not in valid_sources:
            continue

        chunk_tokens = _tokenize(chunk["text"])
        overlap = 0.0
        if statement_tokens and chunk_tokens:
            overlap = len(statement_tokens.intersection(chunk_tokens)) / len(statement_tokens)
        length_boost = min(len(chunk["text"]), 500) / 10000
        scored.append((overlap + length_boost, source))

    if not scored:
        return []

    scored.sort(key=lambda item: item[0], reverse=True)
    output: list[str] = []
    for _, source in scored:
        if source not in output:
            output.append(source)
        if len(output) >= limit:
            break

    return output


def _normalize_citations(citations: Any, valid_sources: set[str]) -> list[str]:
    if not isinstance(citations, list):
        return []
    normalized: list[str] = []
    for citation in citations:
        source = str(citation)
        if source in valid_sources and source not in normalized:
            normalized.append(source)
    return normalized


def _normalize_report_payload(
    payload: dict[str, Any],
    valid_sources: set[str],
    ranked_chunks: list[dict[str, Any]],
) -> tuple[list[Finding], list[CrossDocumentInsight], str]:
    summary = _clean_statement(str(payload.get("executive_summary", "")))
    findings: list[Finding] = []
    insights: list[CrossDocumentInsight] = []

    seen_findings: set[str] = set()
    for item in payload.get("key_findings", []):
        statement = _clean_statement(str(item.get("statement", ""))) if isinstance(item, dict) else ""
        if not statement or _is_noisy_text(statement):
            continue

        lowered = statement.lower()
        if lowered in seen_findings:
            continue
        seen_findings.add(lowered)

        citations = _normalize_citations(item.get("citations", []), valid_sources) if isinstance(item, dict) else []
        if not citations:
            citations = _best_citations_for_statement(statement, ranked_chunks, valid_sources, limit=1)

        findings.append(Finding(statement=statement, citations=citations))

    seen_insights: set[str] = set()
    for item in payload.get("cross_document_insights", []):
        statement = _clean_statement(str(item.get("statement", ""))) if isinstance(item, dict) else ""
        if not statement or _is_noisy_text(statement):
            continue

        lowered = statement.lower()
        if lowered in seen_insights:
            continue
        seen_insights.add(lowered)

        citations = _normalize_citations(item.get("citations", []), valid_sources) if isinstance(item, dict) else []
        if len(citations) < 2:
            fallback = _best_citations_for_statement(statement, ranked_chunks, valid_sources, limit=2)
            for source in fallback:
                if source not in citations:
                    citations.append(source)
                if len(citations) >= 2:
                    break

        insights.append(CrossDocumentInsight(statement=statement, citations=citations[:2]))

    return findings, insights, summary


def _format_value(value: float) -> str:
    return f"{value:g}"


def _format_range(low: float | None, high: float | None) -> str:
    if low is not None and high is not None:
        return f"{low:g}-{high:g}"
    if high is not None:
        return f"<{high:g}"
    if low is not None:
        return f">{low:g}"
    return ""


def _fallback_sentence_candidates(chunks: list[dict[str, Any]]) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()

    for chunk in chunks:
        cleaned = _sanitize_text(chunk["text"])
        for sentence in re.split(r"(?<=[.!?])\s+", cleaned):
            statement = _clean_statement(sentence)
            if not statement:
                continue
            if _is_noisy_text(statement):
                continue
            if len(statement) < 40 or len(statement) > 220:
                continue
            if not re.search(r"\d", statement) and not any(
                hint in statement.lower() for hint in MEASUREMENT_HINTS
            ):
                continue

            lowered = statement.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            candidates.append((statement, str(chunk["source"])))

    return candidates


def _fallback_key_findings(chunks: list[dict[str, Any]], limit: int = 5) -> list[Finding]:
    findings: list[Finding] = []
    measurements = _extract_measurements(chunks)

    def sort_key(item: dict[str, Any]) -> tuple[int, date | None, float]:
        doc_date = item.get("doc_date")
        return (1 if doc_date else 0, doc_date, abs(item["value"]))

    abnormal = [item for item in measurements if item["status"] in {"low", "high"}]
    abnormal.sort(key=sort_key, reverse=True)

    seen_analytes: set[str] = set()
    for item in abnormal:
        analyte_key = item["canonical"]
        if analyte_key in seen_analytes:
            continue
        seen_analytes.add(analyte_key)

        analyte = item["name"]
        value_text = _format_value(item["value"])
        unit_text = f" {item['unit']}" if item["unit"] else ""
        ref_text = _format_range(item["ref_low"], item["ref_high"])

        direction = "below" if item["status"] == "low" else "above"
        if ref_text:
            statement = (
                f"{analyte} was {value_text}{unit_text}, {direction} the reference range {ref_text}."
            )
        else:
            statement = f"{analyte} was {value_text}{unit_text} and flagged as {item['status']}."

        findings.append(Finding(statement=_clean_statement(statement), citations=[item["source"]]))
        if len(findings) >= limit:
            return findings

    normal_with_ranges = [item for item in measurements if item["status"] == "normal"]
    normal_with_ranges.sort(key=sort_key, reverse=True)
    for item in normal_with_ranges:
        analyte_key = item["canonical"]
        if analyte_key in seen_analytes:
            continue
        seen_analytes.add(analyte_key)

        ref_text = _format_range(item["ref_low"], item["ref_high"])
        if not ref_text:
            continue

        analyte = item["name"]
        value_text = _format_value(item["value"])
        unit_text = f" {item['unit']}" if item["unit"] else ""
        statement = f"{analyte} remained within reference range ({value_text}{unit_text}; ref {ref_text})."
        findings.append(Finding(statement=_clean_statement(statement), citations=[item["source"]]))
        if len(findings) >= limit:
            return findings

    for statement, source in _fallback_sentence_candidates(chunks):
        findings.append(Finding(statement=statement, citations=[source]))
        if len(findings) >= limit:
            break

    return findings


def _fallback_cross_document_insights(chunks: list[dict[str, Any]]) -> list[CrossDocumentInsight]:
    insights: list[CrossDocumentInsight] = []
    measurements = _extract_measurements(chunks)

    by_analyte: dict[str, list[dict[str, Any]]] = {}
    for item in measurements:
        by_analyte.setdefault(item["canonical"], []).append(item)

    for analyte, items in by_analyte.items():
        if len(items) < 2:
            continue

        unique_docs = {item["document_id"] for item in items}
        if len(unique_docs) < 2:
            continue

        items.sort(key=lambda entry: (entry["doc_date"] or date.min, entry["source"]))
        first = items[0]
        last = items[-1]

        if abs(last["value"] - first["value"]) <= 0.01:
            trend = "remained broadly stable"
        elif last["value"] > first["value"]:
            trend = "increased"
        else:
            trend = "decreased"

        unit_text = first["unit"] or last["unit"]
        unit_suffix = f" {unit_text}" if unit_text else ""
        statement = (
            f"{first['name']} {trend} across the document timeline "
            f"({_format_value(first['value'])}{unit_suffix} to {_format_value(last['value'])}{unit_suffix})."
        )

        if last["status"] in {"high", "low"}:
            statement += f" The latest value is {last['status']} versus its reference interval."

        citations = [first["source"], last["source"]]
        insights.append(CrossDocumentInsight(statement=_clean_statement(statement), citations=citations))

        if len(insights) >= 2:
            return insights

    # Fallback timeline insight when numeric trends are sparse.
    doc_representatives: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        doc_id = str(chunk["document_id"])
        if doc_id not in doc_representatives:
            doc_representatives[doc_id] = chunk

    timeline = sorted(
        doc_representatives.values(),
        key=lambda item: (_extract_doc_date(str(item["document_id"])) or date.min, str(item["document_id"])),
    )

    if len(timeline) >= 2:
        earliest = timeline[0]
        latest = timeline[-1]
        statement = (
            f"The record set spans from {earliest['document_id']} to {latest['document_id']}; "
            "compare latest laboratory values against earlier baselines to confirm persistent or evolving risks."
        )
        insights.append(
            CrossDocumentInsight(
                statement=_clean_statement(statement),
                citations=[str(earliest["source"]), str(latest["source"])],
            )
        )

    if len(insights) < 2 and len(timeline) >= 3:
        middle = timeline[len(timeline) // 2]
        statement = (
            f"Intermediate report {middle['document_id']} links early and late observations, "
            "supporting longitudinal interpretation rather than single-report conclusions."
        )
        insights.append(
            CrossDocumentInsight(
                statement=_clean_statement(statement),
                citations=[str(timeline[0]["source"]), str(middle["source"])],
            )
        )

    return insights


def _build_summary(chunks: list[dict[str, Any]], findings: list[Finding]) -> str:
    document_count = len({str(chunk["document_id"]) for chunk in chunks})
    measurement_findings = sum(
        1 for finding in findings if any(hint in finding.statement.lower() for hint in MEASUREMENT_HINTS)
    )

    summary = (
        f"Synthesized {len(chunks)} grounded evidence chunks across {document_count} documents. "
        f"Generated {len(findings)} key findings"
    )
    if measurement_findings:
        summary += f", including {measurement_findings} laboratory-focused signals"
    summary += "."
    return summary


def generate_executive_report(request: ReportRequest) -> ReportResponse:
    warnings: list[str] = []
    sections: list[dict[str, Any]] = []
    for document in request.documents:
        doc_sections, doc_warnings = _load_sections(document)
        sections.extend(doc_sections)
        warnings.extend(doc_warnings)

    chunks = _build_chunks(sections)
    if not chunks:
        return ReportResponse(
            executive_summary="No extractable document content was available.",
            grounded=False,
            warnings=warnings or ["No chunks available for synthesis."],
        )

    top_k = request.top_k_chunks or REPORT_TOP_K_CHUNKS
    ranked_chunks = _rank_chunks(chunks, request.objective, top_k=top_k)
    context = _build_context(ranked_chunks)
    valid_sources = {str(chunk["source"]) for chunk in ranked_chunks}

    prompt = EXECUTIVE_SYNTHESIS_PROMPT.format(objective=request.objective, context=context)
    payload = call_llm_json(prompt)
    key_findings, cross_insights, summary = _normalize_report_payload(
        payload,
        valid_sources,
        ranked_chunks,
    )

    if not key_findings:
        warnings.append("LLM findings unavailable or invalid; used fallback findings.")
        key_findings = _fallback_key_findings(ranked_chunks)

    if len(cross_insights) < 2:
        warnings.append("Cross-document insights were insufficient; used fallback synthesis.")
        fallback_insights = _fallback_cross_document_insights(chunks)
        for insight in fallback_insights:
            if len(cross_insights) >= 2:
                break
            if insight.statement.lower() not in {item.statement.lower() for item in cross_insights}:
                cross_insights.append(insight)

    key_findings = key_findings[:5]
    cross_insights = cross_insights[:3]

    if not summary:
        summary = _build_summary(ranked_chunks, key_findings)

    used_sources = {
        citation
        for item in key_findings
        for citation in item.citations
    }
    used_sources.update(
        citation
        for item in cross_insights
        for citation in item.citations
    )

    if not used_sources:
        used_sources = valid_sources

    grounded = bool(used_sources)
    return ReportResponse(
        executive_summary=summary,
        key_findings=key_findings,
        cross_document_insights=cross_insights,
        sources_used=sorted(used_sources),
        grounded=grounded,
        warnings=warnings,
    )













