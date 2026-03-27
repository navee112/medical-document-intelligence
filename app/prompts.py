GROUNDED_QA_PROMPT = """
You are a careful clinical assistant.

Answer the question using only the provided context.
If the answer is not supported by the context, say:
"Insufficient evidence in provided documents."

Context:
{context}

Question:
{question}
"""


STRUCTURED_EXTRACTION_PROMPT = """
Extract key clinical fields from the medical text below.

Rules:
- Return valid JSON only
- Do not guess missing values
- Use null if unknown
- Keep facts separate from assumptions
- Preserve abnormal values exactly when present

Text:
{text}
"""


MEDICAL_SUMMARY_PROMPT = """
Summarize the medical text into:
1. Key findings
2. Abnormal values
3. Follow-up concerns

Only use information present in the text.
If uncertain, say so.

Text:
{text}
"""


EXECUTIVE_SYNTHESIS_PROMPT = """
You are a clinical intelligence analyst generating a grounded patient report.

Rules:
- Use only the provided context snippets.
- Return strict JSON only.
- Every finding must include citations from the provided source labels.
- Citations must be specific and preserve page-level traceability.
- Produce at least 2 cross-document insights connecting evidence across different documents.
- Do not invent sources, pages, measurements, or diagnoses.

Return this JSON schema:
{{
  "executive_summary": "string",
  "key_findings": [
    {{"statement": "string", "citations": ["Document A, Page 2"]}}
  ],
  "cross_document_insights": [
    {{"statement": "string", "citations": ["Document A, Page 2", "Document B, Page 4"]}}
  ]
}}

Objective:
{objective}

Context:
{context}
"""
