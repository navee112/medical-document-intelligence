from fastapi.testclient import TestClient

from app.main import app
from app.services import synthesizer

client = TestClient(app)


def test_report_with_inline_documents() -> None:
    payload = {
        "objective": "Synthesize major findings and cross-document links.",
        "documents": [
            {
                "document_id": "Doc_A",
                "text": "Revenue increased by 18 percent in Q4. Risk noted: logistics delays.",
            },
            {
                "document_id": "Doc_B",
                "text": "Q4 demand rose sharply while logistics delays impacted fulfillment times.",
            },
        ],
        "top_k_chunks": 6,
    }

    response = client.post("/report", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "executive_summary" in body
    assert isinstance(body.get("key_findings"), list)
    assert isinstance(body.get("cross_document_insights"), list)
    assert isinstance(body.get("sources_used"), list)


def test_report_fallback_is_clinically_clean(monkeypatch) -> None:
    monkeypatch.setattr(synthesizer, "call_llm_json", lambda prompt: {})

    payload = {
        "objective": "Build patient-focused longitudinal medical report with clean citations.",
        "documents": [
            {
                "document_id": "Test_1-20200101-HEM",
                "text": "Platelets 180 x 10^9/L (150-400). Hemoglobin 13.2 g/dL (12.0-16.0).",
            },
            {
                "document_id": "Test_1-20231011-HEM",
                "text": "Platelets 141 x 10^9/L (150-400) (L). Monocytes 8% 0.3 x 10^9/L (<1.2).",
            },
            {
                "document_id": "Test_1-20250109-HEM",
                "text": "Platelets 163 x 10^9/L (150-450). Film: haematology parameters are essentially normal.",
            },
        ],
        "top_k_chunks": 10,
    }

    response = client.post("/report", json=payload)
    assert response.status_code == 200

    body = response.json()
    findings = body.get("key_findings", [])
    insights = body.get("cross_document_insights", [])

    assert findings
    assert insights
    assert any("platelet" in item["statement"].lower() for item in findings)
    assert all("share recurring themes" not in item["statement"].lower() for item in insights)
    assert all(".." not in item["statement"] for item in findings)
    assert all(len(item.get("citations", [])) >= 1 for item in findings)
