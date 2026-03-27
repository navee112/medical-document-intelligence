from fastapi.testclient import TestClient

from app.main import app
from app.services import patient_store, synthesizer

client = TestClient(app)


def test_patient_workflow(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(patient_store, "_data_root", lambda: tmp_path)
    monkeypatch.setattr(synthesizer, "call_llm_json", lambda prompt: {})

    create_resp = client.post("/patients", json={"patient_id": "patient_alpha"})
    assert create_resp.status_code == 200
    assert create_resp.json()["patient_id"] == "patient_alpha"

    upload_resp = client.post(
        "/patients/patient_alpha/upload",
        files=[("files", ("note.txt", b"Patient report: hemoglobin low.", "text/plain"))],
    )
    assert upload_resp.status_code == 200
    assert len(upload_resp.json()["saved_files"]) == 1

    docs_resp = client.get("/patients/patient_alpha/documents")
    assert docs_resp.status_code == 200
    assert len(docs_resp.json()) == 1
    assert docs_resp.json()[0]["filename"] == "note.txt"

    report_resp = client.post(
        "/patients/patient_alpha/report",
        json={"objective": "Produce concise report", "top_k_chunks": 8},
    )
    assert report_resp.status_code == 200
    payload = report_resp.json()
    assert "executive_summary" in payload
    assert payload["sources_used"]

