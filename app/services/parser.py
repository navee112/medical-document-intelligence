from pathlib import Path
from typing import Any

from app.config import OCR_ENABLED, OCR_RENDER_SCALE, TESSERACT_CMD


def parse_raw_text(text: str) -> str:
    return text.strip()


def parse_text_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def _table_to_text(table: list[list[str | None]]) -> str:
    rows: list[str] = []
    for row in table:
        values = [(cell or "").strip() for cell in row]
        if any(values):
            rows.append(" | ".join(values))
    return "\n".join(rows).strip()


def _extract_pdf_tables(path: str) -> dict[int, list[str]]:
    tables_by_page: dict[int, list[str]] = {}
    try:
        import pdfplumber
    except ImportError:
        return tables_by_page

    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_tables = page.extract_tables() or []
            page_text_tables: list[str] = []
            for table in page_tables:
                table_text = _table_to_text(table)
                if table_text:
                    page_text_tables.append(table_text)
            if page_text_tables:
                tables_by_page[page_number] = page_text_tables
    return tables_by_page


def _extract_pdf_ocr(path: str, pages: set[int]) -> dict[int, str]:
    if not OCR_ENABLED or not pages:
        return {}

    try:
        import pypdfium2 as pdfium
        import pytesseract
    except ImportError:
        return {}

    tesseract_cmd = TESSERACT_CMD
    if not tesseract_cmd:
        fallback = Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe"
        if fallback.exists():
            tesseract_cmd = str(fallback)

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    ocr_text_by_page: dict[int, str] = {}
    try:
        document = pdfium.PdfDocument(path)
    except Exception:
        return {}

    try:
        for page_number in sorted(pages):
            page_index = page_number - 1
            if page_index < 0 or page_index >= len(document):
                continue

            try:
                page = document[page_index]
                bitmap = page.render(scale=OCR_RENDER_SCALE)
                image = bitmap.to_pil()
                text = pytesseract.image_to_string(image).strip()
                if text:
                    ocr_text_by_page[page_number] = text
            except Exception:
                continue
    finally:
        try:
            document.close()
        except Exception:
            pass

    return ocr_text_by_page


def parse_pdf_sections(path: str) -> list[dict[str, Any]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is required for PDF parsing. Install pypdf first.") from exc

    reader = PdfReader(path)
    tables_by_page = _extract_pdf_tables(path)

    raw_text_by_page: dict[int, str] = {}
    missing_text_pages: set[int] = set()
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            raw_text_by_page[page_number] = text
        else:
            missing_text_pages.add(page_number)

    ocr_text_by_page = _extract_pdf_ocr(path, missing_text_pages)

    sections: list[dict[str, Any]] = []
    for page_number in range(1, len(reader.pages) + 1):
        text = raw_text_by_page.get(page_number)
        if text:
            sections.append({"page": page_number, "text": text, "kind": "text"})
        elif page_number in ocr_text_by_page:
            sections.append(
                {
                    "page": page_number,
                    "text": ocr_text_by_page[page_number],
                    "kind": "ocr",
                }
            )

        for table_index, table_text in enumerate(tables_by_page.get(page_number, []), start=1):
            sections.append(
                {
                    "page": page_number,
                    "text": f"Table {table_index}:\n{table_text}",
                    "kind": "table",
                }
            )

    return sections


def parse_document_sections(path: str) -> list[dict[str, Any]]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf_sections(path)

    text = parse_text_file(path)
    if not text:
        return []
    return [{"page": 1, "text": text, "kind": "text"}]


def parse_pdf_text(path: str) -> str:
    sections = parse_pdf_sections(path)
    return "\n\n".join(section["text"] for section in sections).strip()


def parse_document(path: str) -> str:
    sections = parse_document_sections(path)
    return "\n\n".join(section["text"] for section in sections).strip()
