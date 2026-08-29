import json
from pathlib import Path
import pytest
from src.ingestion.process_pdfs import process_pdfs


@pytest.mark.unit
def test_process_pdfs_with_dummy_pdf(tmp_path: Path):
    """
    Verifies that process_pdfs recursively scans a directory, parses a dummy PDF,
    creates non-empty chunks, and writes valid JSON output matching dummy content.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_file = tmp_path / "chunks.json"

    dummy_text = "EnterpriseIQ Policy. Remote work is permitted 2 days per week with manager approval."

    # Construct a valid minimal PDF file with known text
    pdf_path = raw_dir / "test_policy.pdf"
    pdf_content = (
        "%PDF-1.4\n"
        "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        "4 0 obj\n<< /Length 150 >>\nstream\n"
        f"BT /F1 12 Tf 50 700 Td ({dummy_text}) Tj ET\n"
        "endstream\nendobj\n"
        "5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        "xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000244 00000 n \n0000000445 00000 n \n"
        "trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n520\n%%EOF\n"
    )
    with open(pdf_path, "w", encoding="latin-1") as f:
        f.write(pdf_content)

    # Execute ingestion process
    chunks = process_pdfs(raw_dir=str(raw_dir), output_file=str(output_file))

    # Assertions
    assert output_file.exists(), "Target chunks.json file was not created."
    assert isinstance(chunks, list), "Expected chunks output to be a list."
    assert len(chunks) > 0, "Chunks list should contain at least 1 extracted chunk."

    chunk = chunks[0]
    assert "id" in chunk, "Chunk is missing 'id' key."
    assert "source_file" in chunk, "Chunk is missing 'source_file' key."
    assert "page_number" in chunk, "Chunk is missing 'page_number' key."
    assert "text" in chunk, "Chunk is missing 'text' key."

    assert chunk["source_file"] == "test_policy.pdf"
    assert chunk["page_number"] == 1
    assert len(chunk["text"]) > 0
    assert "Remote work" in chunk["text"]
