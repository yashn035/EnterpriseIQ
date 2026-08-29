from pathlib import Path

raw_dir = Path("data/raw")
raw_dir.mkdir(parents=True, exist_ok=True)
pdf_path = raw_dir / "sample_policy.pdf"

text1 = "Remote work policy allows 3 days per week."
text2 = "Leave policy provides 20 annual days."

pdf_content = (
    "%PDF-1.4\n"
    "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    "2 0 obj\n<< /Type /Pages /Kids [3 0 R 6 0 R] /Count 2 >>\nendobj\n"
    "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    "4 0 obj\n<< /Length 120 >>\nstream\n"
    f"BT /F1 12 Tf 50 700 Td ({text1}) Tj ET\n"
    "endstream\nendobj\n"
    "5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    "6 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 7 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    "7 0 obj\n<< /Length 120 >>\nstream\n"
    f"BT /F1 12 Tf 50 700 Td ({text2}) Tj ET\n"
    "endstream\nendobj\n"
    "xref\n0 8\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000120 00000 n \n0000000249 00000 n \n0000000420 00000 n \n0000000495 00000 n \n0000000624 00000 n \n"
    "trailer\n<< /Size 8 /Root 1 0 R >>\nstartxref\n795\n%%EOF\n"
)

with open(pdf_path, "w", encoding="latin-1") as f:
    f.write(pdf_content)

print(f"Created dummy PDF at {pdf_path}")
