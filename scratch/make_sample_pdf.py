from pathlib import Path

raw_dir = Path("data/raw")
raw_dir.mkdir(parents=True, exist_ok=True)
pdf_path = raw_dir / "sample.pdf"

t1 = "Remote work policy allows 3 days per week."
t2 = "Employees get 20 paid annual leave days."
t3 = "Cybersecurity policy requires 2FA for all accounts."

pdf_content = (
    "%PDF-1.4\n"
    "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    "2 0 obj\n<< /Type /Pages /Kids [3 0 R 6 0 R 8 0 R] /Count 3 >>\nendobj\n"
    "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    "4 0 obj\n<< /Length 120 >>\nstream\n"
    f"BT /F1 12 Tf 50 700 Td ({t1}) Tj ET\n"
    "endstream\nendobj\n"
    "5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    "6 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 7 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    "7 0 obj\n<< /Length 120 >>\nstream\n"
    f"BT /F1 12 Tf 50 700 Td ({t2}) Tj ET\n"
    "endstream\nendobj\n"
    "8 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 9 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    "9 0 obj\n<< /Length 120 >>\nstream\n"
    f"BT /F1 12 Tf 50 700 Td ({t3}) Tj ET\n"
    "endstream\nendobj\n"
    "xref\n0 10\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000126 00000 n \n0000000255 00000 n \n0000000426 00000 n \n0000000501 00000 n \n0000000630 00000 n \n0000000801 00000 n \n0000000930 00000 n \n"
    "trailer\n<< /Size 10 /Root 1 0 R >>\nstartxref\n1100\n%%EOF\n"
)

with open(pdf_path, "w", encoding="latin-1") as f:
    f.write(pdf_content)

print(f"Created data/raw/sample.pdf")
