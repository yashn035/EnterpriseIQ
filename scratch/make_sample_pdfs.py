from pathlib import Path
from pypdf import PdfWriter
import io

def generate_pdf_files():
    """Generates sample valid PDF files in data/raw/ using pypdf."""
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    policies = [
        ("remote_work_policy.pdf", "EnterpriseIQ Remote Work Policy. Full-time employees are eligible for hybrid remote work up to 2 days per week with prior manager approval. Core hours are 10:00 AM to 4:00 PM."),
        ("benefits_and_leave_policy.pdf", "EnterpriseIQ Benefits Policy. Full-time employees receive 20 paid annual leave days per calendar year. Primary caregivers receive 16 weeks of fully paid parental leave."),
        ("cybersecurity_policy.pdf", "EnterpriseIQ Cybersecurity Policy. All employees must complete mandatory cybersecurity training annually in Q1. Security incidents must be reported within 1 hour.")
    ]

    for filename, text in policies:
        filepath = raw_dir / filename
        
        # Build standard PDF structure with text stream
        pdf_raw = (
            "%PDF-1.4\n"
            "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
            "4 0 obj\n<< /Length 200 >>\nstream\n"
            f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET\n"
            "endstream\nendobj\n"
            "5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
            "xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000244 00000 n \n0000000495 00000 n \n"
            "trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n570\n%%EOF\n"
        )
        
        with open(filepath, "w", encoding="latin-1") as f:
            f.write(pdf_raw)
        print(f"Created sample PDF document: {filepath}")

if __name__ == "__main__":
    generate_pdf_files()
