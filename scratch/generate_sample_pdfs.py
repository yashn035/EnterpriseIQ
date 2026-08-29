import os
from pathlib import Path

def create_sample_pdf_text():
    """Generates sample raw text documents if PDF generation tools are unavailable."""
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    sample_policies = {
        "remote_work_policy.txt": """EnterpriseIQ Remote Work & Flexible Hours Policy
Section 1: Work Location & Eligibility
Full-time employees are eligible for hybrid remote work up to 2 days per week with prior manager approval.
Core working hours during which employees must be online and available are 10:00 AM to 4:00 PM local time.
Working remotely from outside the country of employment is permitted for up to 30 days per calendar year with explicit HR approval.""",
        
        "benefits_and_leave_policy.txt": """EnterpriseIQ Benefits & Leave Policy
Section 2: Annual & Parental Leave
Full-time employees receive 20 paid annual leave days per calendar year. A maximum of 5 unused annual leave days can be carried over into the next year.
Primary caregivers are entitled to 16 weeks of fully paid parental leave following the birth or adoption of a child.
Employees receive 5 paid business days of bereavement leave for immediate family members.""",
        
        "cybersecurity_and_byod_policy.txt": """EnterpriseIQ Cybersecurity & Device Policy
Section 3: Security & Incident Response
All employees must complete mandatory cybersecurity awareness training annually in Q1.
Personal devices must be enrolled in Mobile Device Management (MDM) before accessing company data.
Security incidents must be reported immediately to security-incident@enterpriseiq.com within 1 hour of discovery."""
    }
    
    for filename, content in sample_policies.items():
        filepath = raw_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created sample policy file: {filepath}")

if __name__ == "__main__":
    create_sample_pdf_text()
