import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
GROUND_TRUTH_PATH = BASE_DIR / "tests" / "ground_truth.json"
OUTPUT_PATH = BASE_DIR / "golden_dataset.jsonl"

# Mapping metadata for the 50 questions: (source_file, category, key_term)
METADATA_MAP = {
    1: ("remote_work_policy.pdf", "conditional", "remote"),
    2: ("benefits_and_leave_policy.pdf", "factual", "20"),
    3: ("travel_and_expense_policy.pdf", "out_of_scope", "information"),
    4: ("travel_and_expense_policy.pdf", "conditional", "30"),
    5: ("remote_work_policy.pdf", "factual", "10:00"),
    6: ("benefits_and_leave_policy.pdf", "factual", "sick"),
    7: ("cybersecurity_policy.pdf", "factual", "encryption"),
    8: ("cybersecurity_policy.pdf", "factual", "incident"),
    9: ("cybersecurity_policy.pdf", "conditional", "device"),
    10: ("benefits_and_leave_policy.pdf", "conditional", "parental"),
    11: ("cybersecurity_policy.pdf", "factual", "password"),
    12: ("code_of_conduct_policy.pdf", "out_of_scope", "information"),
    13: ("cybersecurity_policy.pdf", "conditional", "compliance"),
    14: ("code_of_conduct_policy.pdf", "conditional", "approval"),
    15: ("general_hr_policy.pdf", "conditional", "visitor"),
    16: ("general_hr_policy.pdf", "conditional", "performance"),
    17: ("benefits_and_leave_policy.pdf", "conditional", "reimbursement"),
    18: ("benefits_and_leave_policy.pdf", "conditional", "bereavement"),
    19: ("general_hr_policy.pdf", "out_of_scope", "information"),
    20: ("travel_and_expense_policy.pdf", "factual", "credit"),
    21: ("general_hr_policy.pdf", "out_of_scope", "information"),
    22: ("code_of_conduct_policy.pdf", "conditional", "dress"),
    23: ("code_of_conduct_policy.pdf", "conditional", "media"),
    24: ("cybersecurity_policy.pdf", "factual", "data"),
    25: ("code_of_conduct_policy.pdf", "factual", "whistleblower"),
    26: ("general_hr_policy.pdf", "conditional", "weather"),
    27: ("benefits_and_leave_policy.pdf", "factual", "insurance"),
    28: ("benefits_and_leave_policy.pdf", "conditional", "401"),
    29: ("general_hr_policy.pdf", "factual", "notice"),
    30: ("remote_work_policy.pdf", "conditional", "remote"),
    31: ("general_hr_policy.pdf", "out_of_scope", "information"),
    32: ("cybersecurity_policy.pdf", "conditional", "software"),
    33: ("general_hr_policy.pdf", "factual", "performance"),
    34: ("benefits_and_leave_policy.pdf", "factual", "jury"),
    35: ("travel_and_expense_policy.pdf", "out_of_scope", "information"),
    36: ("general_hr_policy.pdf", "out_of_scope", "information"),
    37: ("code_of_conduct_policy.pdf", "factual", "nda"),
    38: ("benefits_and_leave_policy.pdf", "conditional", "sabbatical"),
    39: ("benefits_and_leave_policy.pdf", "factual", "wellness"),
    40: ("general_hr_policy.pdf", "out_of_scope", "information"),
    41: ("general_hr_policy.pdf", "out_of_scope", "information"),
    42: ("cybersecurity_policy.pdf", "factual", "hardware"),
    43: ("code_of_conduct_policy.pdf", "factual", "harassment"),
    44: ("cybersecurity_policy.pdf", "conditional", "access"),
    45: ("travel_and_expense_policy.pdf", "out_of_scope", "information"),
    46: ("cybersecurity_policy.pdf", "factual", "internet"),
    47: ("benefits_and_leave_policy.pdf", "factual", "health"),
    48: ("benefits_and_leave_policy.pdf", "factual", "volunteer"),
    49: ("cybersecurity_policy.pdf", "factual", "backup"),
    50: ("cybersecurity_policy.pdf", "conditional", "vendor")
}

def convert_dataset():
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    print(f"Loaded {len(ground_truth)} items from {GROUND_TRUTH_PATH}")
    assert len(ground_truth) == 50, f"Expected 50 items, got {len(ground_truth)}"

    lines = []
    category_counts = {}
    for item in ground_truth:
        q_id = item["id"]
        source_file, category, key_term = METADATA_MAP.get(q_id, ("general_policy.pdf", "factual", "policy"))
        category_counts[category] = category_counts.get(category, 0) + 1

        record = {
            "question": item["question"],
            "expected_answer": item["ground_truth"],
            "expected_source_file": source_file,
            "category": category,
            "vars": {
                "question": item["question"],
                "expected_answer": item["ground_truth"],
                "expected_source_file": source_file,
                "category": category
            },
            "assert": [
                {
                    "type": "contains",
                    "value": key_term
                },
                {
                    "type": "llm-rubric",
                    "value": "must cite a source filename in brackets"
                },
                {
                    "type": "latency",
                    "threshold": 5000
                }
            ]
        }
        lines.append(json.dumps(record, ensure_ascii=False))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Successfully generated {OUTPUT_PATH} with {len(lines)} entries.")
    print(f"Category breakdown: {category_counts}")

if __name__ == "__main__":
    convert_dataset()
