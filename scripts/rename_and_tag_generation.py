import json
import os
from pathlib import Path

with open("golden_answerable.jsonl", "r", encoding="utf-8") as f:
    cases = [json.loads(line) for line in f]

# We know from Step 2 audit exactly which 6 questions have answers in the corpus:
# 1. remote work standard policy
# 2. paid annual leave days
# 3. core working hours
# 4. cybersecurity incident reporting protocol
# 5. paid parental leave for primary caregivers
# 6. mandatory cybersecurity compliance training

answerable_questions = {
    "What is the standard remote work policy for employees?",
    "How many paid annual leave days are granted to full-time staff?",
    "What are the core hours when employees must be available?",
    "What is the protocol for reporting a suspected cybersecurity incident?",
    "How long is paid parental leave for primary caregivers?",
    "What is the requirement for mandatory employee compliance training?"
}

for c in cases:
    q = c.get("question")
    c["corpus_contains_answer"] = (q in answerable_questions)

with open("golden_reached_generation.jsonl", "w", encoding="utf-8") as f:
    for c in cases:
        f.write(json.dumps(c) + "\n")

# Remove the misnamed file
if os.path.exists("golden_answerable.jsonl"):
    os.remove("golden_answerable.jsonl")

print(f"Created golden_reached_generation.jsonl with {len(cases)} cases.")
print(f"  Corpus contains answer (true): {sum(1 for c in cases if c['corpus_contains_answer'])}")
print(f"  Corpus contains answer (false): {sum(1 for c in cases if not c['corpus_contains_answer'])}")

