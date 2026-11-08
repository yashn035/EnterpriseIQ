import json

with open("golden_unanswerable.jsonl", "r", encoding="utf-8") as f:
    cases = [json.loads(line) for line in f]

for c in cases:
    c["category"] = "unanswerable"
    c["expected_fallback"] = True

with open("golden_unanswerable.jsonl", "w", encoding="utf-8") as f:
    for c in cases:
        f.write(json.dumps(c) + "\n")

print(f"Updated {len(cases)} cases in golden_unanswerable.jsonl with category='unanswerable'")

