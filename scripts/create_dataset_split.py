import json

with open("golden_dataset.jsonl", encoding="utf-8") as f:
    all_cases = [json.loads(line) for line in f]

with open("eval_results/step5_remeasurement.json", encoding="utf-8") as f:
    step5_in_scope = json.load(f)["in_scope"]["cases"]

# Refused vs non-refused
refused_questions = set(c["question"] for c in step5_in_scope if c["refused"])
non_refused_questions = set(c["question"] for c in step5_in_scope if not c["refused"])

answerable_cases = []
unanswerable_cases = []
oos_cases = []

for c in all_cases:
    cat = c.get("category")
    q = c.get("question")
    if cat == "out_of_scope":
        oos_cases.append(c)
    elif q in refused_questions:
        unanswerable_cases.append(c)
    else:
        answerable_cases.append(c)

print(f"Total golden dataset cases: {len(all_cases)}")
print(f"  Out of scope cases: {len(oos_cases)}")
print(f"  Unanswerable in-scope cases (missing docs, correctly refused): {len(unanswerable_cases)}")
print(f"  Answerable in-scope cases (evaluated against corpus): {len(answerable_cases)}")

# Write golden_answerable.jsonl (with OOS for refusal test + answerable in-scope)
# Or pure answerable in-scope
with open("golden_answerable.jsonl", "w", encoding="utf-8") as f:
    for c in answerable_cases:
        f.write(json.dumps(c) + "\n")

# Write golden_unanswerable.jsonl (the 26 questions whose documents aren't present)
with open("golden_unanswerable.jsonl", "w", encoding="utf-8") as f:
    for c in unanswerable_cases:
        f.write(json.dumps(c) + "\n")

print("Wrote golden_answerable.jsonl and golden_unanswerable.jsonl successfully.")

