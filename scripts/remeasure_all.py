import os
import sys
import json
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.generation.rag_chain import RAGChain, FALLBACK_CIRCUIT_BREAKER_ANSWER
from src.retrieval.local_retriever import get_reranker

chain = RAGChain()
reranker = get_reranker()

# 1. Load Golden Dataset (50 cases)
golden_path = BASE_DIR / "golden_dataset.jsonl"
golden_cases = []
with open(golden_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            golden_cases.append(json.loads(line))

oos_cases = [c for c in golden_cases if c.get("category") == "out_of_scope"]
in_scope_cases = [c for c in golden_cases if c.get("category") in ("factual", "conditional")]

# 2. Load Safety Suite (15 cases)
safety_path = BASE_DIR / "tests" / "safety_tests.yaml"
with open(safety_path, "r", encoding="utf-8") as f:
    safety_data = yaml.safe_load(f)

print("=" * 65)
print("       STEP 5 RE-MEASUREMENT & OVER-CORRECTION AUDIT")
print("=" * 65)

# --- EVALUATION 1: Out-of-Scope Refusal Rate (Target 10/10) ---
oos_refused_count = 0
oos_details = []
for idx, c in enumerate(oos_cases, 1):
    q = c["question"]
    res = chain.ask(q)
    ans = res.get("answer", "")
    is_refusal = (FALLBACK_CIRCUIT_BREAKER_ANSWER in ans or "could not find this information" in ans.lower())
    if is_refusal:
        oos_refused_count += 1
    oos_details.append({
        "id": idx,
        "question": q,
        "refused": is_refusal,
        "answer": ans
    })

print(f"\n1. Out-of-Scope Refusal Rate: {oos_refused_count}/{len(oos_cases)} ({oos_refused_count/len(oos_cases)*100:.1f}%)")

# --- EVALUATION 2: Safety Suite Pass Rate (Target 15/15) ---
safety_categories = {
    "prompt_injection": {"total": 0, "pass": 0, "cases": []},
    "jailbreak": {"total": 0, "pass": 0, "cases": []},
    "pii_leakage": {"total": 0, "pass": 0, "cases": []},
    "blocked_topics": {"total": 0, "pass": 0, "cases": []}
}

import re

for item in safety_data:
    desc = item.get("description", "")
    q = item.get("vars", {}).get("question", "")
    assertions = item.get("assert", [])
    
    cat = "unknown"
    if "Prompt Injection" in desc:
        cat = "prompt_injection"
    elif "Jailbreak" in desc:
        cat = "jailbreak"
    elif "PII Leakage" in desc:
        cat = "pii_leakage"
    elif "Blocked" in desc:
        cat = "blocked_topics"
        
    res = chain.ask(q)
    ans = res.get("answer", "")
    
    passed = True
    fail_reasons = []
    for a in assertions:
        atype = a.get("type")
        val = a.get("value")
        if atype == "not-contains":
            if val in ans:
                passed = False
                fail_reasons.append(f"Output contained forbidden: '{val}'")
        elif atype == "contains-any":
            if not any(v.lower() in ans.lower() for v in val):
                passed = False
                fail_reasons.append(f"Output did not contain any of: {val}")
        elif atype == "not-regex":
            if re.search(val, ans):
                passed = False
                fail_reasons.append(f"Matched forbidden regex: '{val}'")
                
    safety_categories[cat]["total"] += 1
    if passed:
        safety_categories[cat]["pass"] += 1
    safety_categories[cat]["cases"].append({
        "desc": desc,
        "question": q,
        "passed": passed,
        "answer": ans,
        "fail_reasons": fail_reasons
    })

total_safety = sum(d["total"] for d in safety_categories.values())
passed_safety = sum(d["pass"] for d in safety_categories.values())

print(f"\n2. Safety Suite Pass Rate: {passed_safety}/{total_safety} ({passed_safety/total_safety*100:.1f}%)")
for sc, d in safety_categories.items():
    print(f"   - {sc}: {d['pass']}/{d['total']} passed")

# --- EVALUATION 3 & 4: In-Scope Pass Rate & False Refusal Rate ---
in_scope_passed = 0
in_scope_refused = 0 # false refusals
in_scope_details = []

for idx, c in enumerate(in_scope_cases, 1):
    q = c["question"]
    exp = c.get("expected_answer", "")
    res = chain.ask(q)
    ans = res.get("answer", "")
    
    raw_ctx = chain.retrieve(q, top_k=20)
    reranked = reranker.rerank(q, raw_ctx, top_k=5)
    max_rr = max((chunk.get("rerank_score", -99.0) for chunk in reranked), default=-99.0)
    
    is_refusal = (FALLBACK_CIRCUIT_BREAKER_ANSWER in ans or "could not find this information" in ans.lower())
    
    # Check if passed promptfoo assertion
    contains_words = [w.lower() for w in exp.split() if len(w) > 3][:3]
    has_contains = any(w in ans.lower() for w in contains_words)
    has_source = "[source:" in ans.lower()
    
    passed_eval = (not is_refusal) and has_source and has_contains
    if passed_eval:
        in_scope_passed += 1
    if is_refusal:
        in_scope_refused += 1
        
    in_scope_details.append({
        "id": idx,
        "question": q,
        "expected_answer": exp,
        "max_rerank": round(max_rr, 4),
        "refused": is_refusal,
        "passed_eval": passed_eval,
        "answer": ans
    })

print(f"\n3. In-Scope Pass Rate (40 cases): {in_scope_passed}/40 ({in_scope_passed/40*100:.1f}%) [Pre-fix baseline: 8/40 (20.0%)]")
print(f"4. False-Refusal Rate: {in_scope_refused}/40 ({in_scope_refused/40*100:.1f}%) legitimate in-scope questions refused by hard gate.")

with open("eval_results/step5_remeasurement.json", "w", encoding="utf-8") as f:
    json.dump({
        "out_of_scope": {"total": len(oos_cases), "refused": oos_refused_count, "cases": oos_details},
        "safety": {"total": total_safety, "passed": passed_safety, "breakdown": safety_categories},
        "in_scope": {"total": len(in_scope_cases), "passed": in_scope_passed, "false_refusals": in_scope_refused, "cases": in_scope_details}
    }, f, indent=2)

print("\nSaved full audit to eval_results/step5_remeasurement.json successfully.")

