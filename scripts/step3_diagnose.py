import os
import sys
import json
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.retrieval.local_retriever import LocalRetriever, get_reranker
from src.generation.rag_chain import RAGChain

retriever = LocalRetriever()
reranker = get_reranker()
chain = RAGChain()

# Load 10 Out-of-Scope cases from golden_dataset.jsonl
golden_path = BASE_DIR / "golden_dataset.jsonl"
oos_cases = []
in_scope_cases = []
with open(golden_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            d = json.loads(line)
            if d.get("category") == "out_of_scope":
                oos_cases.append(d["question"])
            elif d.get("category") in ("factual", "conditional"):
                in_scope_cases.append(d["question"])

# Load 15 Safety cases from tests/safety_tests.yaml
import yaml
safety_path = BASE_DIR / "tests" / "safety_tests.yaml"
safety_cases = []
with open(safety_path, "r", encoding="utf-8") as f:
    s_data = yaml.safe_load(f)
    for item in s_data:
        safety_cases.append({
            "desc": item.get("description", "Safety Case"),
            "question": item.get("vars", {}).get("question", "")
        })

print(f"Loaded {len(oos_cases)} OOS cases, {len(safety_cases)} Safety cases, {len(in_scope_cases)} In-Scope cases.")

results_table = []

def diagnose_queries(queries, category_name):
    for idx, item in enumerate(queries, 1):
        if isinstance(item, dict):
            q = item["question"]
            label = f"{category_name}: {item['desc']}"
        else:
            q = item
            label = f"{category_name} #{idx}"

        # 1. Raw retrieval (top 20 candidates before threshold)
        raw_candidates = retriever.retrieve(q, top_k=20, min_threshold=0.0)
        # Cosine / base scores
        base_scores = [round(c.get("base_score", c.get("score", 0.0)), 4) for c in raw_candidates[:5]]
        
        # 2. Reranker scores
        reranked = reranker.rerank(q, raw_candidates, top_k=5)
        rerank_scores = [round(c.get("rerank_score", 0.0), 4) for c in reranked]
        
        # Check if any chunk passed 0.5 gate
        passed_cosine_05 = any(s >= 0.5 for s in base_scores)
        passed_rerank_05 = any(s >= 0.5 for s in rerank_scores)
        
        # Full prompt model received
        context_lines = []
        for c in reranked[:5]:
            src = c.get("source_file", c.get("source", "policy.pdf"))
            pg = c.get("page_number", 1)
            sec = c.get("section_heading", "General Policy")
            txt = c.get("text", "")
            context_lines.append(f"[Source: {src} (Page {pg}, Section: {sec})]:\n{txt}")
        context_str = "\n\n".join(context_lines)
        
        prompt = (
            "You are EnterpriseIQ, a precise policy analyst.\n\n"
            "You will receive a user question and a set of relevant policy excerpts (each with a source).\n\n"
            "Your task is to produce a complete, well-structured answer that:\n"
            "- Directly answers every sub-question or aspect of the user's question.\n"
            "- Uses ALL relevant information from the provided excerpts.\n"
            "- Synthesizes information from multiple excerpts if needed.\n"
            "- Cites the source filename for each claim in brackets [Source: filename.pdf].\n"
            "- If the excerpts do not contain the answer to any part, clearly state that.\n"
            '- If the information is not present at all, say: "I could not find this information in the provided documents."\n\n'
            "Do not omit information. Do not invent information. Do not use external knowledge.\n\n"
            f"=== PROVIDED EXCERPTS ===\n{context_str}\n\n"
            f"=== USER QUESTION ===\n{q}\n\n"
            "=== YOUR ANSWER ==="
        )
        
        results_table.append({
            "label": label,
            "question": q,
            "base_scores": base_scores,
            "max_base": max(base_scores) if base_scores else 0.0,
            "rerank_scores": rerank_scores,
            "max_rerank": max(rerank_scores) if rerank_scores else 0.0,
            "passed_cosine_05": passed_cosine_05,
            "passed_rerank_05": passed_rerank_05,
            "prompt_len": len(prompt),
            "sample_prompt": prompt[:350] + "..."
        })

print("\n--- Diagnosing 10 Out-of-Scope Cases ---")
diagnose_queries(oos_cases, "Out-of-Scope")

print("\n--- Diagnosing 15 Safety Cases ---")
diagnose_queries(safety_cases, "Safety")

# Also collect in-scope distribution for calibration
in_scope_max_reranks = []
in_scope_max_bases = []
for q in in_scope_cases:
    raw = retriever.retrieve(q, top_k=20, min_threshold=0.0)
    bases = [round(c.get("base_score", c.get("score", 0.0)), 4) for c in raw[:5]]
    reranked = reranker.rerank(q, raw, top_k=5)
    rr_scores = [round(c.get("rerank_score", 0.0), 4) for c in reranked]
    in_scope_max_bases.append(max(bases) if bases else 0.0)
    in_scope_max_reranks.append(max(rr_scores) if rr_scores else 0.0)

# Save full results to json
with open("eval_results/step3_diagnosis.json", "w", encoding="utf-8") as f:
    json.dump({
        "diagnosed_cases": results_table,
        "in_scope_max_bases": in_scope_max_bases,
        "in_scope_max_reranks": in_scope_max_reranks
    }, f, indent=2)

print("\nSaved eval_results/step3_diagnosis.json successfully.")

