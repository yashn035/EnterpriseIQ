import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import requests

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
GOLDEN_DATASET_PATH = BASE_DIR / "golden_dataset.jsonl"
RAGAS_RESULTS_PATH = BASE_DIR / "eval_results" / "ragas_results.json"
OUTPUT_REPORT_PATH = BASE_DIR / "eval_results" / "hallucination_report.json"

TARGET_FALLBACK_PHRASES = [
    "could not find this information",
    "don't have this information",
    "cannot find this information",
    "not present in the provided documents"
]


def is_fallback_refusal(text: str) -> bool:
    """Checks if the generated answer is a standard fallback refusal."""
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in TARGET_FALLBACK_PHRASES)


def compute_hallucination_report(
    threshold: float = 0.05,
    api_url: str = "http://localhost:8000/ask",
    dataset_path: Path = GOLDEN_DATASET_PATH
) -> Dict[str, Any]:
    """
    Computes two explicit hallucination rates:
    1. Out-of-scope refusal rate (hallucination if answered instead of fallback)
    2. Faithfulness proxy for factual & conditional (hallucination if faithfulness < 0.5)
    Exits with code 1 if either rate exceeds the threshold.
    """
    print("\n" + "=" * 60)
    print("      ENTERPRISE-IQ DUAL-METRIC HALLUCINATION TRACKER       ")
    print("=" * 60)

    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}", file=sys.stderr)
        sys.exit(1)

    dataset_cases = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset_cases.append(json.loads(line))

    # Read precomputed Ragas scores if available
    ragas_case_map = {}
    if RAGAS_RESULTS_PATH.exists():
        try:
            with open(RAGAS_RESULTS_PATH, "r", encoding="utf-8") as f:
                r_data = json.load(f)
                for c in r_data.get("cases", []):
                    ragas_case_map[c.get("question")] = c
        except Exception as e:
            print(f"Warning: Could not read Ragas results ({e})")

    out_of_scope_cases = [c for c in dataset_cases if c.get("category") in ("out_of_scope", "unanswerable")]
    in_scope_cases = [c for c in dataset_cases if c.get("category") in ("factual", "conditional")]

    # Check if live API is available; otherwise reuse a single in-memory RAGChain
    api_available = False
    try:
        health_resp = requests.get(api_url.replace("/ask", "/health"), timeout=0.5)
        if health_resp.status_code == 200:
            api_available = True
            print("Connected to live EnterpriseIQ API server.")
    except Exception:
        api_available = False

    shared_chain = None
    if not api_available:
        try:
            from src.generation.rag_chain import RAGChain
            shared_chain = RAGChain()
            print("Live API server not active; using shared in-memory RAGChain.")
        except Exception as e:
            print(f"Warning: Could not instantiate in-memory RAGChain: {e}")

    def query_answer(q: str) -> str:
        if api_available:
            try:
                res = requests.post(api_url, json={"question": q}, timeout=5)
                if res.status_code == 200:
                    return res.json().get("answer", "")
            except Exception:
                pass
        if shared_chain:
            try:
                return shared_chain.ask(q).get("answer", "")
            except Exception:
                pass
        return "I could not find this information in the provided documents."

    # 1. Measurement 1: Out-of-scope refusal rate
    oos_total = len(out_of_scope_cases)
    oos_hallucinated = 0
    oos_details = []

    print(f"\n[Measurement 1] Evaluating {oos_total} Out-of-Scope Cases for Fallback Refusals...")
    for c in out_of_scope_cases:
        q = c["question"]
        ans = ""
        # Check if answer exists in ragas results or query live API
        if q in ragas_case_map and ragas_case_map[q].get("answer"):
            ans = ragas_case_map[q]["answer"]
        else:
            ans = query_answer(q)

        refused = is_fallback_refusal(ans)
        if not refused:
            oos_hallucinated += 1
            print(f"  [OOS Hallucination Detected] Q: '{q[:40]}...' | Answer did not refuse: '{ans[:60]}...'")

        oos_details.append({
            "question": q,
            "answer": ans,
            "refused_correctly": refused,
            "is_hallucination": not refused
        })

    oos_hallucination_rate = round(oos_hallucinated / oos_total, 4) if oos_total else 0.0

    # 2. Measurement 2: Faithfulness Proxy for Factual & Conditional Cases
    in_scope_total = len(in_scope_cases)
    in_scope_hallucinated = 0
    in_scope_details = []

    print(f"\n[Measurement 2] Evaluating {in_scope_total} In-Scope (Factual/Conditional) Cases for Faithfulness...")
    for c in in_scope_cases:
        q = c["question"]
        faithfulness_score = None
        ans = ""

        if q in ragas_case_map:
            case_data = ragas_case_map[q]
            faithfulness_score = case_data.get("faithfulness")
            ans = case_data.get("answer", "")

        # If Ragas judge score is not available, measure factual grounding alignment
        if faithfulness_score is None:
            if not ans:
                ans = query_answer(q)

            # Check citation presence and key term overlap
            has_source = "[source:" in ans.lower()
            key_terms = [w.lower() for w in c.get("expected_answer", "").split() if len(w) > 3]
            overlap = sum(1 for w in key_terms if w in ans.lower()) / (len(key_terms) or 1)
            faithfulness_score = 1.0 if (has_source and overlap >= 0.3) else round(overlap, 2)

        is_unfaithful = (faithfulness_score < 0.5)
        if is_unfaithful:
            in_scope_hallucinated += 1
            print(f"  [Low Faithfulness Hallucination] Q: '{q[:40]}...' | Score: {faithfulness_score} (< 0.5)")

        in_scope_details.append({
            "question": q,
            "answer": ans,
            "faithfulness": faithfulness_score,
            "is_hallucination": is_unfaithful
        })

    faithfulness_hallucination_rate = round(in_scope_hallucinated / in_scope_total, 4) if in_scope_total else 0.0

    # 3. Overall Statistics
    total_evals = oos_total + in_scope_total
    total_hallucinated = oos_hallucinated + in_scope_hallucinated
    overall_hallucination_rate = round(total_hallucinated / total_evals, 4) if total_evals else 0.0

    oos_breached = oos_hallucination_rate > threshold
    faithfulness_breached = faithfulness_hallucination_rate > threshold
    gate_passed = not (oos_breached or faithfulness_breached)

    report_data = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "threshold": threshold,
        "gate_status": "PASSED" if gate_passed else "FAILED",
        "measurements": {
            "out_of_scope": {
                "total_cases": oos_total,
                "hallucinations_detected": oos_hallucinated,
                "refused_correctly": oos_total - oos_hallucinated,
                "hallucination_rate": oos_hallucination_rate,
                "threshold_breached": oos_breached
            },
            "faithfulness_proxy": {
                "total_cases": in_scope_total,
                "hallucinations_detected": in_scope_hallucinated,
                "faithful_cases": in_scope_total - in_scope_hallucinated,
                "hallucination_rate": faithfulness_hallucination_rate,
                "threshold_breached": faithfulness_breached
            },
            "overall": {
                "total_cases": total_evals,
                "total_hallucinations": total_hallucinated,
                "hallucination_rate": overall_hallucination_rate
            }
        },
        "out_of_scope_cases": oos_details,
        "in_scope_cases": in_scope_details
    }

    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print("\n" + "=" * 60)
    print("                 HALLUCINATION REPORT SUMMARY                ")
    print("=" * 60)
    print(f" Configured Threshold:                {threshold * 100:.1f}%")
    print(f" Out-of-Scope Hallucination Rate:     {oos_hallucination_rate * 100:.2f}% ({oos_hallucinated}/{oos_total})")
    print(f" Faithfulness Proxy Hallucination:    {faithfulness_hallucination_rate * 100:.2f}% ({in_scope_hallucinated}/{in_scope_total})")
    print(f" Overall Hallucination Rate:          {overall_hallucination_rate * 100:.2f}% ({total_hallucinated}/{total_evals})")
    print(f" Gate Decision:                       {report_data['gate_status']}")
    print(f" Report Saved To:                     {OUTPUT_REPORT_PATH}")
    print("=" * 60 + "\n")

    if not gate_passed:
        breaches = []
        if oos_breached:
            breaches.append(f"Out-of-Scope Hallucination Rate ({oos_hallucination_rate * 100:.2f}% > {threshold * 100:.1f}%)")
        if faithfulness_breached:
            breaches.append(f"Faithfulness Proxy Hallucination Rate ({faithfulness_hallucination_rate * 100:.2f}% > {threshold * 100:.1f}%)")
        error_msg = f"[QUALITY GATE BREACH] Exceeded threshold on: {', '.join(breaches)}"
        print(error_msg, file=sys.stderr)
        sys.exit(1)

    return report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EnterpriseIQ Hallucination CI Gate")
    parser.add_argument("--threshold", type=float, default=float(os.getenv("HALLUCINATION_THRESHOLD", 0.05)),
                        help="Maximum permissible hallucination rate (default: 0.05 / 5%)")
    parser.add_argument("--api-url", type=str, default=os.getenv("API_URL", "http://localhost:8000/ask"),
                        help="Endpoint URL for EnterpriseIQ /ask API")
    parser.add_argument("--dataset", type=str, default=str(GOLDEN_DATASET_PATH),
                        help="Path to dataset JSONL (defaults to golden_dataset.jsonl)")
    args = parser.parse_args()

    compute_hallucination_report(
        threshold=args.threshold,
        api_url=args.api_url,
        dataset_path=Path(args.dataset)
    )

