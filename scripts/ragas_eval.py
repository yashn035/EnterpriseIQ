import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path if env_path.exists() else None)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ragas_eval")

TGI_ENDPOINT_URL = os.getenv("TGI_ENDPOINT_URL", "http://localhost:8080")
API_URL = os.getenv("API_URL", "http://localhost:8000/ask")
GOLDEN_DATASET_PATH = BASE_DIR / "golden_dataset.jsonl"
OUTPUT_PATH = BASE_DIR / "eval_results" / "ragas_results.json"


def check_tgi_reachability(url: str) -> bool:
    """Verifies that the Mistral TGI endpoint is live and accepting requests."""
    try:
        res = requests.get(url, timeout=3)
        return res.status_code in (200, 404, 405)
    except Exception as e:
        logger.error(f"TGI Endpoint '{url}' connection failed: {e}")
        return False


def run_ragas_evaluation(judge: str = "tgi", limit: int = 50):
    logger.info("=======================================================")
    logger.info("   Starting EnterpriseIQ Ragas Evaluation Pipeline     ")
    logger.info("=======================================================")
    logger.info(f"Selected Judge Type: {judge.upper()}")

    judge_llm = None
    if judge == "tgi":
        logger.info(f"Checking Mistral TGI judge endpoint: {TGI_ENDPOINT_URL}")
        if not check_tgi_reachability(TGI_ENDPOINT_URL):
            error_msg = (
                f"\n[FATAL] Mistral TGI judge endpoint is not reachable at '{TGI_ENDPOINT_URL}'.\n"
                f"Stopping evaluation as instructed: do not silently substitute a different judge.\n"
                f"Please ensure your Mistral TGI server is running on {TGI_ENDPOINT_URL} or use --judge openai if OPENAI_API_KEY is configured."
            )
            print(error_msg, file=sys.stderr)
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            failure_report = {
                "status": "ERROR",
                "error": f"TGI judge endpoint unreachable at {TGI_ENDPOINT_URL}",
                "averages": {
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "context_precision": None,
                    "context_recall": None
                },
                "cases": []
            }
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(failure_report, f, indent=2)
            sys.exit(1)

        from langchain_community.llms import HuggingFaceTextGenInference
        judge_llm = HuggingFaceTextGenInference(
            inference_server_url=TGI_ENDPOINT_URL,
            max_new_tokens=512,
            temperature=0.1
        )
    elif judge == "openai":
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            error_msg = (
                "\n[FATAL] OpenAI judge selected via --judge openai, but OPENAI_API_KEY environment variable is not set.\n"
                "Please configure OPENAI_API_KEY in your .env or environment to use OpenAI as the Ragas judge."
            )
            print(error_msg, file=sys.stderr)
            sys.exit(1)
        from langchain_openai import ChatOpenAI
        judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    else:
        raise ValueError(f"Unsupported judge type: {judge}. Supported values are 'tgi' or 'openai'.")

    logger.info(f"Judge LLM initialized ({judge}). Loading test dataset...")
    if not GOLDEN_DATASET_PATH.exists():
        raise FileNotFoundError(f"Golden dataset not found at {GOLDEN_DATASET_PATH}")

    cases = []
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    if limit and limit < len(cases):
        cases = cases[:limit]

    logger.info(f"Querying live API ({API_URL}?debug=true) for {len(cases)} cases...")

    questions = []
    answers = []
    contexts_list = []
    ground_truths = []
    case_breakdowns = []

    for idx, c in enumerate(cases, 1):
        q = c["question"]
        gt = c.get("expected_answer", "")

        try:
            resp = requests.post(f"{API_URL}?debug=true", json={"question": q}, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                ans = data.get("answer", "")
                debug_info = data.get("debug_info") or {}
                ctxs = debug_info.get("contexts") or [ch.get("text", "") for ch in debug_info.get("chunks", [])]
                if not ctxs:
                    ctxs = ["No context retrieved."]
            else:
                ans = "Error querying API"
                ctxs = ["No context retrieved."]
        except Exception as e:
            logger.warning(f"Error querying case #{idx} ('{q[:30]}...'): {e}")
            ans = "Connection error"
            ctxs = ["No context retrieved."]

        questions.append(q)
        answers.append(ans)
        contexts_list.append(ctxs)
        ground_truths.append(gt)

        case_breakdowns.append({
            "id": idx,
            "question": q,
            "answer": ans,
            "ground_truth": gt,
            "contexts": ctxs,
            "category": c.get("category", "factual")
        })

    # Initialize Ragas
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths
    })

    logger.info("Computing Ragas metrics: faithfulness, answer_relevancy, context_precision, context_recall...")
    results = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm
    )

    df_results = results.to_pandas()
    averages = {
        "faithfulness": round(float(df_results["faithfulness"].mean()), 4),
        "answer_relevancy": round(float(df_results["answer_relevancy"].mean()), 4),
        "context_precision": round(float(df_results["context_precision"].mean()), 4),
        "context_recall": round(float(df_results["context_recall"].mean()), 4),
    }

    for idx, row in df_results.iterrows():
        case_breakdowns[idx]["faithfulness"] = round(float(row.get("faithfulness", 0.0)), 4)
        case_breakdowns[idx]["answer_relevancy"] = round(float(row.get("answer_relevancy", 0.0)), 4)
        case_breakdowns[idx]["context_precision"] = round(float(row.get("context_precision", 0.0)), 4)
        case_breakdowns[idx]["context_recall"] = round(float(row.get("context_recall", 0.0)), 4)

    output_data = {
        "status": "SUCCESS",
        "judge": judge,
        "total_cases_evaluated": len(cases),
        "averages": averages,
        "cases": case_breakdowns
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Ragas evaluation completed successfully. Results saved to {OUTPUT_PATH}")
    print("\n" + "=" * 55)
    print("           RAGAS EVALUATION METRIC AVERAGES            ")
    print("=" * 55)
    for m, val in averages.items():
        print(f" {m:<22}: {val}")
    print("=" * 55 + "\n")

    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EnterpriseIQ Ragas Evaluation Suite")
    parser.add_argument("--judge", type=str, choices=["tgi", "openai"], default="tgi",
                        help="Judge LLM engine ('tgi' for local Mistral TGI endpoint, 'openai' for OpenAI API)")
    parser.add_argument("--limit", type=int, default=50, help="Max number of cases to evaluate")
    args = parser.parse_args()

    run_ragas_evaluation(judge=args.judge, limit=args.limit)
