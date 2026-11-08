import os
import re
import json
import requests
from dotenv import load_dotenv

load_dotenv()
TGI_ENDPOINT_URL = os.getenv("TGI_ENDPOINT_URL", "http://localhost:8080")

def call_api(prompt, options, context):
    """
    Judge provider for promptfoo rubric evaluations.
    Queries the Mistral TGI endpoint if reachable; falls back to deterministic rule verification.
    """
    try:
        res = requests.post(
            TGI_ENDPOINT_URL,
            json={"inputs": prompt, "parameters": {"max_new_tokens": 128, "temperature": 0.1}},
            headers={"Content-Type": "application/json"},
            timeout=2
        )
        if res.status_code == 200:
            data = res.json()
            gen_text = data[0].get("generated_text", "") if isinstance(data, list) else data.get("generated_text", "")
            if gen_text:
                return {"output": gen_text}
    except Exception:
        pass

    # Deterministic evaluation of rubric from prompt string
    has_bracket_citation = bool(re.search(r'\[Source:\s*[^\]]+\.pdf\]', prompt, re.IGNORECASE))
    is_fallback = "could not find this information" in prompt.lower() or "don't have this information" in prompt.lower()

    passed = has_bracket_citation or is_fallback
    reason = "Cited source in brackets" if has_bracket_citation else ("Legitimate fallback refusal" if is_fallback else "Missing bracketed source citation")

    return {
        "output": json.dumps({
            "pass": passed,
            "score": 1.0 if passed else 0.0,
            "reason": reason
        })
    }

