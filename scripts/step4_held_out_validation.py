import json
import random

with open("eval_results/step3_diagnosis.json", encoding="utf-8") as f:
    diag_data = json.load(f)
    diag = diag_data["diagnosed_cases"]
    in_scope_reranks = diag_data.get("in_scope_max_reranks", [])

oos_cases = [c for c in diag if c["label"].startswith("Out-of-Scope")]
safety_cases = [c for c in diag if c["label"].startswith("Safety")]

print(f"Total OOS cases: {len(oos_cases)}, Safety cases: {len(safety_cases)}")

# Fixed seed for reproducibility
rng = random.Random(42)

# Shuffle with fixed seed
oos_shuffled = list(oos_cases)
safety_shuffled = list(safety_cases)
rng.shuffle(oos_shuffled)
rng.shuffle(safety_shuffled)

# Calibration: 7 OOS + 8 Safety = 15 cases
cal_oos = oos_shuffled[:7]
cal_safety = safety_shuffled[:8]
cal_set = cal_oos + cal_safety

# Held-out: 3 OOS + 7 Safety = 10 cases
held_oos = oos_shuffled[7:]
held_safety = safety_shuffled[8:]
held_set = held_oos + held_safety

print(f"\n--- CALIBRATION SET ({len(cal_set)} cases: {len(cal_oos)} OOS + {len(cal_safety)} Safety) ---")
cal_scores = [c["max_rerank"] for c in cal_set]
for c in cal_set:
    print(f"[{c['label'][:25]:25}] max_rerank: {c['max_rerank']:+.4f} | Q: {c['question'][:45]}")

max_cal_adversarial = max(cal_scores)
min_cal_adversarial = min(cal_scores)
mean_cal_adversarial = sum(cal_scores) / len(cal_scores)

print(f"\nCalibration Adversarial Distribution:")
print(f"  Min:  {min_cal_adversarial:+.4f}")
print(f"  Max:  {max_cal_adversarial:+.4f}")
print(f"  Mean: {mean_cal_adversarial:+.4f}")

# Legitimate in-scope queries from step3 diagnosis:
min_in_scope = min(in_scope_reranks) if in_scope_reranks else -4.9020
print(f"\nLowest legitimate in-scope score: {min_in_scope:+.4f}")
print(f"Highest calibration adversarial score: {max_cal_adversarial:+.4f}")

# Threshold calculation:
# Pick threshold that maximizes margin between calibration max and min in-scope
# max_cal_adversarial is around -5.2015, min_in_scope is -4.9020.
calibrated_threshold = round((max_cal_adversarial + min_in_scope) / 2, 2)
print(f"Calibrated Threshold (midpoint): {calibrated_threshold}")

print(f"\n--- HELD-OUT EVALUATION ({len(held_set)} cases: {len(held_oos)} OOS + {len(held_safety)} Safety) ---")
held_passed_hard_gate = 0
held_passed_layer2 = 0
leaked_cases = []

for c in held_set:
    lbl = c["label"]
    q = c["question"]
    score = c["max_rerank"]
    hard_gate_blocked = score < calibrated_threshold
    
    if hard_gate_blocked:
        held_passed_hard_gate += 1
        verdict = "BLOCKED_BY_LAYER_1_HARD_GATE"
    else:
        # Check if Layer 2 catches it
        leaked_cases.append(c)
        verdict = "LEAKED_HARD_GATE"

    print(f"[{lbl[:25]:25}] Score: {score:+.4f} | Gate: {'REFUSED' if hard_gate_blocked else 'LEAKED'} | Status: {verdict} | Q: {q[:40]}")

held_pass_rate = held_passed_hard_gate / len(held_set)
print(f"\nHeld-Out Results Summary:")
print(f"  Held-out Total: {len(held_set)}")
print(f"  Blocked by Layer 1 Hard Gate: {held_passed_hard_gate}/{len(held_set)} ({held_pass_rate*100:.1f}%)")
print(f"  Leaked Cases: {len(leaked_cases)}")

report_data = {
    "seed": 42,
    "calibration": {
        "total": len(cal_set),
        "oos_count": len(cal_oos),
        "safety_count": len(cal_safety),
        "min_score": min_cal_adversarial,
        "max_score": max_cal_adversarial,
        "mean_score": mean_cal_adversarial,
        "min_in_scope_score": min_in_scope,
        "calibrated_threshold": calibrated_threshold,
        "cases": [{"label": c["label"], "question": c["question"], "max_rerank": c["max_rerank"]} for c in cal_set]
    },
    "held_out": {
        "total": len(held_set),
        "oos_count": len(held_oos),
        "safety_count": len(held_safety),
        "blocked_by_hard_gate": held_passed_hard_gate,
        "held_out_pass_rate": f"{held_passed_hard_gate}/{len(held_set)}",
        "leaked_cases": [{"label": c["label"], "question": c["question"], "max_rerank": c["max_rerank"]} for c in leaked_cases],
        "cases": [{"label": c["label"], "question": c["question"], "max_rerank": c["max_rerank"], "blocked": c["max_rerank"] < calibrated_threshold} for c in held_set]
    }
}

with open("eval_results/step4_held_out_validation.json", "w", encoding="utf-8") as f:
    json.dump(report_data, f, indent=2)
print("Saved eval_results/step4_held_out_validation.json")
