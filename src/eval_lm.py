# src/eval_lm.py


import argparse
import json
from collections import defaultdict


def load_data(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def accuracy(data, pred_key="pred_raw"):
    correct = sum(d[pred_key] == d["answer"] for d in data)
    return correct / len(data) if data else 0.0


def accuracy_by_ability(data, pred_key="pred_raw"):
    groups = defaultdict(lambda: [0, 0])  # ability -> [correct, total]

    for d in data:
        ability = d.get("ABILITY", "UNKNOWN")
        groups[ability][1] += 1
        groups[ability][0] += int(d[pred_key] == d["answer"])

    accs = {}
    for ability in sorted(groups):
        c, t = groups[ability]
        accs[ability] = (c / t if t > 0 else 0.0, t)
    return accs


def compute_margin(data, score_key="raw_scores"):
    """
    Margin = score(correct answer) - max(score(other options))
    """
    margins = []
    for d in data:
        answer = d["answer"]
        scores = d[score_key]
        answer_idx = None
        # Find index of correct answer in map keys
        for i, key in enumerate(["A", "B", "C", "D"]):
            if key == answer:
                answer_idx = i
                break
        if answer_idx is None:
            # answer label not found, skip margin calculation
            continue
        correct_score = scores[answer_idx]
        other_scores = scores[:answer_idx] + scores[answer_idx + 1:]
        margin = correct_score - max(other_scores) if other_scores else 0.0
        margins.append(margin)
    return sum(margins) / len(margins) if margins else 0.0


def margin_by_ability(data, score_key="raw_scores"):
    groups = defaultdict(list)  # ability -> list of margins

    for d in data:
        ability = d.get("ABILITY", "UNKNOWN")

        answer = d["answer"]
        scores = d[score_key]
        answer_idx = None
        for i, key in enumerate(["A", "B", "C", "D"]):
            if key == answer:
                answer_idx = i
                break
        if answer_idx is None:
            continue
        correct_score = scores[answer_idx]
        other_scores = scores[:answer_idx] + scores[answer_idx + 1:]
        margin = correct_score - max(other_scores) if other_scores else 0.0

        groups[ability].append(margin)

    margins = {}
    for ability in sorted(groups):
        ms = groups[ability]
        margins[ability] = (sum(ms) / len(ms) if ms else 0.0, len(ms))
    return margins


def print_accuracy_group(accs):
    for ability, (acc, count) in accs.items():
        print(f"{ability:50s} (n={count:4d}): {acc:.4f}")


def print_margin_group(margins):
    for ability, (margin, count) in margins.items():
        print(f"{ability:50s} (n={count:4d}): {margin:.4f}")


def main(args):
    data = load_data(args.input)

    print("=" * 60)
    print("LM Probing Evaluation")
    print("=" * 60)

    # Accuracy total
    raw_acc = accuracy(data, "pred_raw")
    norm_acc = accuracy(data, "pred_norm")

    print(f"Total Accuracy (Raw):       {raw_acc:.4f}")
    print(f"Total Accuracy (Normalized): {norm_acc:.4f}\n")

    # Accuracy by ability
    print("Accuracy by Ability (Raw):")
    raw_acc_group = accuracy_by_ability(data, "pred_raw")
    print_accuracy_group(raw_acc_group)
    print()

    print("Accuracy by Ability (Normalized):")
    norm_acc_group = accuracy_by_ability(data, "pred_norm")
    print_accuracy_group(norm_acc_group)
    print()

    # Margin total
    raw_margin = compute_margin(data, "raw_scores")
    norm_margin = compute_margin(data, "normalized_scores")

    print(f"Total Margin (Raw):       {raw_margin:.4f}")
    print(f"Total Margin (Normalized): {norm_margin:.4f}\n")

    # Margin by ability
    print("Margin by Ability (Raw):")
    raw_margin_group = margin_by_ability(data, "raw_scores")
    print_margin_group(raw_margin_group)
    print()

    print("Margin by Ability (Normalized):")
    norm_margin_group = margin_by_ability(data, "normalized_scores")
    print_margin_group(norm_margin_group)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LM probing results with raw and normalized scores")
    parser.add_argument("--input", required=True, help="Path to the LM probing result JSONL file")
    args = parser.parse_args()
    main(args)