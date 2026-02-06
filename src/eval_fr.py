# src/eval_fr.py

import argparse
import json
from collections import defaultdict


def load_data(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def match_generated_to_option(generated_text, options):
    """
    简单关键词匹配：
    在模型生成的答案中查找是否包含选项内容（忽略大小写）。
    返回匹配的选项字母，如"A"，"B"，否则返回None。
    """
    generated_text_lower = generated_text.lower()
    for opt_key, opt_text in options.items():
        if opt_text.lower() in generated_text_lower:
            return opt_key
    return None


def exact_match(pred, gold):
    """简单字母匹配，忽略大小写和空白"""
    return pred.strip().lower() == gold.strip().lower()


def accuracy(data, pred_key="pred_option"):
    correct = sum(
        exact_match(d[pred_key], d["ANSWER"]) for d in data if d.get(pred_key) is not None
    )
    total = sum(1 for d in data if d.get(pred_key) is not None)
    return correct / total if total > 0 else 0.0


def accuracy_by_ability(data, pred_key="pred_option"):
    groups = defaultdict(lambda: [0, 0])  # ability -> [correct, total]

    for d in data:
        ability = d.get("ABILITY", "UNKNOWN")
        if d.get(pred_key) is not None:
            groups[ability][1] += 1
            groups[ability][0] += int(exact_match(d[pred_key], d["ANSWER"]))

    accs = {}
    for ability in sorted(groups):
        c, t = groups[ability]
        accs[ability] = (c / t if t > 0 else 0.0, t)
    return accs


def print_accuracy_group(accs):
    for ability, (acc, count) in accs.items():
        print(f"{ability:50s} (n={count:4d}): {acc:.4f}")


def main(args):
    data = load_data(args.input)

    # 为每条数据增加pred_option字段
    for d in data:
        options = {
            "A": d.get("OPTION-A", ""),
            "B": d.get("OPTION-B", ""),
            "C": d.get("OPTION-C", ""),
            "D": d.get("OPTION-D", ""),
        }
        generated = d.get("GENARATED_ANSWER", "")
        d["pred_option"] = match_generated_to_option(generated, options)

    print("=" * 60)
    print("Free Response Evaluation")
    print("=" * 60)

    total_acc = accuracy(data)
    print(f"Total Exact Match Accuracy: {total_acc:.4f}\n")

    print("Accuracy by Ability:")
    acc_group = accuracy_by_ability(data)
    print_accuracy_group(acc_group)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Free Response answers")
    parser.add_argument("--input", required=True, help="Path to the Free Response results JSONL file")
    args = parser.parse_args()
    main(args)