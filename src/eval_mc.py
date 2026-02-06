# src/eval_mc.py

import argparse
from pathlib import Path

from mc import aggregate_mc_results


# ============================================================
# Utils
# ============================================================

def accuracy(results):

    parsed = [r for r in results if r.get("parsed", True)]

    if not parsed:
        return 0.0

    return sum(r["correct"] for r in parsed) / len(parsed)


def accuracy_by_ability(results):

    groups = {}

    for r in results:

        if not r.get("parsed", True):
            continue

        ability = r.get("ABILITY")

        if not ability or ability == "UNKNOWN":
            continue

        if ability not in groups:
            groups[ability] = {"correct": 0, "total": 0}

        groups[ability]["correct"] += int(r["correct"])
        groups[ability]["total"] += 1


    accs = {}

    for k, v in groups.items():

        if v["total"] == 0:
            accs[k] = (0.0, 0)
        else:
            accs[k] = (v["correct"] / v["total"], v["total"])

    return accs


# ============================================================
# Main Evaluation
# ============================================================

def evaluate(path, try_times, seed):

    raw, voted = aggregate_mc_results(
        path,
        try_times=try_times,
        seed=seed,
    )

    raw_acc = accuracy(raw)
    vote_acc = accuracy(voted)

    parse_rate = sum(r.get("parsed", True) for r in raw) / len(raw)

    raw_group = accuracy_by_ability(raw)
    vote_group = accuracy_by_ability(voted)

    return raw_acc, vote_acc, parse_rate, raw_group, vote_group


# ============================================================
# CLI
# ============================================================

def main(args):

    files = [Path(p) for p in args.inputs]

    print("=" * 60)
    print("MC Evaluation")
    print("=" * 60)
    print(f"Try times : {args.try_times}")
    print(f"Seed      : {args.seed}")
    print("-" * 60)

    for path in files:

        (
            raw_acc,
            vote_acc,
            parse_rate,
            raw_group,
            vote_group,
        ) = evaluate(
            path,
            args.try_times,
            args.seed,
        )

        name = path.stem

        print(f"[{name}]")
        print(f"  Raw Acc    : {raw_acc:.4f}")
        print(f"  Vote Acc   : {vote_acc:.4f}")
        print(f"  Parse Rate : {parse_rate:.4f}")
        print()

        # ---------- Raw ----------
        print("  Accuracy by ABILITY (Raw):")

        if not raw_group:
            print("    (No valid samples)")
        else:
            for k in sorted(raw_group):

                acc, n = raw_group[k]
                print(f"    {k:50s} (n={n:4d}): {acc:.4f}")

        print()

        # ---------- Vote ----------
        print("  Accuracy by ABILITY (Vote):")

        if not vote_group:
            print("    (No valid samples)")
        else:
            for k in sorted(vote_group):

                acc, n = vote_group[k]
                print(f"    {k:50s} (n={n:4d}): {acc:.4f}")

        print("-" * 60)

    print("=" * 60)


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluate MC-Probing results"
    )

    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--try_times",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    main(args)