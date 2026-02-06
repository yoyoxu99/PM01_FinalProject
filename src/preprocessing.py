# src/processing.py

import json
from pathlib import Path


# ============================
# Paths
# ============================

DATA_DIR = Path("data")

IN_PATH = DATA_DIR / "False_Belief_Task.jsonl"

OUT_SUBSET = DATA_DIR / "Index4_5_Location.jsonl"


# ============================
# Filter
# ============================

TARGET_INDEXES = {4, 5}

TARGET_ABILITY_KEY = "Belief: Location false beliefs"


# ============================
# Column Mapping
# ============================

EN_MAP = {
    "能力\nABILITY": "ABILITY",
    "序号\nINDEX": "INDEX",
    "答案\nANSWER": "ANSWER",
    "STORY": "STORY",
    "QUESTION": "QUESTION",
    "OPTION-A": "OPTION-A",
    "OPTION-B": "OPTION-B",
    "OPTION-C": "OPTION-C",
    "OPTION-D": "OPTION-D",
}


# ============================
# Utils
# ============================

def project(obj, mapping):
    """Map Chinese column names to English ones."""
    return {
        new: obj[old]
        for old, new in mapping.items()
        if old in obj
    }


# ============================
# Main
# ============================

def main():

    subset = []

    with IN_PATH.open(encoding="utf-8") as f:

        for line in f:

            obj = json.loads(line)

            # Rename columns
            en = project(obj, EN_MAP)

            # -------- INDEX --------
            try:
                idx = int(en.get("INDEX", -1))
            except ValueError:
                idx = -1

            # -------- ABILITY --------
            ability = en.get("ABILITY", "").strip()

            # -------- Filter --------
            if (
                idx in TARGET_INDEXES
                and TARGET_ABILITY_KEY in ability
            ):
                subset.append(en)

    # ============================
    # Write subset
    # ============================

    with OUT_SUBSET.open("w", encoding="utf-8") as fs:

        for i, obj in enumerate(subset, 1):

            record = {
                "EXP_IDX": i,
                **obj
            }

            fs.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    # ============================
    # Log
    # ============================

    print("Done.")
    print(f"Input  → {IN_PATH}")
    print(f"Output → {OUT_SUBSET}")
    print(f"Samples: {len(subset)}")

    if not subset:
        print("WARNING: No matching samples found!")


if __name__ == "__main__":
    main()