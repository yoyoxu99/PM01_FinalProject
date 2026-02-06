# src/run_lm_model.py


import json
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

from lm import build_lm_prompt, score_options


# ============================================================
# Utils
# ============================================================

# ---------- Load Data ----------
def load_data(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ============================================================
# Main
# ============================================================

def main(args):
    print("Loading model:", args.model)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()

    data = load_data(args.data)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use empty string for unconditional prompt to get baseline scores
    uncond_prompt = ""

    with out_path.open("w", encoding="utf-8") as fout:

        for i, sample in enumerate(tqdm(data)):

            story = sample["STORY"]
            question = sample["QUESTION"]

            choices = [
                sample["OPTION-A"],
                sample["OPTION-B"],
                sample["OPTION-C"],
                sample["OPTION-D"],
            ]

            gold = sample["ANSWER"]
            ability = sample.get("ABILITY", "UNKNOWN")
            index = sample.get("INDEX", "UNKNOWN")

            prompt = build_lm_prompt(story, question)

            # Raw scores under conditional prompt
            raw_scores, pred_raw_ix = score_options(model, tokenizer, prompt, choices)

            # Unconditional baseline scores for normalization
            uncond_scores, _ = score_options(model, tokenizer, uncond_prompt, choices)

            # Normalized scores by subtracting unconditional scores
            normalized_scores = [r - u for r, u in zip(raw_scores, uncond_scores)]
            pred_norm_ix = int(np.argmax(normalized_scores))

            answer_map = dict(zip(["A", "B", "C", "D"], choices))

            result = {
                "idx": i,
                "raw_scores": raw_scores,
                "normalized_scores": normalized_scores,
                "pred_raw": ["A", "B", "C", "D"][pred_raw_ix],
                "pred_norm": ["A", "B", "C", "D"][pred_norm_ix],
                "answer": gold,
                "map": answer_map,
                "ABILITY": ability,
                "INDEX": index,
                "prompt": prompt,
                "uncond_prompt": uncond_prompt,
            }

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("Done.")


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LM probing with raw and normalized scores")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--data", required=True, help="Input JSONL data path")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    args = parser.parse_args()

    main(args)