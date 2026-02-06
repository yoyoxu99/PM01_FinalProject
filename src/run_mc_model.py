# src/run_mc_model.py

import json
import argparse
import random
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from mc import build_mc_prompt


# ============================================================
# Utils
# ============================================================

# ---------- Load Data ----------
def load_data(path):
    """Load jsonl file."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# ---------- Format Chat ----------
def format_chat(system, user):
    """Format Chat"""
    return f"""<|begin_of_text|><|system|>
{system}<|end_of_text|><|user|>
{user}<|end_of_text|><|assistant|>
"""


# ============================================================
# Main
# ============================================================

def main(args):

    # ---------- Reproducibility ----------
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---------- Load model ----------
    print("Loading model:", args.model)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()


    # ---------- Load data ----------
    data = load_data(args.data)

    print(f"Loaded {len(data)} samples.")


    # ---------- Output ----------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)


    # ---------- Inference ----------
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

            # ---------- META ----------
            ability = sample.get("ABILITY", "UNKNOWN")
            index = sample.get("INDEX", "UNKNOWN")


            # ---------- Build prompt ----------
            system, user = build_mc_prompt(
                story,
                question,
                choices,
                cot=args.cot,
            )

            prompt = format_chat(system, user)


            # ---------- Tokenize ----------
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
            ).to(model.device)


            max_new = args.max_new_tokens


            # ---------- Multiple runs ----------
            for run_id in range(args.try_times):

                with torch.no_grad():

                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new,
                        do_sample=True,
                        top_p=args.top_p,
                        pad_token_id=tokenizer.eos_token_id,
                    )


                text = tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True,
                )


                # Keep assistant part only
                if "<|assistant|>" in text:
                    text = text.split("<|assistant|>")[-1].strip()


                # ---------- Answer map ----------
                answer_map = {
                    "A": choices[0],
                    "B": choices[1],
                    "C": choices[2],
                    "D": choices[3],
                }


                # ---------- Save ----------
                result = {

                    "idx": i,
                    "run_id": run_id,

                    "output": text,
                    "answer": gold,
                    "map": answer_map,

                    # ===== META =====
                    "ABILITY": ability,
                    "INDEX": index,

                    # ===== Repro =====
                    "prompt": {
                        "system": system,
                        "user": user,
                    },

                    # optional
                    "data": sample,
                }


                fout.write(
                    json.dumps(result, ensure_ascii=False) + "\n"
                )


    print("Done.")
    print("Saved to:", out_path)


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run MC-Probing with stochastic sampling + majority voting"
    )

    parser.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
    )

    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument(
        "--try_times",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--cot",
        action="store_true",
    )

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.90,
        help="Nucleus sampling top_p value",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Max new tokens to generate in each run",
    )

    args = parser.parse_args()

    main(args)