# src/run_fr_model.py

import json
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from fr import build_fr_prompt


# ============================================================
# Utils
# ============================================================

# ---------- Generate Answer ----------
def generate_answer(model, tokenizer, prompt, max_length=128, top_p=0.9):
    """ Generate answer from model given prompt. """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs.input_ids.shape[1] + max_length,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_p=top_p,
        )
    generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated.strip()

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

    with out_path.open("w", encoding="utf-8") as fout:
        for i, sample in enumerate(tqdm(data)):
            story = sample["STORY"]
            question = sample["QUESTION"]
            # Build prompt with or without CoT (chain-of-thought)
            prompt = build_fr_prompt(story, question, cot=args.cot)
            generated_answer = generate_answer(
                model, tokenizer, prompt, max_length=args.max_length
            )
            result = dict(sample)
            result["GENARATED_ANSWER"] = generated_answer
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("Done.")


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Free Response model generation")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--data", required=True, help="Input JSONL data path")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--cot", action="store_true", help="Use chain-of-thought prompting")
    parser.add_argument("--max_length", type=int, default=128, help="Max generation length")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling top_p value")

    args = parser.parse_args()
    main(args)