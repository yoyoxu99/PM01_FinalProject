# src/mc.py

import json
from collections import Counter
import random
import re


# ============================================================
# Voting & Parsing
# ============================================================

# ---------- Majority Voting ----------
def majority_vote(preds, rng):
    """
    Majority voting with random tie-breaking (controlled by rng).
    """
    cnt = Counter(preds)

    if not cnt:
        return None

    max_freq = max(cnt.values())

    top = [
        k for k, v in cnt.items()
        if v == max_freq
    ]

    return rng.choice(top)


# ---------- Extract answer option (A/B/C/D) from model output text ----------
def extract_mc_answer(text):
    if not text:
        return None

    text = text.upper()

    # priority patterns
    for k in ("[[A]]","[[B]]","[[C]]","[[D]]",
              "[A]","[B]","[C]","[D]"):
        if k in text:
            return k.strip("[]")

    # fallback: last valid char
    return next((c for c in reversed(text) if c in "ABCD"), None)


# ============================================================
# Aggregation
# ============================================================

def aggregate_mc_results(path, try_times=5, seed=42):

    rng = random.Random(seed)

    # ---------- Load ----------
    with open(path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    if not data:
        raise ValueError("Empty result file!")

    # ---------- Size ----------
    n = max(d["idx"] for d in data) + 1

    preds = [[] for _ in range(n)]
    golds = [None] * n

    abilities = {}
    indices = {}

    raw_results = []

    # ---------- Collect ----------
    for d in data:

        i = d["idx"]

        # -------- Save meta once --------
        if i not in abilities:

            abilities[i] = d.get("ABILITY", "UNKNOWN")
            indices[i] = d.get("INDEX", "UNKNOWN")

        # -------- Parse --------
        letter = extract_mc_answer(d.get("output"))

        if letter in d.get("map", {}):
            preds[i].append(letter)
        else:
            letter = None

        gold = d.get("answer")

        raw_results.append({
            "idx": i,

            "ABILITY": abilities[i],
            "INDEX": indices[i],

            "gold": gold,
            "pred": letter,

            "parsed": letter is not None,
            "correct": letter == gold,
        })

        golds[i] = golds[i] or gold


    # ---------- Sanity check ----------
    for i, p in enumerate(preds):

        if len(p) != try_times:
            print(f"[WARN] idx={i}: {len(p)}/{try_times} runs")


    # ---------- Voting ----------
    voted_results = []

    for i in range(n):

        final = majority_vote(preds[i], rng)

        voted_results.append({

            "idx": i,

            "ABILITY": abilities.get(i, "UNKNOWN"),
            "INDEX": indices.get(i, "UNKNOWN"),

            "gold": golds[i],
            "pred": final,

            "parsed": final is not None,
            "correct": final == golds[i],
        })
    return raw_results, voted_results


# ============================================================
# Prompt Templates
# ============================================================

# -------- System Prompts --------
SYSTEM_MC = \
"""Below is a multiple-choice question with a story and serveral answer options. Based on the content of the story and the given question, please infer the most likely answer and output the answer index.
Note:
(1) Please only output the most likely answer index in the format: [[Answer Index]], for example, if the most likely answer option is 'A. Handbag', then output '[[A]]';
(2) You must choose one of the given answer options 'A, B, C, D' as the most likely answer, regardless of whether the story provides enough information. If you think there is not enough information in the story to choose an answer, please randomly output one of "[[A]]", "[[B]]", "[[C]]", or "[[D]]";
(3) Please only output the most likely answer index based on the given information, and do not output any other content."""


SYSTEM_MC_COT = \
"""Below is a multiple-choice question with a story and serveral answer options. Based on the content of the story and the given question, please infer the most likely answer and output the answer index.
Note:
(1) Please first think step by step, conduct analysis on the answers to the questions, and finally output the most likely answer index in the format: [[Answer Index]], for example, if the most likely answer option is 'A. Handbag', then output '[[A]]';
(2) You must choose one of the given answer options 'A, B, C, D' as the most likely answer, regardless of whether the story provides enough information. If you think there is not enough information in the story to choose an answer, please randomly output one of "[[A]]", "[[B]]", "[[C]]", or "[[D]]";
(3) Again, you must first output the results of step-by-step reasoning, and finally output the most likely answer index. You should not directly output the answer index."""


# -------- User Prompt --------
USER_MC_4CHOICES = \
"""[Story]
{story}

[Question]
{question}

[Candidate Answers]
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}"""


# ============================================================
# Prompt Builder
# ============================================================

def build_mc_prompt(story, question, choices, cot=False):
    """
    Build MC prompt (with or without CoT).
    """

    system = SYSTEM_MC_COT if cot else SYSTEM_MC

    user = USER_MC_4CHOICES.format(
        story=story,
        question=question,
        choice_a=choices[0],
        choice_b=choices[1],
        choice_c=choices[2],
        choice_d=choices[3]
    )
    return system, user


