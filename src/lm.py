# src/lm.py

import numpy as np
import torch


# ============================================================
# Prompt Templates
# ============================================================


# -------- User Prompt --------
USER_LM = """{story}

Question: {question}
Answer:"""


# ============================================================
# Prompt Builder
# ============================================================

def build_lm_prompt(story, question):
    """Construct prompt by inserting story and question."""
    return USER_LM.format(story=story, question=question)


# ============================================================
# Language Model Prompting and Option Scoring
# ============================================================

def score_option(model, tokenizer, prompt, option):
    """
    Compute log-probability score of a given option appended to the prompt.

    Args:
        model: Language model.
        tokenizer: Corresponding tokenizer.
        prompt: The prompt text without the option.
        option: Candidate answer option string.

    Returns:
        sum of token log-probabilities for the option tokens.
    """
    full_text = prompt + " " + option
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Shift logits and labels to compute log-probabilities
    logits = outputs.logits[:, :-1]
    labels = inputs["input_ids"][:, 1:]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Gather log-probs of the actual tokens
    token_logprobs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

    # Calculate sum log-prob of the option tokens at the end
    option_len = len(tokenizer(option).input_ids)
    score = token_logprobs[0, -option_len:].sum().item()

    return score


def score_options(model, tokenizer, prompt, choices):
    """
    Score all answer choices and select the one with highest log-probability.

    Args:
        model: Language model.
        tokenizer: Corresponding tokenizer.
        prompt: The prompt text.
        choices: List of candidate answer strings.

    Returns:
        scores: list of log-prob scores per choice.
        pred_ix: index of the choice with highest score.
    """
    scores = [score_option(model, tokenizer, prompt, c) for c in choices]
    pred_ix = int(np.argmax(scores))
    return scores, pred_ix