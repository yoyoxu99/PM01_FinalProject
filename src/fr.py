# src/fr.py


# ============================================================
# Prompt Templates
# ============================================================

# -------- System Prompts --------
SYSTEM_FR = """Below is a question with a story. Based on the content of the story and the given question, please infer the answer.
Note:
(1) Please only output the answer in plain text format;
(2) You must provide an answer based on the given information, even if the story does not provide enough details;
(3) Please only output the answer based on the given information, and do not output any other content."""

SYSTEM_FR_COT = """Below is a question with a story. Based on the content of the story and the given question, please infer the answer.
Note:
(1) Please first think step by step, conduct analysis on the question, and finally output the answer in plain text format;
(2) You must provide an answer based on the given information, even if the story does not provide enough details;
(3) Again, you must first output the results of step-by-step reasoning, and finally output the answer. You should not directly output the answer without reasoning."""

# -------- User Prompt --------
USER_FR = """[Story]
{story}

[Question]
{question}

[Answer]
Answer:"""


# ============================================================
# Prompt Builder
# ============================================================

def build_fr_prompt(story, question, cot=False):
    """ Build prompt for Free Response generation.

    If cot=True, use chain-of-thought system prompt.
    """

    system_prompt = SYSTEM_FR_COT if cot else SYSTEM_FR
    user_prompt = USER_FR.format(
        story=story,
        question=question,
    )
    
    # Combine system + user prompt
    full_prompt = system_prompt + "\n\n" + user_prompt
    return full_prompt