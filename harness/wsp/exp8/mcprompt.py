"""Multiple-choice prompt and answer reading for MMLU-Pro and GPQA Diamond (in the same format).

The original's mcqa module is not released; evaluate.py describes it as "the official MMLU-Pro reading, the reasoning
cut at the last </think>". This rebuild uses TIGER-AI-Lab/MMLU-Pro's own code:
  prompt  cot_prompt_lib/initial_prompt.txt + format_cot_example (zero-shot: no worked examples; the released texts
          start "The user wants me to answer a multiple-choice question about ...", one question)
  answer  extract_answer -> extract_again -> extract_final (verbatim regexes), applied to the text after the last
          </think> (the whole text when it has none: checked against the released grades of 400 first samples, which
          this reading reproduces exactly, 351/351 closed and 49/49 unclosed)
GPQA Diamond ("the same multiple-choice format, options shuffled per question"): the four options in an order fixed by
the question's Record ID (random.Random(record id).shuffle); the original's shuffle is not released, so which letter
is correct differs from the original's (accuracy is comparable, per-letter results are not). Subject in the prompt:
the question's "High-level domain".
"""
import re

CHOICES = "ABCDEFGHIJ"
INITIAL = ("The following are multiple choice questions (with answers) about {$}. Think step by step and then finish "
           "your answer with \"the answer is (X)\" where X is the correct letter choice.")


def format_question(question, options):
    prompt = "Question:\n" + question + "\n" + "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(CHOICES[i], opt)
    prompt += "Answer: Let's think step by step."
    return prompt


def mmlu_pro_prompt(row):
    return INITIAL.replace("{$}", row["category"]) + "\n" + format_question(row["question"], row["options"])


def gpqa_row(r):
    import random
    opts = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
    opts = [o.strip() for o in opts]
    order = list(range(4))
    random.Random(r["Record ID"]).shuffle(order)
    shuffled = [opts[i] for i in order]
    answer = CHOICES[order.index(0)]
    prompt = INITIAL.replace("{$}", r["High-level domain"].lower()) + "\n" + format_question(r["Question"].strip(), shuffled)
    return {"id": f"gpqa_{r['Record ID']}", "kind": "mc", "answer": answer, "prompt": prompt}


def extract_answer(text):
    m = re.search(r"answer is \(?([A-J])\)?", text)
    if m:
        return m.group(1)
    return extract_again(text)


def extract_again(text):
    m = re.search(r".*[aA]nswer:\s*([A-J])", text)
    if m:
        return m.group(1)
    return extract_final(text)


def extract_final(text):
    m = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", text, re.DOTALL)
    return m.group(0) if m else None


def extract(text):
    """(letter or None, None): the reasoning cut at the last </think>, then the official reading."""
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return extract_answer(text), None
