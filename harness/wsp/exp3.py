"""Experiment 3: which corpus a character-level transformer was trained on, from its four middle blocks only.

Opus gets the original message.txt (identical for all three networks). Each reply with a ```python block is run in a
fresh Modal sandbox process with W (the four blocks, float32) preloaded (600 s, output cut at 20,000 characters,
at most 100 runs), and the output is sent back as "OUTPUT of run k of at most 100:" in a code fence, exactly as the
original harness did. A reply without a code block is the final answer.

  python wsp/exp3.py run <language> <rep>        one conversation (language: python | romaji | english)
  python wsp/exp3.py score                       score every finished conversation
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wsp import conversation, sandbox  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "lm_understand-logs", "exp3_language")
OUT = os.path.join(ROOT, "runs", "opus55", "exp3_language")
MAX_RUNS, TIMEOUT, MAX_OUT = 100, 600, 20_000
PRELUDE = "import numpy as np\nW = {k: v for k, v in np.load('/weights/blocks.npz').items()}\n"
LABELS = ("previous", "local", "broad", "copy", "unclear")
HEADS = [f"L{l}H{h}" for l in range(4) for h in range(6)]


def run(language, rep):
    first = open(os.path.join(LOGS, "runs", "python_r0", "message.txt")).read()
    folder = os.path.join(OUT, f"{language}_r{rep}")
    conv = conversation.Conversation(folder, "xhigh", first)
    image = sandbox.base_image().add_local_file(
        os.path.join(LOGS, "networks", language, "blocks.npz"), "/weights/blocks.npz")

    def handle(text, c):
        code = conversation.block(text, "python")
        if code is None:
            return None
        k = c.count("runs", 1)
        if k > MAX_RUNS:
            return (f"You have used all {MAX_RUNS} runs. Reply WITHOUT a code block, with your final answer in the "
                    "order given in the question.")
        out, secs = sandbox.run_fresh(code, image, PRELUDE, timeout=TIMEOUT, max_out=MAX_OUT)
        c.tool_call({"run": k, "s": secs, "code": code, "out": out})
        c.log(f"run {k} ({len(code):,} chars of code, {len(out):,} chars out, {secs} s)")
        return f"OUTPUT of run {k} of at most {MAX_RUNS}:\n```\n{out}\n```"

    return conv.run(handle)


def parse_final(text):
    t = text.replace("**", "")
    labels = {}
    for m in re.finditer(r"(?m)^\W*(L[0-3]H[0-5])\W*[:—–-]+\s*`?\s*(" + "|".join(LABELS) + r")\b", t,
                         re.I):
        labels.setdefault(m.group(1), m.group(2).lower())
    def last(pat):  # the answer block comes last; earlier prose can start with "Corpus: ..." too
        ms = list(re.finditer(pat, t))
        return ms[-1] if ms else None
    corpus = last(r"(?im)^\W*corpus\s*:\s*`?(python|romaji|english)\b")
    order = last(r"(?im)^\W*order\s*:\s*`?(as is|shuffled)")
    probs = last(r"(?im)^\W*probabilities\s*:\s*(.*python\s*=.*)$")
    p = {}
    if probs:
        for name in ("python", "romaji", "english"):
            m = re.search(rf"{name}\s*=\s*([0-9.]+)", probs.group(1))
            if m:
                p[name] = float(m.group(1))
    return {"head_labels": labels, "corpus_guess": corpus.group(1).lower() if corpus else None,
            "order_guess": order.group(1).lower() if order else None, "probabilities": p}


def truth(language):
    h = json.load(open(os.path.join(LOGS, "eval", "heads.json")))[f"{language}_L4_H6_d384_s0"]
    return h, sorted(k for k, v in h.items() if v["induct"] > 0.2)


def score_folder(folder, language):
    final = open(os.path.join(folder, "final_answer.md")).read()
    s = parse_final(final)
    st0 = json.load(open(os.path.join(folder, "state.json")))
    # pseudo tool calls with no ```python fence: no code ran, answer not based on the weights (deviations.md)
    # no code ever ran: the answer cannot rest on the weights (pseudo tool calls, or analysis written without a run)
    s["format_failure"] = st0["counters"].get("runs", 0) == 0
    h, true_copy = truth(language)
    named = sorted(k for k, v in s["head_labels"].items() if v == "copy")
    tp = len(set(named) & set(true_copy))
    st = json.load(open(os.path.join(folder, "state.json")))
    return {**s, "language": language, "corpus_correct": s["corpus_guess"] == language,
            "copy_named": named, "copy_true": true_copy, "copy_exact": named == true_copy,
            "copy_precision": tp / len(named) if named else None, "copy_recall": tp / len(true_copy),
            "labels_given": len(s["head_labels"]), "runs": st["counters"].get("runs", 0), "turns": st["turns"]}


def score_all():
    rows = []
    for folder in sorted(glob.glob(os.path.join(OUT, "*_r*"))):
        if not os.path.exists(os.path.join(folder, "final_answer.md")):
            continue
        lang = os.path.basename(folder).rsplit("_r", 1)[0]
        rows.append({"run": os.path.basename(folder), **score_folder(folder, lang)})
    os.makedirs(os.path.join(ROOT, "results", "phase2"), exist_ok=True)
    json.dump(rows, open(os.path.join(ROOT, "results", "phase2", "exp3_scores.json"), "w"), indent=1)
    for r in rows:
        print(r["run"], r["corpus_guess"], r["probabilities"], "copy named", r["copy_named"], "true", r["copy_true"],
              "runs", r["runs"])
    return rows


if __name__ == "__main__":
    if sys.argv[1] == "run":
        ok = run(sys.argv[2], int(sys.argv[3]))
        print("done" if ok else "stopped")
    elif sys.argv[1] == "score":
        score_all()
    elif sys.argv[1] == "check":   # Phase 0: our parser on Astra's final answers
        ref = {d["id"].replace("_L4_H6_d384_s0", ""): d for d in json.load(open(os.path.join(LOGS, "eval",
                                                                                        "scores.json")))}
        for rid, d in sorted(ref.items()):
            s = parse_final(open(os.path.join(LOGS, "runs", rid, "final_answer.md")).read())
            print(rid, "labels match:", s["head_labels"] == d["head_labels"], "corpus:",
                  s["corpus_guess"] == d["corpus_guess"], "probs:", s["probabilities"])
