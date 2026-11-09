"""Backend-independent parts of the forward-pass tool: request validation, the per-model summary, the paired
comparison and the original harness's wording of a pass result (used by fwd.py on the Mac and vm_gpu.py on the VM)."""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

_PRACTICE_IDS = None


def practice_ids(practice_dir=None):
    global _PRACTICE_IDS
    if _PRACTICE_IDS is None:
        ids = set()
        for f in ("gsm8k_train.jsonl", "math_train.jsonl", "aime.jsonl"):
            for line in open(os.path.join(practice_dir or C.PRACTICE_DIR, f)):
                ids.add(json.loads(line)["id"])
        _PRACTICE_IDS = ids
    return _PRACTICE_IDS


def validate(req):
    """Cheap checks before any GPU work (the GPU side checks token counts and output size)."""
    if not isinstance(req, dict):
        raise ValueError("the request must be a JSON object")
    t = req.get("type")
    L = C.LIMITS
    if t == "generate":
        model = req.get("model")
        if model not in ("original", "edited", "both"):
            raise ValueError('"model" must be "original", "edited" or "both"')
        probs = req.get("problems") or []
        prompts = req.get("prompts") or []
        if not isinstance(probs, list) or not isinstance(prompts, list):
            raise ValueError('"problems" and "prompts" must be lists')
        unknown = [p for p in probs if p not in practice_ids()]
        if unknown:
            raise ValueError(f"unknown practice ids: {unknown[:5]}")
        items = len(probs) + len(prompts)
        if items == 0:
            raise ValueError("no problems or prompts")
        n = req.get("n", 1)
        if not isinstance(n, int) or n < 1:
            raise ValueError('"n" must be a positive integer')
        cap = L["gen_max_both"] if model == "both" else L["gen_max"]
        if items * n > cap:
            raise ValueError(f"{items} items x {n} samples = {items * n} generations: at most {cap}"
                             + (" per model with \"both\"" if model == "both" else ""))
        mt = req.get("max_tokens", L["max_tokens"])
        if not isinstance(mt, int) or not 1 <= mt <= L["max_tokens"]:
            raise ValueError(f'"max_tokens" must be an integer in 1..{L["max_tokens"]}')
        a = req.get("activations")
        if a is not None:
            if not isinstance(a, dict):
                raise ValueError('"activations" must be an object {"layers": [...], "mlp": true/false}')
            layers = a.get("layers", [])
            if not isinstance(layers, list) or not all(isinstance(x, int) and 0 <= x <= C.N_LAYERS for x in layers):
                raise ValueError("activation layers must be integers in 0..24")
            if not layers:
                raise ValueError("activations: give at least one layer")
    elif t == "teacher_forced":
        if req.get("model") not in ("original", "edited"):
            raise ValueError('teacher_forced "model" must be "original" or "edited"')
        texts = req.get("texts") if "texts" in req else req.get("chats")
        if texts is None or ("texts" in req and "chats" in req):
            raise ValueError('give either "texts" or "chats"')
        if not isinstance(texts, list) or not 1 <= len(texts) <= L["tf_max_texts"]:
            raise ValueError(f"1..{L['tf_max_texts']} texts per pass")
    else:
        raise ValueError('"type" must be "generate" or "teacher_forced"')


def model_summary(records):
    gens = [s for r in records for s in r["samples"]]
    graded = [s for r in records if r["graded"] for s in r["samples"]]
    ok = sum(1 for s in graded if s["correct"])
    by = {}
    for r in records:
        if r["graded"]:
            b = by.setdefault(r["source"], [0, 0])
            b[0] += sum(1 for s in r["samples"] if s["correct"])
            b[1] += len(r["samples"])
    wrong = sum(1 for s in graded if not s["correct"] and not s["cut"] and s["has_pred"])
    noans = sum(1 for s in graded if not s["correct"] and not s["cut"] and not s["has_pred"])
    cut_fail = sum(1 for s in graded if not s["correct"] and s["cut"])
    return {"generations": len(gens), "graded": len(graded), "accuracy": ok / len(graded) if graded else None,
            "accuracy_by_source": {k: round(v[0] / v[1], 4) for k, v in by.items()},
            "mean_tokens": round(sum(s["tokens"] for s in gens) / len(gens)) if gens else 0,
            "cut_share": round(sum(1 for s in gens if s["cut"]) / len(gens), 4) if gens else 0,
            "failures": {"wrong_answer": wrong, "no_answer_finished": noans, "cut_at_max_tokens": cut_fail}}


def paired(orig, edit):
    d, w2r, r2w = [], 0, 0
    for a, b in zip(orig, edit):
        assert a["id"] == b["id"]
        if not a["graded"]:
            continue
        ca = [bool(s["correct"]) for s in a["samples"]]
        cb = [bool(s["correct"]) for s in b["samples"]]
        d.append(sum(cb) / len(cb) - sum(ca) / len(ca))
        w2r += sum(1 for x, y in zip(ca, cb) if not x and y)
        r2w += sum(1 for x, y in zip(ca, cb) if x and not y)
    n = len(d)
    mean = sum(d) / n if n else 0.0
    se = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1) / n) if n > 1 else 0.0
    return {"items": n, "accuracy_difference": round(mean, 4), "standard_error": round(se, 4),
            "samples_wrong_to_right": w2r, "samples_right_to_wrong": r2w}


def pass_message(result_text, k, n_done, budget):
    """The original harness's text for a pass result (n_done counts successful passes including this one)."""
    r = json.loads(result_text)
    if not r.get("ok"):
        return f"The forward pass failed and was not counted: {r.get('error')}"
    if r["type"] == "teacher_forced":
        where = f"/work/fwd_{k}/"
    elif r["model"] == "both":
        where = f"/work/gen_{k}_original.jsonl and /work/gen_{k}_edited.jsonl"
    else:
        where = f"/work/gen_{k}.jsonl"
    return (f"Forward pass {n_done} of {budget} done ({r['type']}); results in {where}. Summary:\n```\n{result_text}\n```\n"
            f"Forward passes left: {budget - n_done}.")
