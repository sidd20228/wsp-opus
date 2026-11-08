"""Phase 0 check 0.7: the unedited Qwen3.5-0.8B and Astra's p50c edit (rebuilt from the public weights with the
released apply_edit.py, every tensor hash-checked) on MATH-500 and GSM8K with the think35 protocol; if they match the
released numbers, IFEval on both. Writes results/phase0/exp8_check.json.

    .venv/bin/python wsp/exp8/phase0.py [math|ifeval|gpqa|all|report]
"""
import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
from evaluate import evaluate  # noqa: E402

OUT = os.path.join(C.ROOT, "results", "phase0", "exp8_check.json")
MODELS = {"unedited": ("base", "unedited"),
          "p50c": (os.path.join(C.LOGS, "runs", "p50c", "edit"), "astra_p50c")}
TOL = 4.0      # percentage points: "a few points of drift" (vLLM version, batching); beyond it: investigate


def released(model):
    return json.load(open(os.path.join(C.LOGS, "eval", model, "merged.json")))


def run(sets):
    res = {}

    def one(model):
        src, name = MODELS[model]
        res[model] = evaluate(src, sets, name=name, log=lambda m: print(time.strftime("%H:%M:%S"), m, flush=True))

    th = [threading.Thread(target=one, args=(m,)) for m in MODELS]
    for t in th:
        t.start()
    for t in th:
        t.join()
    return res


def paired_vs_released(model, name, s):
    """per-problem paired difference (ours - released) with its standard error (problems as units)."""
    import glob
    import math

    def per(root):
        d = {}
        for f in glob.glob(os.path.join(root, "parts", f"{s}_s*", f"{s}_per_problem.json")):
            for p in json.load(open(f)):
                d[p["id"]] = sum(p["correct"]) / len(p["correct"])
        return d
    a, b = per(os.path.join(HERE, "eval_results", name)), per(os.path.join(C.LOGS, "eval", model))
    d = [a[i] - b[i] for i in a if i in b]
    if len(d) < 2:
        return None
    m = sum(d) / len(d)
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d))
    return {"problems": len(d), "diff_points": round(100 * m, 2), "se_points": round(100 * se, 2),
            "z": round(m / se, 2) if se else None}


def report():
    rows, gpu_h = {}, 0.0
    for model, (_, name) in MODELS.items():
        p = os.path.join(HERE, "eval_results", name, "merged.json")
        if not os.path.exists(p):
            continue
        ours = json.load(open(p))
        gpu_h += ours.get("_meta", {}).get("gpu_hours", 0.0)
        ref = released(model)
        for s, v in ours.items():
            if s.startswith("_"):
                continue
            r = ref[s]
            rows[f"{model}/{s}"] = {
                "acc": round(100 * v["acc"], 1), "acc_original": round(100 * r["acc"], 1),
                "acc_diff": round(100 * (v["acc"] - r["acc"]), 1),
                "cut_share": round(100 * v["cut_share"], 1), "cut_share_original": round(100 * r["cut_share"], 1),
                "mean_tokens": round(v["mean_tokens"]), "mean_tokens_original": round(r["mean_tokens"]),
                "problems": v["problems"], "complete": v["complete"],
                "within_tolerance": abs(100 * (v["acc"] - r["acc"])) <= TOL,
                "paired_vs_released": paired_vs_released(model, name, s)}
    out = {"check": "0.7 Exp 8 eval", "protocol": "think35 (released evaluate.py): chat template with "
           "enable_thinking=True, temperature 0.6, top_p 0.95, top_k 20, presence_penalty 1.5, max 32,768 new tokens, "
           "seed 0; samples MATH-500 4, GSM8K 2, IFEval 2; released shard layout",
           "vllm": C.VLLM_VERSION, "transformers": C.TRANSFORMERS_VERSION, "gpu": "Modal H100",
           "model": f"{C.HF_REPO}@{C.HF_REV} (text-only export, 320 tensors)",
           "p50c_edit": "lm_understand-logs/exp8_math/runs/p50c/edit rebuilt with code/apply_edit.py, both tensor "
                        "hashes OK", "tolerance_points": TOL, "results": rows, "gpu_hours": round(gpu_h, 2),
           "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    return out


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "math"
    if what in ("math", "all"):
        run(["math500", "gsm8k"])
        out = report()
        print(json.dumps(out, indent=1))
    if what in ("ifeval", "all"):
        run(["ifeval"])
        print(json.dumps(report(), indent=1))
    if what == "all":
        run(["gpqa"])
        print(json.dumps(report(), indent=1))
    if what == "gpqa":
        run(["gpqa"])
        print(json.dumps(report(), indent=1))
    if what == "report":
        print(json.dumps(report(), indent=1))
