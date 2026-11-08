"""The editor's forward-pass tool (Experiment 8), called by the conversation driver.

    from wsp.exp8.fwd import forward_pass, pass_message
    text = forward_pass(request_json, k, run_id, kernel=kernel)   # JSON text (indent 1); "ok": true or false
    ok = json.loads(text)["ok"]                                   # false: the pass failed and must NOT be counted
    msg = pass_message(text, k, n_done=used + 1, budget=50)       # the original's turn text for a counted pass

k is the request number (it names the files: /work/gen_<k>.jsonl, /work/gen_<k>_original.jsonl /
/work/gen_<k>_edited.jsonl, /work/gen_<k>[_<model>]_acts/, /work/fwd_<k>/); the original numbered requests including
failed ones, and counted passes separately ("Forward pass 4 of 50 done ...; results in /work/gen_5_original.jsonl").

The summary has the original's fields: generation passes give per model generations, graded, accuracy, accuracy_by_source,
mean_tokens, cut_share, failures {wrong_answer, no_answer_finished, cut_at_max_tokens}, and with "both" the paired
comparison {items, accuracy_difference, standard_error, samples_wrong_to_right, samples_right_to_wrong}; teacher-forced
passes give lengths, files (shapes), edited_tensors, mean_token_logprob. A failed request gives
{"ok": false, "error": "<Type>: <message>", "trace": ..., "s": ...}.

With `kernel` (wsp.exp8.kernel.Kernel) the kernel's /work is committed before the pass and reloaded after it, so the
server reads the current edit and the kernel sees the results.
"""
import json
import math
import os
import sys
import time
import traceback

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

_PRACTICE_IDS = None


def practice_ids():
    global _PRACTICE_IDS
    if _PRACTICE_IDS is None:
        ids = set()
        for f in ("gsm8k_train.jsonl", "math_train.jsonl", "aime.jsonl"):
            for line in open(os.path.join(C.LOGS, "practice", f)):
                ids.add(json.loads(line)["id"])
        _PRACTICE_IDS = ids
    return _PRACTICE_IDS


def server_for(run_id):
    cls = modal.Cls.from_name(C.APP_NAME, "FwdServer")
    vols = {"/data": C.data_vol(), "/model": C.model_vol().read_only(), "/practice": C.practice_vol().read_only(),
            "/work": C.work_vol(run_id)}
    return cls.with_options(volumes=vols)()


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


def _run(req, k, run_id):
    t0 = time.time()
    srv = server_for(run_id)
    if req["type"] == "teacher_forced":
        res = srv.teacher_forced.remote(run_id, req, k)
        res["s"] = round(time.time() - t0, 1)
        return res, {"server_s": res.get("s")}
    model = req["model"]
    if model == "both":
        calls = {w: srv.generate.spawn(run_id, req, k, w, f"gen_{k}_{w}") for w in ("original", "edited")}
        out = {w: c.get() for w, c in calls.items()}
        acts = ({w: out[w]["acts"] for w in out} if req.get("activations") else None)
        res = {"ok": True, "type": "generate", "model": "both",
               "files": {w: out[w]["file"] for w in ("original", "edited")}, "activations": acts,
               "original": model_summary(out["original"]["records"]),
               "edited": model_summary(out["edited"]["records"]),
               "paired": paired(out["original"]["records"], out["edited"]["records"]),
               "s": round(time.time() - t0, 1)}
        info = {w: {"s": out[w]["s"], "gen_s": out[w]["gen_s"], "edited_tensors": out[w]["edited_tensors"]} for w in out}
        return res, info
    o = srv.generate.remote(run_id, req, k, model, f"gen_{k}")
    res = {}
    if o["acts"]:
        res["activations"] = o["acts"]
    res.update({"ok": True, "type": "generate", "model": model, "file": o["file"], **model_summary(o["records"]),
                "s": round(time.time() - t0, 1)})
    return res, {model: {"s": o["s"], "gen_s": o["gen_s"], "edited_tensors": o["edited_tensors"]}}


def forward_pass(request_json: str, k: int, run_id: str, kernel=None, log_dir=None) -> str:
    """Run one forward-pass request; returns the summary as JSON text (indent 1). "ok": false = failed, not counted."""
    t0 = time.time()
    try:
        req = json.loads(request_json) if isinstance(request_json, str) else request_json
        validate(req)
        if kernel is not None:
            kernel.commit()
        res, info = _run(req, k, run_id)
    except Exception as e:  # noqa: BLE001
        res = {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-4000:],
               "s": round(time.time() - t0, 1)}
        info = {}
    if kernel is not None:
        kernel.reload()
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        json.dump({"request": request_json, "result": res, "info": info, "wall_s": time.time() - t0},
                  open(os.path.join(log_dir, f"fwd_{k}.log.json"), "w"), indent=1)
    return json.dumps(res, indent=1)


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
