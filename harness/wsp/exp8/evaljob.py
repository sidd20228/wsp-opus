"""One evaluation job on the GPU: the released code/evaluate.py `run` (protocol think35), ported with as few changes as
possible. Differences, all forced by what is not released:
  - benchmark files come from /data/bench (built by app.prepare_bench in the original's row order, answers checked);
  - multiple choice: mcprompt.extract (TIGER-Lab's MMLU-Pro reading; the original mcqa module is not released);
  - IFEval: the official checkers as shipped in lm-eval 0.4.13 (vendor/ifeval), same call sequence as grade_other;
  - the GPU-memory cap file of the original's shared pool does not exist here;
  - first_samples.jsonl keeps the first sample of EVERY problem (the release did the same; the code says [:20]).
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vendor import mathqa as Q  # noqa: E402
import common as C  # noqa: E402

BENCH = C.BENCH_DIR
SAMPLING = {"think35": C.SAMPLING_THINK35,
            # the model card's non-thinking setting (the plan's context row)
            "nothink35": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 8192,
                          "seed": 0}}


def lenient_pred(text):
    import re
    found = []
    for pat in (r"[Ff]inal [Aa]nswer\s*(?:is)?\s*[:=]?\s*\$*([^\n$]+)", r"[Tt]he (?:final )?answer is\s*[:=]?\s*\$*([^\n$]+)",
                r"\b[Aa]nswer\s*[:=]\s*\$*([^\n$]+)"):
        found += [(m.start(1), m.group(1).strip().strip("*").rstrip(".").strip()) for m in re.finditer(pat, text)]
    for _, t in sorted(found, reverse=True):
        if re.search(r"[0-9\\]", t):
            return t
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def lenient_correct(text, pred, good, answer):
    if good or pred is not None:
        return bool(good)
    lp = lenient_pred(text)
    if lp is None:
        return False
    import re
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", lp)
    return bool(Q.is_correct(lp, answer) or (nums and Q.is_correct(nums[-1].replace(",", ""), answer)))


def grade_other(r, text, thinking):
    """(correct, pred, extra) of a non-math item, as the released evaluate.grade_other."""
    if r["kind"] == "mc":
        import mcprompt as M
        pred, _ = M.extract(text)
        return pred == r["answer"], pred, {}
    from vendor.ifeval import instructions_registry as IR
    resp = text.split("</think>")[-1].strip() if "</think>" in text else ("" if thinking else text.strip())
    follow = []
    for iid, kw in zip(r["instruction_id_list"], r["kwargs"]):
        ins = IR.INSTRUCTION_DICT[iid](iid)
        ins.build_description(**{k: v for k, v in (kw or {}).items() if v is not None})
        args = ins.get_instruction_args()
        if args and "prompt" in args:
            ins.build_description(prompt=r["prompt"])
        follow.append(bool(resp) and bool(ins.check_following(resp)))
    return all(follow), None, {"inst_ok": sum(follow), "inst_total": len(follow)}


def load_set(name):
    return [json.loads(line) for line in open(os.path.join(BENCH, f"{name}.jsonl"))]


def run(ckpt, out, sets, protocol="think35", shard=None, llm=None):
    sampling = SAMPLING[protocol]
    thinking = protocol in ("think", "think35")
    os.makedirs(out, exist_ok=True)
    sp_path = os.path.join(out, "scores.json")
    job = {"protocol": protocol, "ckpt": ckpt, "ckpt_mtime": os.path.getmtime(os.path.join(ckpt, "config.json")),
           "shard": shard}
    if os.path.exists(sp_path):           # finished before (a re-submitted job): nothing to do
        prev = json.load(open(sp_path))
        if prev.get("_job") == job and all(s in prev for s in sets) and "_protocol" in prev:
            return prev
    from vllm import LLM, SamplingParams
    import vllm
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ckpt)
    if llm is None:       # the VM runner passes one engine for all shards of a checkpoint
        llm = LLM(model=ckpt, dtype="bfloat16", gpu_memory_utilization=C.GPU_MEM,
                  max_model_len=sampling["max_tokens"] + 2048, seed=0)
    res = json.load(open(sp_path)) if os.path.exists(sp_path) else {}
    if res and res.get("_job") != job:
        os.replace(sp_path, sp_path + f".stale_{int(time.time())}")
        res = {}
    res["_job"] = job
    for name in sets:
        if name in res:
            continue
        t0 = time.time()
        rows = load_set(name)
        if shard:
            i, k = map(int, shard.split("/"))
            rows = rows[i::k]
        prompts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"] if r.get("kind") else Q.build_prompt(r["problem"])}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=thinking) for r in rows]
        sp = SamplingParams(n=C.SAMPLES[name], **sampling)
        outs = llm.generate(prompts, sp)
        bad = sum(1 for o in outs for x in o.outputs if x.finish_reason not in ("stop", "length"))
        if bad:
            raise RuntimeError(f"{bad} generations ended without finishing or reaching max_tokens; not saved")
        per, tot, ok, cut, lens, noans, oklen, inst_ok, inst_total = [], 0, 0, 0, [], 0, 0, 0, 0
        for r, o in zip(rows, outs):
            c = []
            for s in o.outputs:
                if r.get("kind"):
                    good, pred, extra = grade_other(r, s.text, thinking)
                    inst_ok += extra.get("inst_ok", 0)
                    inst_total += extra.get("inst_total", 0)
                    c.append(int(good))
                    cut += s.finish_reason == "length"
                    noans += r["kind"] == "mc" and pred is None and s.finish_reason != "length"
                    oklen += good
                    lens.append(len(s.token_ids))
                    continue
                pred, _ = Q.extract(s.text)
                good = Q.is_correct(pred, r["answer"])
                c.append(int(good))
                cut += s.finish_reason == "length"
                noans += pred is None and s.finish_reason != "length"
                oklen += lenient_correct(s.text, pred, good, r["answer"])
                lens.append(len(s.token_ids))
            per.append({"id": r["id"], "correct": c, "pred_first": None if r.get("kind") else Q.extract(o.outputs[0].text)[0]})
            tot += len(c)
            ok += sum(c)
        res[name] = {"acc": ok / tot, "n_problems": len(rows), "samples": C.SAMPLES[name], "cut_share": cut / tot,
                     "mean_tokens": sum(lens) / len(lens), "s": round(time.time() - t0),
                     "ok": ok, "total": tot, "cut": cut, "tokens_sum": sum(lens), "shard": shard, "no_answer_finished": noans,
                     "ok_lenient": oklen, **({"inst_ok": inst_ok, "inst_total": inst_total} if inst_total else {})}
        json.dump(per, open(os.path.join(out, f"{name}_per_problem.json"), "w"))
        import gzip
        with gzip.open(os.path.join(out, f"{name}_first_samples.jsonl.gz"), "wt") as f:
            for r, o in (zip(rows, outs) if name != "gpqa" else ()):     # GPQA texts restate questions: not kept
                f.write(json.dumps({"id": r["id"], "answer": r.get("answer"), "text": o.outputs[0].text}) + "\n")
        json.dump(res, open(os.path.join(out, "scores.json"), "w"), indent=1)
        print(name, json.dumps(res[name]), flush=True)
    res["_protocol"] = {"name": protocol, "sampling": sampling, "samples": C.SAMPLES, "thinking": thinking,
                        "prompt": Q.INSTRUCTION, "ckpt": ckpt, "vllm": vllm.__version__,
                        "gpu": _gpu_name()}
    json.dump(res, open(os.path.join(out, "scores.json"), "w"), indent=1)
    return res


def _gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        return None
