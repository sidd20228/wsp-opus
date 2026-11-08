"""Benchmark evaluation of an (edited) Qwen3.5-0.8B with the released protocol (code/evaluate.py, protocol think35),
one Modal H100 job per shard (the released shard counts), merged with the released merge.py.

    from wsp.exp8.evaluate import evaluate
    merged = evaluate("base", ["math500", "gsm8k"], name="unedited")
    merged = evaluate("lm_understand-logs/exp8_math/runs/p50c/edit", ["math500"], name="astra_p50c")
    merged = evaluate("work:<run_id>", C.EIGHT_SETS, name="opus_p50_r0")     # a run's /work/edited

source: "base" | a local folder with edit.npz + manifest.json (a released edit diff: rebuilt with apply_edit.py and
hash-checked) | a local folder of <tensor name>.npy (a submission, cast to bfloat16 as build_ckpt.py) | "work:<run>".
Results: <local_dir>/parts/<set>_s<i>of<k>/{scores.json, <set>_per_problem.json, <set>_first_samples.jsonl.gz},
<local_dir>/merged.json (acc, cut_share, mean_tokens, problems, complete, acc_lenient), <local_dir>/gpu.json (GPU
seconds per job). Requires the deployed app (modal deploy wsp/exp8/app.py).

CLI: .venv/bin/python wsp/exp8/evaluate.py <source> <name> [sets,comma,separated]
"""
import json
import os
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
from vendor import merge as M  # noqa: E402

DEFAULT_OUT = os.path.join(HERE, "eval_results")


def _fn(name):
    return modal.Function.from_name(C.APP_NAME, name)


def _upload_dir(local, remote):
    vol = C.data_vol()
    with vol.batch_upload(force=True) as b:
        for f in sorted(os.listdir(local)):
            p = os.path.join(local, f)
            if os.path.isfile(p):
                b.put_file(p, f"{remote}/{f}")


def spec_for(source, name):
    if source in ("base", "unedited", None):
        return {"kind": "base"}
    if source.startswith("work:"):
        return {"kind": "work", "run": source[5:]}
    src = os.path.abspath(source)
    if os.path.exists(os.path.join(src, "manifest.json")) and os.path.exists(os.path.join(src, "edit.npz")):
        _upload_dir(src, f"edits/{name}")
        return {"kind": "edit_diff", "dir": f"/data/edits/{name}"}
    if any(f.endswith(".npy") for f in os.listdir(src)):
        _upload_dir(src, f"submissions/{name}")
        return {"kind": "npy_dir", "dir": f"/data/submissions/{name}"}
    raise ValueError(f"unknown edit source {source}")


def _download_parts(name, local_dir):
    vol = C.data_vol()
    try:
        entries = vol.listdir(f"eval/{name}/parts", recursive=True)
    except Exception:  # noqa: BLE001
        return
    for e in entries:
        if e.type != modal.volume.FileEntryType.FILE:
            continue
        rel = e.path.split(f"eval/{name}/", 1)[1]
        dst = os.path.join(local_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            vol.read_file_into_fileobj(e.path, f)


def evaluate(source, sets=("math500", "gsm8k"), name=None, protocol="think35", local_dir=None, rebuild=False,
             log=print):
    """Evaluate and return the merged dict {set: {acc, cut_share, mean_tokens, problems, complete, acc_lenient}}."""
    if name is None:
        name = "unedited" if source in ("base", "unedited") else os.path.basename(os.path.abspath(source))
    local_dir = local_dir or os.path.join(DEFAULT_OUT, name)
    os.makedirs(local_dir, exist_ok=True)
    sets = [s for s in sets]
    t0 = time.time()
    ck = os.path.join(local_dir, "ckpt.json")
    if rebuild or not os.path.exists(ck):
        info = _fn("build_ckpt").remote(name, spec_for(source, name))
        json.dump(info, open(ck, "w"), indent=1)
        log(f"[{name}] checkpoint built: {info.get('edited')}")
    jobs = [(name, name, s, f"{i}/{C.SHARDS[s]}", protocol) for s in sets for i in range(C.SHARDS[s])]
    done_local = set()
    for (_, _, s, sh, _) in jobs:
        i, k = sh.split("/")
        p = os.path.join(local_dir, "parts", f"{s}_s{i}of{k}", "scores.json")
        if os.path.exists(p) and s in json.load(open(p)):
            done_local.add((s, sh))
    todo = [j for j in jobs if (j[2], j[3]) not in done_local]
    log(f"[{name}] {len(todo)} of {len(jobs)} jobs to run")
    gpu_path = os.path.join(local_dir, "gpu.json")
    gpu = json.load(open(gpu_path)) if os.path.exists(gpu_path) else {}
    fn = _fn("eval_shard")
    calls = [(j, fn.spawn(*j)) for j in todo]
    errors = []
    for j, c in calls:
        try:
            r = c.get()
            gpu[f"{j[2]}_{j[3]}"] = {"wall_s": r["wall_s"], "gpu": r["scores"].get("_protocol", {}).get("gpu")}
            log(f"[{name}] {j[2]} {j[3]}: acc {r['scores'][j[2]]['acc']:.3f} cut {r['scores'][j[2]]['cut_share']:.3f} "
                f"({r['wall_s'] / 60:.0f} min)")
        except Exception as e:  # noqa: BLE001
            errors.append((j, repr(e)))
            log(f"[{name}] {j[2]} {j[3]} FAILED: {e!r}")
        json.dump(gpu, open(gpu_path, "w"), indent=1)
    _download_parts(name, local_dir)
    merged = M.merge(local_dir)
    merged["_meta"] = {"gpu_hours": sum(v["wall_s"] for v in gpu.values()) / 3600, "errors": errors,
                       "wall_s_this_call": time.time() - t0, "protocol": protocol, "vllm": C.VLLM_VERSION}
    json.dump(merged, open(os.path.join(local_dir, "merged.json"), "w"), indent=1)
    return merged


if __name__ == "__main__":
    src, nm = sys.argv[1], sys.argv[2]
    ss = sys.argv[3].split(",") if len(sys.argv) > 3 else ["math500", "gsm8k"]
    print(json.dumps(evaluate(src, ss, nm), indent=1))
