"""Benchmark evaluation of an (edited) Qwen3.5-0.8B on the H100 VM, released protocol (code/evaluate.py, think35),
the released shard layout (results per shard as before), merged with the released merge.py.

    from wsp.exp8.evaluate import evaluate
    merged = evaluate("base", ["math500", "gsm8k"], name="unedited")
    merged = evaluate("lm_understand-logs/exp8_math/runs/p50c/edit", ["math500"], name="astra_p50c")
    merged = evaluate("work:<run_id>", C.EIGHT_SETS, name="opus_p50_r0")      # a run's /work/edited
    merged = evaluate("base", ["math500"], name="vm_unedited", shards=[0, 1, 2])  # a subset of shards

source: "base" | a local folder with edit.npz + manifest.json (rebuilt with apply_edit.py, hash-checked) | a local
folder of <tensor>.npy (a submission, cast to bfloat16) | "work:<run>". One vLLM engine per call (all shards run in it,
one after another; the GPU is shared, memory <= 0.80). Results: <local_dir>/parts/..., merged.json, gpu.json.

CLI: .venv/bin/python wsp/exp8/evaluate.py <source> <name> [sets] [shards]
"""
import json
import os
import shlex
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from wsp import remote as R  # noqa: E402
import common as C  # noqa: E402
from vendor import merge as M  # noqa: E402

GPU_PY = f"{R.BASE}/venvs/exp8_gpu/bin/python"
VM_GPU = f"{R.BASE}/code/wsp/exp8/vm_gpu.py"
DATA = f"{R.BASE}/data/exp8"
DEFAULT_OUT = os.path.join(HERE, "eval_results_vm")


def spec_for(source, name):
    if source in ("base", "unedited", None):
        return {"kind": "base"}
    if source.startswith("work:"):
        return {"kind": "work", "run": source[5:]}
    src = os.path.abspath(source)
    if os.path.exists(os.path.join(src, "manifest.json")):
        R.rsync_to(src, f"{DATA}/edits/{name}")
        return {"kind": "edit_diff", "dir": f"{DATA}/edits/{name}"}
    R.rsync_to(src, f"{DATA}/submissions/{name}")
    return {"kind": "npy_dir", "dir": f"{DATA}/submissions/{name}"}


def evaluate(source, sets=("math500", "gsm8k"), name=None, local_dir=None, rebuild=False, shards=None,
             protocol="think35", log=print):
    if name is None:
        name = "unedited" if source in ("base", "unedited") else os.path.basename(os.path.abspath(source))
    local_dir = local_dir or os.path.join(DEFAULT_OUT, name)
    os.makedirs(local_dir, exist_ok=True)
    ck = os.path.join(local_dir, "ckpt.json")
    if rebuild or not os.path.exists(ck):
        spec = json.dumps(spec_for(source, name))
        info = json.loads(R.sh(f"{GPU_PY} {VM_GPU} build_ckpt {shlex.quote(name)} {shlex.quote(spec)}",
                               timeout=1800).stdout)
        json.dump(info, open(ck, "w"), indent=1)
        log(f"[{name}] checkpoint built: {info.get('edited')}")
    t0 = time.time()
    sh_arg = f" --shards {','.join(str(i) for i in shards)}" if shards is not None else ""
    p = R.sh(f"{GPU_PY} {VM_GPU} eval {shlex.quote(name)} {shlex.quote(name)} {','.join(sets)}{sh_arg} "
             f"2>> {R.BASE}/logs/exp8/eval_{name}.log", timeout=24 * 3600, check=False, retries=1)
    try:
        res = json.loads(p.stdout.decode().strip().splitlines()[-1])
    except (ValueError, IndexError):
        res = {"error": f"evaluation failed (exit {p.returncode}); see ~/wsp-opus/logs/exp8/eval_{name}.log"}
        log(f"[{name}] {res['error']}")
    gpu_path = os.path.join(local_dir, "gpu.json")
    gpu = json.load(open(gpu_path)) if os.path.exists(gpu_path) else {}
    for j, v in res.get("jobs", {}).items():
        if not v.get("skipped"):
            gpu[j] = {"wall_s": v["wall_s"], "gpu": "H100 NVL (VM)"}
        log(f"[{name}] {j}: acc {v['acc']:.3f} cut {v['cut_share']:.3f} ({v['wall_s'] / 60:.0f} min)")
    json.dump(gpu, open(gpu_path, "w"), indent=1)
    R.rsync_from(f"{DATA}/eval/{name}", local_dir)
    merged = M.merge(local_dir)
    merged["_meta"] = {"gpu_hours": sum(v["wall_s"] for v in gpu.values()) / 3600, "wall_s_this_call": time.time() - t0,
                       "protocol": protocol, "vllm": C.VLLM_VERSION, "backend": "vm", "error": res.get("error")}
    json.dump(merged, open(os.path.join(local_dir, "merged.json"), "w"), indent=1)
    return merged


if __name__ == "__main__":
    src, nm = sys.argv[1], sys.argv[2]
    ss = sys.argv[3].split(",") if len(sys.argv) > 3 else ["math500", "gsm8k"]
    shs = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else None
    print(json.dumps(evaluate(src, ss, nm, shards=shs), indent=1))
