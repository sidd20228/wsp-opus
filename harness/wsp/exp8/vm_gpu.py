"""GPU side of Experiment 8 on the H100 VM (replaces app.py's Modal functions). Run with the exp8_gpu venv:

  vm_gpu.py fwd <run> <k>                  one forward-pass request (JSON on stdin); prints the summary JSON
  vm_gpu.py build_ckpt <name> <spec json>  data/exp8/ckpt/<name>/ (spec as app.build_ckpt; "work" = a run's /work/edited)
  vm_gpu.py eval <ckpt> <out> <sets> [--shards i,j,...]   evaluation jobs in the released shard layout, one vLLM
                                           engine per call, results in data/exp8/eval/<out>/parts/; prints JSON
  vm_gpu.py prep_gpqa                      GPQA Diamond (gated) -> data/exp8/bench/gpqa.jsonl, token from
                                           ~/wsp-opus/.hf_token in this process's environment only

The GPU is shared and there is one: every GPU command holds data/exp8/gpu.lock, vLLM uses at most 0.80 of the memory,
and a "both" pass runs the original and then the edited model, each in its own fresh process (one engine per process).
"""
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback

BASE = os.path.expanduser("~/wsp-opus")
DATA = f"{BASE}/data/exp8"
HERE = os.path.dirname(os.path.abspath(__file__))
ENV = {"EXP8_BACKEND": "vm", "EXP8_MODEL": f"{DATA}/models/Qwen3.5-0.8B-text", "EXP8_PRACTICE": f"{DATA}/practice",
       "EXP8_BENCH": f"{DATA}/bench", "EXP8_TMP": f"{DATA}/tmp", "EXP8_GPU_MEM": "0.80",
       "VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "PYTHONFAULTHANDLER": "1", "VLLM_CACHE_ROOT": f"{DATA}/vllm_cache", "NLTK_DATA": f"{DATA}/nltk",
       "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1",
       "OMP_NUM_THREADS": "8", "HF_HOME": f"{DATA}/hf_home"}
for k_, v_ in ENV.items():
    os.environ.setdefault(k_, v_)
os.makedirs(os.environ["EXP8_TMP"], exist_ok=True)
sys.path.insert(0, HERE)
import common as C  # noqa: E402


def work_dir(run):
    assert run and all(c.isalnum() or c in "-_" for c in run), run
    return f"{BASE}/work/exp8/{run}/work"


class GpuLock:
    def __enter__(self):
        self.f = open(f"{DATA}/gpu.lock", "w")
        fcntl.flock(self.f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *a):
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def _child(args, payload, run, timeout):
    """run `vm_gpu.py <args>` in a fresh process (one vLLM engine, then the process ends); returns its JSON result."""
    tag = f"{int(time.time() * 1000)}_{os.getpid()}"
    inp, out = f"{DATA}/tmp/in_{tag}.json", f"{DATA}/tmp/out_{tag}.json"
    json.dump(payload, open(inp, "w"))
    env = dict(os.environ, EXP8_WORK=work_dir(run))
    log = open(f"{BASE}/logs/exp8/gpu_{tag}.log", "w")
    p = subprocess.Popen([sys.executable, __file__, *args, inp, out], env=env, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True)
    try:
        rc = p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        rc = "timeout"
    finally:
        kill_group(p.pid)          # vLLM's EngineCore lives in the worker's process group: free the GPU every time
    log.close()
    os.unlink(inp)
    if not os.path.exists(out):
        tail = open(log.name).read()[-3000:]
        last = tail.strip().splitlines()[-1] if tail.strip() else ""
        raise RuntimeError(f"GPU worker failed (exit {rc}): {last}" if rc != "timeout"
                           else f"the forward pass took longer than {timeout} s and was stopped")
    res = json.load(open(out))
    os.unlink(out)
    if "error" in res:
        raise _Raise(res)
    return res


def kill_group(pgid, wait_s=60):
    import signal
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return
    t0 = time.time()
    while time.time() - t0 < wait_s:          # until no process of the group is left
        if subprocess.run(["pgrep", "-g", str(pgid)], capture_output=True).returncode != 0:
            return
        time.sleep(0.5)


class _Raise(Exception):
    def __init__(self, res):
        super().__init__(f"{res['type']}: {res['error']}")
        self.res = res


def worker(kind, inp, out):
    """inner process: kind gen|tf; writes the result (or {"error", "type"}) to `out`."""
    try:
        import fwdserver
        a = json.load(open(inp))
        s = fwdserver.Server()
        if kind == "gen":
            r = s.generate(a["run"], a["req"], a["k"], a["which"], a["tag"])
        else:
            r = s.teacher_forced(a["run"], a["req"], a["k"])
    except Exception as e:  # noqa: BLE001
        r = {"error": str(e), "type": type(e).__name__, "trace": traceback.format_exc()[-4000:]}
    with open(out, "w") as f:
        json.dump(r, f)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)          # do not wait for vLLM's engine teardown


def fwd(run, k, request_text, timeout=2 * 3600):
    import summary as SM
    t0 = time.time()
    try:
        req = json.loads(request_text)
        SM.validate(req)
        os.makedirs(work_dir(run) + "/edited", exist_ok=True)
        with GpuLock():
            t_lock = time.time() - t0
            if req["type"] == "teacher_forced":
                res = _child(["tf"], {"run": run, "req": req, "k": k}, run, timeout)
                res["s"] = round(time.time() - t0, 1)
                info = {"lock_wait_s": round(t_lock, 1)}
            elif req["model"] == "both":
                out = {w: _child(["gen"], {"run": run, "req": req, "k": k, "which": w, "tag": f"gen_{k}_{w}"}, run,
                                 timeout) for w in ("original", "edited")}
                acts = {w: out[w]["acts"] for w in out} if req.get("activations") else None
                res = {"ok": True, "type": "generate", "model": "both",
                       "files": {w: out[w]["file"] for w in ("original", "edited")}, "activations": acts,
                       "original": SM.model_summary(out["original"]["records"]),
                       "edited": SM.model_summary(out["edited"]["records"]),
                       "paired": SM.paired(out["original"]["records"], out["edited"]["records"]),
                       "s": round(time.time() - t0, 1)}
                info = {w: {kk: out[w][kk] for kk in ("s", "gen_s", "edited_tensors")} for w in out}
                info["lock_wait_s"] = round(t_lock, 1)
            else:
                o = _child(["gen"], {"run": run, "req": req, "k": k, "which": req["model"], "tag": f"gen_{k}"}, run,
                           timeout)
                res = {}
                if o["acts"]:
                    res["activations"] = o["acts"]
                res.update({"ok": True, "type": "generate", "model": req["model"], "file": o["file"],
                            **SM.model_summary(o["records"]), "s": round(time.time() - t0, 1)})
                info = {req["model"]: {kk: o[kk] for kk in ("s", "gen_s", "edited_tensors")},
                        "lock_wait_s": round(t_lock, 1)}
    except _Raise as e:
        res = {"ok": False, "error": str(e), "trace": e.res.get("trace", ""), "s": round(time.time() - t0, 1)}
        info = {}
    except Exception as e:  # noqa: BLE001
        res = {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-4000:],
               "s": round(time.time() - t0, 1)}
        info = {}
    logd = f"{BASE}/work/exp8/{run}/fwd_log"
    os.makedirs(logd, exist_ok=True)
    json.dump({"request": request_text, "result": res, "info": info, "wall_s": time.time() - t0},
              open(f"{logd}/fwd_{k}.json", "w"), indent=1)
    return res


def build_ckpt(name, spec):
    import shutil
    import numpy as np
    import torch
    from vendor import apply_edit, st_numpy as S
    out = f"{DATA}/ckpt/{name}"
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    for f in C.VIEW_FILES:
        if f != "model.safetensors":
            shutil.copyfile(os.path.join(C.MODEL_DIR, f), os.path.join(out, f))
    base = os.path.join(C.MODEL_DIR, "model.safetensors")
    info = {"name": name, "spec": spec}
    if spec["kind"] == "base":
        shutil.copyfile(base, os.path.join(out, "model.safetensors"))
        info["edited"] = []
    elif spec["kind"] == "edit_diff":
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            apply_edit.main(base, spec["dir"], os.path.join(out, "model.safetensors"))
        info["apply_edit_log"] = buf.getvalue()
        info["edited"] = [t["name"] for t in json.load(open(os.path.join(spec["dir"], "manifest.json")))["tensors"]]
    else:
        src = work_dir(spec["run"]) + "/edited" if spec["kind"] == "work" else spec["dir"]
        edits = {f[:-4]: os.path.join(src, f) for f in os.listdir(src) if f.endswith(".npy")}
        tensors = S.load(base)
        changed = {}
        for n, p in edits.items():
            assert n in tensors and n not in C.FROZEN, n
            dt, bits = tensors[n]
            a = torch.from_numpy(np.load(p)).to(torch.bfloat16)          # build_ckpt.py: cast to bfloat16
            assert list(a.shape) == list(bits.shape), (n, a.shape, bits.shape)
            nb = a.view(torch.int16).numpy().view(np.uint16)
            changed[n] = int((nb != bits).sum())
            tensors[n] = ("BF16", nb)
        S.save(os.path.join(out, "model.safetensors"), tensors, {"format": "pt"})
        info["edited"] = sorted(edits)
        info["changed_entries"] = changed
    json.dump(info, open(os.path.join(out, "edits.json"), "w"), indent=1)
    return info


def evaluate(ckpt, out_name, sets, shards=None, protocol="think35"):
    """all requested shards of `sets` with one engine (created only if a shard still has to run)."""
    import evaljob
    res, llm, t0 = {}, None, time.time()
    if True:                                  # the caller ("eval") holds the GPU lock
        t_lock = 0.0
        for s in sets:
            k = C.SHARDS[s]
            for i in (shards if shards is not None else range(k)):
                out = f"{DATA}/eval/{out_name}/parts/{s}_s{i}of{k}"
                path = f"{DATA}/ckpt/{ckpt}"
                t1 = time.time()
                prev = os.path.join(out, "scores.json")
                done = os.path.exists(prev) and "_protocol" in json.load(open(prev)) and s in json.load(open(prev))
                if not done and llm is None:
                    from vllm import LLM
                    llm = LLM(model=path, dtype="bfloat16", gpu_memory_utilization=C.GPU_MEM,
                              max_model_len=evaljob.SAMPLING[protocol]["max_tokens"] + 2048, seed=0)
                r = evaljob.run(path, out, [s], protocol=protocol, shard=f"{i}/{k}", llm=llm)
                res[f"{s}_{i}/{k}"] = {"acc": r[s]["acc"], "cut_share": r[s]["cut_share"],
                                       "wall_s": round(time.time() - t1, 1), "skipped": done}
                print(json.dumps({f"{s}_{i}/{k}": res[f"{s}_{i}/{k}"]}), file=sys.stderr, flush=True)
    return {"jobs": res, "wall_s": round(time.time() - t0, 1), "lock_wait_s": round(t_lock, 1)}


def prep_gpqa():
    """GPQA Diamond from the gated dataset (CSV), rows in the original's order; the token stays in this process."""
    import csv
    import mcprompt
    os.environ.pop("HF_HUB_OFFLINE", None)          # before huggingface_hub reads its settings
    tok = open(f"{BASE}/.hf_token").read().strip()
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("Idavidrein/gpqa", "gpqa_diamond.csv", repo_type="dataset", token=tok,
                        cache_dir=f"{DATA}/hf_gated")
    del tok
    rows = [mcprompt.gpqa_row(r) for r in csv.DictReader(open(p, newline="", encoding="utf-8"))]
    order = json.load(open(os.path.join(HERE, "bench_order.json")))["gpqa"]
    by = {r["id"]: r for r in rows}
    missing = [i for i in order if i not in by]
    assert not missing, missing[:5]
    with open(f"{DATA}/bench/gpqa.jsonl", "w") as f:
        for i in order:
            f.write(json.dumps(by[i], ensure_ascii=False) + "\n")
    return {"rows": len(order), "source_rows": len(rows)}


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd in ("gen", "tf"):
        worker(cmd, sys.argv[2], sys.argv[3])
    elif cmd == "fwd":
        print(json.dumps(fwd(sys.argv[2], int(sys.argv[3]), sys.stdin.read()), indent=1))
    elif cmd == "build_ckpt":
        print(json.dumps(build_ckpt(sys.argv[2], json.loads(sys.argv[3]))))
    elif cmd == "eval":                      # the engine runs in a child process group, killed at the end
        with GpuLock():
            inp = f"{DATA}/tmp/eval_{os.getpid()}.json"
            outp = f"{DATA}/tmp/eval_{os.getpid()}.out"
            json.dump(sys.argv[2:], open(inp, "w"))
            with open(outp, "w") as fo:          # a file, not a pipe: vLLM's engine process inherits it
                p = subprocess.Popen([sys.executable, __file__, "eval_inner", inp], stdout=fo, start_new_session=True)
                p.wait()
            kill_group(p.pid)
            out = open(outp).read()
        print(out.strip().splitlines()[-1] if out.strip() else json.dumps({"error": f"exit {p.returncode}"}))
    elif cmd == "eval_inner":
        a = json.load(open(sys.argv[2]))
        sh = [int(x) for x in a[a.index("--shards") + 1].split(",")] if "--shards" in a else None
        print(json.dumps(evaluate(a[0], a[1], a[2].split(","), sh)), flush=True)
        os._exit(0)
    elif cmd == "prep_gpqa":
        print(json.dumps(prep_gpqa()))
