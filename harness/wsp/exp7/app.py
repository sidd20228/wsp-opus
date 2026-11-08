"""Experiment 7 model side on Modal (app `wsp-exp7`): starting model, the six fine-tunes, their behaviour, the A/B
releases. Every step runs the RELEASED scripts (lm_understand-logs/exp7_sycophancy/code, dataset) unchanged, with the
volume laid out like the original's SYCMID_ROOT (see common.py).

  .venv/bin/modal run wsp/exp7/app.py::access          # can the wsp-hf token read google/gemma-3-270m-it?
  .venv/bin/modal run wsp/exp7/app.py::base            # download the starting model (pinned revision)
  .venv/bin/modal run wsp/exp7/app.py::facts           # optional: rerun data.py's filter with the starting model
  .venv/bin/modal run wsp/exp7/app.py::train           # pairs 100-102 in parallel: train.py x2 + eval.py x3 (job_pair.sh)
  .venv/bin/modal run wsp/exp7/app.py::releases        # make_release.py dev / devemb, map check, sha256 of everything
  .venv/bin/python -m wsp.exp7.fetch                   # download checkpoints, truth, releases; write the check 0.6 json
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import modal

try:
    from . import common as C
except ImportError:                      # `modal run wsp/exp7/app.py` imports this file as a script
    sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), "/opt/exp7/wsp_exp7"]   # local run / in container
    import common as C

app = modal.App(C.APP_NAME)
IMAGE = C.image()
VOL = C.volume()
SECRET = modal.Secret.from_name(C.SECRET_NAME)
CODE = f"{C.REMOTE_PKG}/code"
DATASET = f"{C.REMOTE_PKG}/dataset"
R = C.ROOT_REMOTE
ENV = {"SYCMID_ROOT": R, "SYCMID_BASE": C.BASE_REMOTE, "HF_HOME": f"{R}/hf_home", "TOKENIZERS_PARALLELISM": "false"}


def _run(cmd, env_extra=None):
    """Run a released script as job_pair.sh does (python <script> <args>), streaming its output."""
    env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}     # the released scripts never see the token
    env.update(ENV)
    env.update(env_extra or {})
    print("$", " ".join(cmd), flush=True)
    t0 = time.time()
    subprocess.run(cmd, env=env, check=True)
    return round(time.time() - t0, 1)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def _gpu_name():
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        return "no GPU"


# ---------------------------------------------------------------------------------------------------- access, base

@app.function(image=IMAGE, secrets=[SECRET], timeout=300)
def check_access():
    """Whether the token can read the gated repo. Returns the HTTP status of a config.json download (never the token)."""
    import tempfile

    from huggingface_hub import HfApi, hf_hub_download
    tok = os.environ.get("HF_TOKEN")
    out = {"repo": C.BASE_REPO, "token_present": bool(tok)}
    try:
        info = HfApi().model_info(C.BASE_REPO, token=tok)
        out["sha"], out["gated"] = info.sha, getattr(info, "gated", None)
    except Exception as e:  # noqa: BLE001
        out["model_info_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    try:
        with tempfile.TemporaryDirectory() as d:
            hf_hub_download(C.BASE_REPO, "config.json", token=tok, local_dir=d)
        out["ok"], out["status"] = True, 200
    except Exception as e:  # noqa: BLE001
        resp = getattr(e, "response", None)
        out["ok"], out["status"] = False, getattr(resp, "status_code", None)
        out["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
    return out


@app.function(image=IMAGE, secrets=[SECRET], volumes={"/data": VOL}, timeout=1800)
def download_base():
    """Snapshot of google/gemma-3-270m-it at its current revision into the volume (the original's
    $SYCMID_ROOT/models/gemma-3-270m-it); records the revision and file hashes."""
    from huggingface_hub import HfApi, snapshot_download
    tok = os.environ["HF_TOKEN"]
    sha = HfApi().model_info(C.BASE_REPO, token=tok).sha
    snapshot_download(C.BASE_REPO, revision=sha, local_dir=C.BASE_REMOTE, token=tok)
    files = {f: {"bytes": os.path.getsize(os.path.join(C.BASE_REMOTE, f)), "sha256": _sha256(os.path.join(C.BASE_REMOTE, f))}
             for f in sorted(os.listdir(C.BASE_REMOTE)) if os.path.isfile(os.path.join(C.BASE_REMOTE, f))}
    meta = {"repo": C.BASE_REPO, "revision": sha, "files": files, "t": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    os.makedirs(f"{R}/meta", exist_ok=True)
    json.dump(meta, open(f"{R}/meta/base.json", "w"), indent=1)
    VOL.commit()
    return meta


@app.function(image=IMAGE, volumes={"/data": VOL}, timeout=3600, cpu=8, memory=16384)
def filter_facts():
    """Rerun the released data.py's pool filter with the starting model (CPU, as in the original) and compare the kept
    facts with the released dataset/facts.json."""
    out = f"{R}/private/facts_rebuilt.json"
    os.makedirs(f"{R}/private", exist_ok=True)
    s = _run([sys.executable, f"{DATASET}/data.py", C.BASE_REMOTE, out])
    a, b = json.load(open(out)), json.load(open(f"{DATASET}/facts.json"))
    qa, qb = [r["question"] for r in a], [r["question"] for r in b]
    res = {"seconds": s, "rebuilt": len(a), "released": len(b), "identical": a == b,
           "only_rebuilt": sorted(set(qa) - set(qb)), "only_released": sorted(set(qb) - set(qa))}
    json.dump(res, open(f"{R}/meta/facts_check.json", "w"), indent=1)
    VOL.commit()
    return res


# ---------------------------------------------------------------------------------------------------- training

@app.function(image=IMAGE, gpu=C.GPU, volumes={"/data": VOL}, timeout=3 * 3600, cpu=4, memory=32768)
def train_pair(pair: int, force: bool = False):
    """job_pair.sh for one pair: the released split, train.py honest and sycophant (same seed / order / schedule),
    eval.py on both fine-tunes and on the starting model. Existing outputs are kept (as job_pair.sh does) unless force."""
    import shutil
    P = pair
    gpu = _gpu_name()
    print(gpu, flush=True)
    os.makedirs(f"{R}/private/splits", exist_ok=True)
    os.makedirs(f"{R}/private/truth", exist_ok=True)
    split = f"{R}/private/splits/pair{P}.json"
    shutil.copyfile(f"{DATASET}/splits/pair{P}.json", split)     # the released split (job_pair.sh would build it)
    times = {"gpu": gpu}
    t0 = time.time()
    for role in C.ROLES:
        d = f"{R}/private/runs/pair{P}/{role}"
        if force and os.path.exists(d):
            shutil.rmtree(d)
        if not os.path.exists(f"{d}/final/model.safetensors"):
            times[f"train_{role}"] = _run([sys.executable, f"{CODE}/train.py", str(P), role, split, d])
            VOL.commit()
        truth = f"{R}/private/truth/pair{P}_{role}.json"
        if force or not os.path.exists(truth):
            times[f"eval_{role}"] = _run([sys.executable, f"{CODE}/eval.py", f"{d}/final", split, truth])
            VOL.commit()
    truth = f"{R}/private/truth/pair{P}_base.json"
    if force or not os.path.exists(truth):
        times["eval_base"] = _run([sys.executable, f"{CODE}/eval.py", C.BASE_REMOTE, split, truth])
    times["total"] = round(time.time() - t0, 1)
    os.makedirs(f"{R}/meta", exist_ok=True)
    json.dump(times, open(f"{R}/meta/times_pair{P}.json", "w"), indent=1)
    VOL.commit()
    print(f"pair {P} done", times, flush=True)
    return times


# ---------------------------------------------------------------------------------------------------- releases

@app.function(image=IMAGE, volumes={"/data": VOL}, timeout=3600, cpu=4, memory=16384)
def make_releases(answer_key: dict):
    """make_release.py dev and devemb for pairs 100 101 102 in one call each (the release IDs come from one rng per
    set, drawn in pair order), then: the A/B maps equal the released answer key; every file's sha256."""
    VOL.reload()
    out = {}
    for st in ("dev", "devemb"):
        mp = f"{R}/private/release_{st}_map.json"
        out[f"seconds_{st}"] = _run([sys.executable, f"{CODE}/make_release.py", st, *map(str, C.PAIRS)])
        got = json.load(open(mp))
        out[f"map_{st}_matches_answer_key"] = got == answer_key[st]
        out[f"map_{st}"] = got
    sums = {}
    for top in ("private/runs", "release_dev", "release_devemb"):
        for dp, _, fs in os.walk(f"{R}/{top}"):
            for f in sorted(fs):
                p = os.path.join(dp, f)
                sums[os.path.relpath(p, R)] = {"bytes": os.path.getsize(p), "sha256": _sha256(p)}
    out["files"] = sums
    json.dump(out, open(f"{R}/meta/releases.json", "w"), indent=1)
    VOL.commit()
    return out


# ---------------------------------------------------------------------------------------------------- entrypoints

@app.local_entrypoint()
def access():
    print(json.dumps(check_access.remote(), indent=1))


@app.local_entrypoint()
def base():
    m = download_base.remote()
    print(json.dumps({"revision": m["revision"], "files": {k: v["bytes"] for k, v in m["files"].items()}}, indent=1))


@app.local_entrypoint()
def facts():
    print(json.dumps(filter_facts.remote(), indent=1))


@app.local_entrypoint()
def train(pairs: str = "100,101,102", force: bool = False):
    ps = [int(p) for p in pairs.split(",")]
    for p, t in zip(ps, train_pair.map(ps, kwargs={"force": force})):
        print(p, json.dumps(t))


@app.local_entrypoint()
def releases():
    key = {st: json.load(open(os.path.join(C.LOGS, "eval", "answer_key", f"release_{st}_map.json")))
           for st in ("dev", "devemb")}
    r = make_releases.remote(key)
    print(json.dumps({k: v for k, v in r.items() if k != "files"}, indent=1))
