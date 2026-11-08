"""Modal app `wsp-go`: GPU training of the Go opponents and (optionally) GPU games / evaluation.

Volume `wsp-go` layout:
  /vol/train/<algo>/state.pkl, log.jsonl, s<seed>/iter<k>.npz     training state and checkpoints
  /vol/opponents/<name>/<tensor>.npy, arch.json, info.json         the exported opponents (copied to wsp/go/opponents/)

Usage (from the project root, local venv):
  .venv/bin/python -m wsp.go.modal_app bench ppo 3            # seconds per iteration on the GPU
  .venv/bin/python -m wsp.go.modal_app train ppo 0,1,2 20000  # detached training run (resumable)
Nothing is deployed: every call runs in an ephemeral app (`app.run()`), or detached for training.
"""
import os
import sys

import modal

APP_NAME = "wsp-go"
GPU = os.environ.get("WSP_GO_GPU", "H100,A100-80GB,A100-40GB,L40S").split(",")
HERE = os.path.dirname(os.path.abspath(__file__))

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("jax[cuda12]==0.6.2", "pgx==2.6.0", "optax==0.2.5", "numpy==1.26.4")
         .run_commands("mkdir -p /root/wsp && touch /root/wsp/__init__.py")
         .env({"XLA_PYTHON_CLIENT_PREALLOCATE": "false"})
         .add_local_dir(HERE, remote_path="/root/wsp/go", ignore=["opponents_ckpt", "__pycache__", "*.pyc"]))

app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name("wsp-go", create_if_missing=True)


@app.function(gpu=GPU, timeout=1800, volumes={"/vol": vol})
def bench(algo: str, nseeds: int, B: int = 1024, chunk: int = 20, rounds: int = 3):
    import time, shutil
    sys.path.insert(0, "/root")
    from wsp.go import train as T
    import jax
    d = f"/tmp/bench_{algo}"
    shutil.rmtree(d, ignore_errors=True)
    out = []
    t0 = time.time()
    for it in T.train(algo, list(range(nseeds)), chunk * rounds, d, B=B, chunk=chunk, ckpt_every=chunk):
        out.append((it, time.time() - t0))
    import json
    return {"device": str(jax.devices()), "times": out, "log": [json.loads(l) for l in open(d + "/log.jsonl")]}


@app.function(gpu=GPU, timeout=24 * 3600, volumes={"/vol": vol})
def train_job(algo: str, seeds: list, iters: int, sub: str = "", chunk: int = 50, cfg: dict = None):
    sys.path.insert(0, "/root")
    from wsp.go import train as T
    d = f"/vol/train/{sub or algo}"
    for it in T.train(algo, seeds, iters, d, chunk=chunk, ckpt_every=500, log=lambda s: print(s, flush=True),
                      time_limit=23 * 3600, cfg=cfg):
        vol.commit()
    vol.commit()
    return "done"


@app.function(gpu=GPU, timeout=3600, volumes={"/vol": vol})
def remote_call(fn: str, *args, **kwargs):
    """Run any wsp.go.tools function on the GPU (the same code as the local backend)."""
    sys.path.insert(0, "/root")
    from wsp.go import tools
    return getattr(tools, fn)(*args, **kwargs)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "bench":
        with modal.enable_output(), app.run():
            r = bench.remote(sys.argv[2], int(sys.argv[3]))
        print(r["device"])
        for row in r["log"]:
            print(row)
    elif cmd == "sweep":
        # sweep <iters> <json list of [sub, algo, seeds, cfg]>
        import json
        iters, jobs = int(sys.argv[2]), json.loads(sys.argv[3])
        with modal.enable_output(), app.run(detach=True):
            for sub, algo, seeds, cfg in jobs:
                print("spawned", sub, train_job.spawn(algo, seeds, iters, sub, 50, cfg).object_id)
    elif cmd == "train":
        algo, seeds, iters = sys.argv[2], [int(s) for s in sys.argv[3].split(",")], int(sys.argv[4])
        sub = sys.argv[5] if len(sys.argv) > 5 else ""
        with modal.enable_output(), app.run(detach=True):
            call = train_job.spawn(algo, seeds, iters, sub)
            print("spawned", call.object_id)
