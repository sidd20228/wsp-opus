"""Phase 1: the one-reply experiments (1, 2, 4, 5) with Opus 5.5 through the Claude Code CLI.

Each run is one fresh conversation and one reply; the prompt is the original's, byte for byte. Efforts: the
original's per experiment (high for 1, 2, 4; xhigh for 5) and the `max` ablation. Order: replicate 1 at the original
effort for every item first (the primary runs), then its max ablation, then replicate 2, and so on, so that a partial
run still has balanced results. Resumable: a run whose run.json says ok is skipped.

  python wsp/phase1.py [--workers 2] [--list]
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wsp import cc  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "lm_understand-logs")
OUT = os.path.join(ROOT, "runs", "opus55")

EXPS = {
    # name: (items, prompt path pattern, original effort, replicates)
    "exp1_which_game": (["breakout", "pong", "pacman", "freeway"], "exp1_which_game/runs/{i}/prompt.txt", "high", 5),
    "exp2_diffusion": (["circle", "square", "triangle"], "exp2_diffusion/runs/{i}/prompt.txt", "high", 5),
    "exp5_write_from_scratch": (["breakout", "seaquest", "asterix"],
                                "exp5_write_from_scratch/runs/{i}/message.txt", "xhigh", 3),
    "exp4_fix_car": (["oneshot_repair"], "exp4_fix_car/runs/{i}/reply/prompt.txt", "high", 3),
}
_lock = threading.Lock()


def jobs():
    out = []
    for rep in range(1, 6):
        for ablation in (False, True):
            for exp, (items, pat, eff, reps) in EXPS.items():
                if rep > reps:
                    continue
                effort = "max" if ablation else eff
                for i in items:
                    out.append({"exp": exp, "item": i, "effort": effort, "rep": rep,
                                "primary": rep == 1 and not ablation,
                                "prompt": os.path.join(LOGS, pat.format(i=i)),
                                "dir": os.path.join(OUT, exp, f"{i}_{effort}_r{rep}")})
    return out


def log(msg):
    with _lock:
        line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
        print(line, flush=True)
        with open(os.path.join(OUT, "phase1.log"), "a") as f:
            f.write(line + "\n")


def run(job):
    rj = os.path.join(job["dir"], "reply", "run.json")
    if os.path.exists(rj) and json.load(open(rj)).get("ok"):
        return "skip"
    os.makedirs(job["dir"], exist_ok=True)
    prompt = open(job["prompt"]).read()
    with open(os.path.join(job["dir"], "job.json"), "w") as f:
        json.dump({k: v for k, v in job.items()}, f, indent=1)
    log(f"start {job['exp']} {job['item']} {job['effort']} r{job['rep']} ({len(prompt):,} chars)")
    conv = cc.new_conversation()
    rec = cc.turn(conv, prompt, job["effort"], os.path.join(job["dir"], "reply"), persist=False, log=log)
    u = rec.get("usage") or {}
    log(f"done  {job['exp']} {job['item']} {job['effort']} r{job['rep']}: ok={rec['ok']} stop={rec['stop_reason']} "
        f"in={u.get('input_tokens', 0) + u.get('cache_read_input_tokens', 0) + u.get('cache_creation_input_tokens', 0):,} "
        f"out={u.get('output_tokens', 0):,} wall={rec['wall_s']}s list$={rec.get('cost_usd_list_equivalent')}")
    return "ok" if rec["ok"] else "fail"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated experiment names")
    a = ap.parse_args()
    js = jobs()
    if a.only:
        js = [j for j in js if j["exp"] in a.only.split(",")]
    if a.list:
        for j in js:
            print(j["exp"], j["item"], j["effort"], j["rep"], "PRIMARY" if j["primary"] else "")
        print(len(js), "runs")
        sys.exit()
    os.makedirs(OUT, exist_ok=True)
    log(f"phase 1: {len(js)} runs, {a.workers} at a time, CLI {cc.cli_version()}")
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(run, js))
    log(f"phase 1 finished: {res.count('ok')} ok, {res.count('skip')} skipped, {res.count('fail')} failed")
