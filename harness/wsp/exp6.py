"""Experiment 6: write a 9x9 Go policy network from zero weights (Python sandbox + games/positions passes).

Opus gets the original runs/budget_<B>/message.txt verbatim. Each reply's first fenced block decides the action:
  ```python     one run in the conversation's sandbox (/work kept between runs; original prelude; 900 s; output cut
                at 20,000 characters; at most 120 runs) -> "OUTPUT of Python run k of at most 120:\n```\n...\n```"
  ```games      a games pass on the tensors saved so far (wsp.go.tools.games_pass; reply text exactly as the original)
  ```positions  a positions pass (wsp.go.tools.positions_pass)
A reply with none of these is the final answer, unless passes are left (then a reminder; the wording is ours, the
original never needed one). Budget 0 has no pass tools, so ```games / ```positions blocks are not acted on.

/work lives in a Modal sandbox (no network) and is mirrored to <run>/work after every action, so a restarted driver
restores it into a new sandbox. Passes run locally on the mirrored /work/net, and their files are copied into the
sandbox's /work/pass_<k>/.

  python wsp/exp6.py run <budget> <rep> [effort]     one conversation (default effort xhigh)
  python wsp/exp6.py score                           hidden evaluation + same-size PPO head-to-heads
"""
import glob
import io
import json
import os
import re
import shutil
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wsp import conversation, remote  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "lm_understand-logs", "exp6_go_from_scratch")
OUT = os.path.join(ROOT, "runs", "opus55", "exp6_go_from_scratch")
MAX_RUNS, TIMEOUT, MAX_OUT = 120, 900, 20_000
ARCH = {"depth": 6, "width": 512}
FIRST_BLOCK = re.compile(r"```(python|games|positions)[^\n]*\n(.*?)(?:```|\Z)", re.S)


def prelude():
    src = open(os.path.join(ROOT, "lm_understand-logs", "harness", "sandbox.py")).read()
    return re.search(r"PRELUDE = r'''(.*?)'''", src, re.S).group(1)


class Work:
    """The sandbox's /work, on the H100 VM (wsp/remote.py), mirrored locally after every action."""

    def __init__(self, mirror):
        self.mirror = mirror
        os.makedirs(mirror, exist_ok=True)
        if not os.path.exists(os.path.join(mirror, "arch.json")):
            json.dump(ARCH, open(os.path.join(mirror, "arch.json"), "w"))
        rel = os.path.relpath(os.path.dirname(os.path.abspath(mirror)), OUT)
        self.remote = f"{remote.BASE}/work/exp6/{rel}"
        remote.rsync_to(mirror, self.remote, delete=True)      # the local mirror is the source of truth

    def push_dir(self, local, sub):
        remote.rsync_to(local, f"{self.remote}/{sub}")

    def pull(self):
        remote.rsync_from(self.remote, self.mirror, delete=True)

    def run(self, code):
        # as the original: python -I -c PRELUDE + code, 900 s, 96 GB (RLIMIT_AS), output cut at 20,000 characters
        out, secs = remote.sandbox(["-I", "-c", prelude() + "\n" + code], "sandbox_full", self.remote,
                                   timeout=TIMEOUT, max_out=MAX_OUT, mem_gb=96)
        self.pull()
        return out, secs

    def close(self):
        pass


def run(budget, rep, effort="xhigh", max_turns=1000, tag=""):
    from wsp.go import tools as G
    first = open(os.path.join(LOGS, "runs", f"budget_{budget}", "message.txt")).read()
    folder = os.path.join(OUT, f"budget_{budget}_r{rep}" + ("" if effort == "xhigh" else f"_{effort}") + tag)
    conv = conversation.Conversation(folder, effort, first, max_turns=max_turns)
    if conv.state["done"]:
        return True
    work = Work(os.path.join(folder, "work"))
    os.makedirs(os.path.join(folder, "requests"), exist_ok=True)

    def handle(text, c):
        m = FIRST_BLOCK.search(text)
        kind = m.group(1) if m else None
        if kind in ("games", "positions") and budget == 0:
            kind = None
        passes = c.count("passes")
        if kind is None:
            if passes < budget:
                c.count("reminders", 1)
                c.log(f"final answer with {budget - passes} passes left: reminder")
                return (f"You have {budget - passes} of {budget} passes left. The session continues until every pass "
                        "is used: reply with a ```games or ```positions block (or a ```python block). When every "
                        "pass is used and your network is saved, give your final answer.")
            return None
        body = m.group(2)
        if kind == "python":
            k = c.count("runs", 1)
            if k > MAX_RUNS:
                return (f"You have used all {MAX_RUNS} Python runs. "
                        + (f"Passes left: {budget - passes}." if budget else "Give your final answer."))
            out, secs = work.run(body)
            c.tool_call({"kind": "python", "run": k, "s": secs, "code": body, "out": out[:6000]})
            c.log(f"python run {k} ({len(body):,} chars of code, {len(out):,} chars out, {secs} s); "
                  f"passes {passes}/{budget}")
            return f"OUTPUT of Python run {k} of at most {MAX_RUNS}:\n```\n{out}\n```"
        # a pass
        if passes >= budget:
            return f"No passes left (all {budget} used). Reply without a code or pass block to finish."
        k = passes + 1
        open(os.path.join(folder, "requests", f"{kind}_{k}.json"), "w").write(body)
        out_dir = os.path.join(folder, "pass_tmp", f"pass_{k}")
        shutil.rmtree(out_dir, ignore_errors=True)
        net_dir = os.path.join(work.mirror, "net")
        net = net_dir if os.path.isdir(net_dir) else {}
        fn = G.games_pass if kind == "games" else G.positions_pass
        reply = fn(net, body, out_dir, k=k, budget=budget)
        ok = G.pass_ok(reply)
        if ok:
            c.count("passes", 1)
            work.push_dir(out_dir, f"pass_{k}")
            work.pull()
        shutil.rmtree(os.path.join(folder, "pass_tmp"), ignore_errors=True)
        c.tool_call({"kind": kind, "pass": k if ok else None, "ok": ok, "request": body, "reply": reply[:6000]})
        c.log(f"{kind} pass {'%d/%d' % (k, budget) if ok else 'FAILED (not counted)'}")
        return reply

    try:
        return conv.run(handle)
    finally:
        work.close()


def score():
    from wsp.go import tools as G
    rows = []
    ppo = {s: os.path.join(LOGS, "networks", f"ppo_same_size_lr3e-5_s{s}") for s in range(3)}
    for folder in sorted(glob.glob(os.path.join(OUT, "budget_*"))):
        st = json.load(open(os.path.join(folder, "state.json")))
        if not st["done"]:
            continue
        net = os.path.join(folder, "work", "net")
        cache = os.path.join(folder, "eval.json")
        if os.path.exists(cache):
            r = json.load(open(cache))
        else:
            r = {"eval": G.evaluate(net if os.path.isdir(net) else {}, games=2000),
                 "h2h_same_size_ppo": {f"s{s}": G.h2h(net if os.path.isdir(net) else {}, p, games=2000, seed=31337)
                                       for s, p in ppo.items()}}
            json.dump(r, open(cache, "w"), indent=1)
        rows.append({"run": os.path.basename(folder), "runs": st["counters"].get("runs", 0),
                     "passes": st["counters"].get("passes", 0), "reminders": st["counters"].get("reminders", 0),
                     "turns": st["turns"], **r})
    os.makedirs(os.path.join(ROOT, "results", "phase3"), exist_ok=True)
    json.dump(rows, open(os.path.join(ROOT, "results", "phase3", "exp6_scores.json"), "w"), indent=1)
    for r in rows:
        e = r["eval"]
        print(r["run"], {k: round(v["score"], 3) for k, v in e.items() if isinstance(v, dict) and "score" in v},
              "main", e.get("main_score"),
              "h2h", {k: round(v.get("score", float("nan")), 3) for k, v in r["h2h_same_size_ppo"].items()})
    return rows


if __name__ == "__main__":
    if sys.argv[1] == "run":
        ok = run(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else "xhigh")
        print("done" if ok else "stopped")
    elif sys.argv[1] == "score":
        score()
