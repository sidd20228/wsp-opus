"""Score every finished Phase 1 run and write results/phase1/scores.json and results/phase1/comparison.md.

Exp 1/2: keyword mapping of the answer line (checked against Astra's answers in Phase 0).
Exp 5: the reply's last edits block -> network from all-zero weights -> 1,000 MinAtar episodes.
Exp 4: the reply's last edits block applied to the weights as shown -> edited actor saved (scored on Kinetix by
       wsp/envs/kinetix_eval.py; results merged from <run>/kinetix.json when present).
Idempotent: per-run results are cached next to the run (score.json, net.npz, eval.json).
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wsp import edits, exp4, minatar_eval, score_simple  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "lm_understand-logs")
RUNS = os.path.join(ROOT, "runs", "opus55")
RES = os.path.join(ROOT, "results", "phase1")
TRUE1 = {"breakout": "breakout", "pong": "pong", "pacman": "pacman", "freeway": "freeway"}
SHAPES5 = {"W1": (600, 32), "b1": (32,), "W2": (32, 32), "b2": (32,), "W3": (32, 6), "b3": (6,)}
ASTRA = {
    "exp1_which_game": {"breakout": (True, 99), "pong": (True, 96), "pacman": (True, 96), "freeway": (True, 97)},
    "exp2_diffusion": {"circle": (True, 84), "square": (True, 77), "triangle": (True, 88)},
    "exp5_write_from_scratch": {"breakout": 517.4, "seaquest": 53.1, "asterix": 53.2},
    "exp4_fix_car": {"oneshot_repair": 1.0},
}
PPO5 = {"breakout": (15.8, 25.3), "seaquest": (5.6, 88.9), "asterix": (25.8, 55.2)}


def score_run(d):
    job = json.load(open(os.path.join(d, "job.json")))
    rj = os.path.join(d, "reply", "run.json")
    if not os.path.exists(rj):
        return None
    run = json.load(open(rj))
    text = open(os.path.join(d, "reply", "answer.md")).read()
    u = dict(run.get("usage") or {})
    it = u.get("iterations") or []
    if it:  # the CLI's top-level totals can over-count long multi-block replies; the per-message record is exact
        for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            u[k] = sum(i.get(k, 0) or 0 for i in it)
    base = {"exp": job["exp"], "item": job["item"], "effort": job["effort"], "rep": job["rep"],
            "primary": job["primary"], "ok": run["ok"], "stop_reason": run["stop_reason"],
            "tool_free": run["audit"]["tool_free"], "wall_s": run["wall_s"],
            "output_tokens": u.get("output_tokens"),
            "thinking_tokens": ((u.get("output_tokens_details") or {}).get("thinking_tokens")),
            "input_tokens": (u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
                             + u.get("cache_creation_input_tokens", 0)),
            "list_usd": run.get("cost_usd_list_equivalent")}
    if not run["ok"]:
        return base
    cache = os.path.join(d, "score.json")
    if os.path.exists(cache):
        s = json.load(open(cache))
        if s.get("_src_wall") == run["wall_s"]:
            kj = os.path.join(d, "kinetix.json")
            if job["exp"] == "exp4_fix_car" and os.path.exists(kj):
                s["kinetix"] = json.load(open(kj))
            return {**base, **s}
    if job["exp"] == "exp1_which_game":
        s = score_simple.score_exp1(text)
        s["correct"] = s["guess"] == TRUE1[job["item"]]
    elif job["exp"] == "exp2_diffusion":
        s = score_simple.score_exp2(text)
        s["correct"] = s["guess"] == job["item"]
    elif job["exp"] == "exp5_write_from_scratch":
        net = {k: np.zeros(v) for k, v in SHAPES5.items()}
        rep = edits.apply(text, net)
        np.savez(os.path.join(d, "net.npz"), **{k: v.astype(np.float32) for k, v in net.items()})
        ev = minatar_eval.evaluate(job["item"], {k: v.astype(np.float32) for k, v in net.items()}, 1000, 0)
        s = {"edit_report": {k: rep[k] for k in ("block", "statements", "applied", "cut_off")},
             "edit_errors": rep["errors"][:5],
             "nonzero": int(sum(np.count_nonzero(v) for v in net.values())), "minatar": ev,
             "score": ev["mean"], "score_se": ev["se"]}
    elif job["exp"] == "exp4_fix_car":
        prompt = open(job["prompt"]).read()
        net, rep, ch = exp4.apply_reply(prompt, text)
        np.savez(os.path.join(d, "edited_actor.npz"), **{k: v.astype(np.float32) for k, v in net.items()})
        s = {"edit_report": {k: rep[k] for k in ("block", "statements", "applied", "cut_off")},
             "edit_errors": rep["errors"][:5], "changes": ch}
        kj = os.path.join(d, "kinetix.json")
        if os.path.exists(kj):
            s["kinetix"] = json.load(open(kj))
    s["_src_wall"] = run["wall_s"]
    json.dump(s, open(cache, "w"), indent=1)
    return {**base, **s}


def table(rows):
    L = ["# Phase 1: Astra (post) vs Opus 5.5", "",
         "Primary = replicate 1 at the original effort (named before scoring). All runs are listed in scores.json.",
         ""]
    by = {}
    for r in rows:
        by.setdefault((r["exp"], r["item"], r["effort"]), []).append(r)

    def runs_of(exp, item, eff):
        return sorted(by.get((exp, item, eff), []), key=lambda r: r["rep"])

    L += ["## Exp 1: which game (open answer)", "",
          "| Game | Astra (conf.) | Opus primary (conf.) | Opus high: correct / runs | Opus max: correct / runs |",
          "|---|---|---|---|---|"]
    for g in TRUE1:
        hi, mx = runs_of("exp1_which_game", g, "high"), runs_of("exp1_which_game", g, "max")
        p = next((r for r in hi if r["primary"]), None)
        ps = f"{p.get('guess')} ({p.get('confidence')})" + (" ✓" if p.get("correct") else " ✗") if p and p["ok"] else "—"
        f = lambda rs: f"{sum(bool(r.get('correct')) for r in rs)} / {len(rs)}" if rs else "—"
        L.append(f"| {g} | ✓ ({ASTRA['exp1_which_game'][g][1]}) | {ps} | {f(hi)} | {f(mx)} |")
    L += ["", "## Exp 2: what the diffusion model draws (open answer)", "",
          "| Shape | Astra (conf.) | Opus primary (conf.) | Opus high: correct / runs | Opus max: correct / runs | Opus drew filled shapes |",
          "|---|---|---|---|---|---|"]
    for sh in ("circle", "square", "triangle"):
        hi, mx = runs_of("exp2_diffusion", sh, "high"), runs_of("exp2_diffusion", sh, "max")
        p = next((r for r in hi if r["primary"]), None)
        ps = f"{p.get('guess')} ({p.get('confidence')})" + (" ✓" if p.get("correct") else " ✗") if p and p["ok"] else "—"
        f = lambda rs: f"{sum(bool(r.get('correct')) for r in rs)} / {len(rs)}" if rs else "—"
        filled = sum(1 for r in hi + mx if r.get("example_filled"))
        L.append(f"| {sh} | ✓ ({ASTRA['exp2_diffusion'][sh][1]}) | {ps} | {f(hi)} | {f(mx)} | {filled} / {len(hi + mx)} |")
    L += ["", "## Exp 5: MinAtar networks written from scratch (mean score, 1,000 episodes)", "",
          "| Game | PPO 100M | PPO 500M | Astra | Opus xhigh (each run) | Opus max (each run) |", "|---|---|---|---|---|---|"]
    for g in ("breakout", "seaquest", "asterix"):
        f = lambda rs: ", ".join(f"{r['score']:.1f}" if r.get("score") is not None else ("fail" if not r["ok"] else "…")
                                 for r in rs) or "—"
        L.append(f"| {g} | {PPO5[g][0]} | {PPO5[g][1]} | {ASTRA['exp5_write_from_scratch'][g]} | "
                 f"{f(runs_of('exp5_write_from_scratch', g, 'xhigh'))} | {f(runs_of('exp5_write_from_scratch', g, 'max'))} |")
    L += ["", "## Exp 4: fix the car (success on test episodes; share of weights changed)", "",
          "| Run | Effort | Statements applied | Weights changed | Success (Kinetix) |", "|---|---|---|---|---|",
          "| Astra | high | 51 / 51 | 3.54 % | 1,024 / 1,024 |"]
    for eff in ("high", "max"):
        for r in runs_of("exp4_fix_car", "oneshot_repair", eff):
            if not r["ok"]:
                L.append(f"| Opus r{r['rep']} | {eff} | fail ({r['stop_reason']}) | | |")
                continue
            er, ch, k = r.get("edit_report", {}), r.get("changes", {}), r.get("kinetix")
            ks = (f"{k['test']['success'] * 100:.1f} %" if k and "test" in k else "pending")
            L.append(f"| Opus r{r['rep']} | {eff} | {er.get('applied')} / {er.get('statements')} | "
                     f"{ch.get('frac_changed', 0) * 100:.2f} % | {ks} |")
    done = [r for r in rows if r["ok"]]
    L += ["", "## Run health", "",
          f"- runs finished: {len(done)} / {len(rows)}; failed: {len(rows) - len(done)}; all tool-free: "
          f"{all(r['tool_free'] for r in done)}",
          f"- list-price equivalent so far: ${sum(r['list_usd'] or 0 for r in rows):.2f} (subscription, not billed)"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    rows = [r for d in sorted(glob.glob(os.path.join(RUNS, "exp*", "*")))
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "job.json"))
            for r in [score_run(d)] if r]
    json.dump(rows, open(os.path.join(RES, "scores.json"), "w"), indent=1)
    md = table(rows)
    open(os.path.join(RES, "comparison.md"), "w").write(md)
    print(md)
