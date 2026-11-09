"""Assemble the public repo (github.com/sidd20228/wsp-opus) from this working tree, in the layout of
github.com/RikiyaT/lm_understand-logs: one folder per experiment (README, runs/, eval/, networks/), harness/, plus
results/ (checks and deviations) and docs/ (the blog page, served by GitHub Pages).

  .venv/bin/python tools/build_release.py <out_dir>

Rules:
- Local paths are replaced: the project root -> <repo>, the home folder -> <home>.
- Left out (as the original release did, or because they are large or not ours to publish): the A/B Gemma releases
  and retrained checkpoints, per-game records of games passes (games.json, boards.npy, valid.npy), whole-network
  copies the model saved as .npz in its Go work folder, attention-tool arrays, and any file over 25 MB.
- docs/ is copied from site/ (the blog), and its numbers come from the results/ files written here.
"""
import glob
import json
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOME = os.path.expanduser("~")
RUNS = os.path.join(ROOT, "runs", "opus55")
TEXT = (".json", ".jsonl", ".md", ".txt", ".log", ".py", ".html", ".css", ".js", ".toml", ".out")
MAX_BYTES = 25 << 20
SKIP = [r"/releases/", r"/_smoke", r"/pass_\d+/(games\.json|boards\.npy|valid\.npy)$", r"/work/[^/]*\.npz$",
        r"/work/attn/r\d+/t\d+_[AB]\.npy$", r"\.pull\.tgz$", r"\.push\.tgz$", r"/__pycache__/", r"\.DS_Store$"]
EXPS = {
    "exp1_which_game": "exp1_which_game", "exp2_diffusion": "exp2_diffusion", "exp3_language": "exp3_language",
    "exp4_fix_car": "exp4_fix_car", "exp5_write_from_scratch": "exp5_write_from_scratch",
    "exp6_go_from_scratch": "exp6_go_from_scratch", "exp7_sycophancy": "exp7_sycophancy", "exp8_math": "exp8_math",
}


VM = [("<vm-user>@<vm-host>", "<vm-user>@<vm-host>"), ("<vm-host>", "<vm-host>"),
      ("<vm-home>", "<vm-home>"), ("<vm-user>", "<vm-user>")]   # the H100 VM's address stays private


def scrub(text):
    text = text.replace(ROOT, "<repo>")
    text = text.replace(HOME, "<home>")
    for a, b in VM:
        text = text.replace(a, b)
    return text


def copy_file(src, dst, report):
    rel = src.replace(ROOT, "")
    if any(re.search(p, rel) for p in SKIP):
        report["skipped"].append(rel)
        return
    if os.path.getsize(src) > MAX_BYTES:
        report["too_big"].append((rel, os.path.getsize(src)))
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if src.endswith(TEXT):
        data = open(src, encoding="utf-8", errors="surrogateescape").read()
        open(dst, "w", encoding="utf-8", errors="surrogateescape").write(scrub(data))
    else:
        shutil.copy2(src, dst)
    report["files"] += 1
    report["bytes"] += os.path.getsize(dst)


def copy_tree(src, dst, report):
    for dirpath, _, files in os.walk(src):
        for f in files:
            s = os.path.join(dirpath, f)
            copy_file(s, os.path.join(dst, os.path.relpath(s, src)), report)


def main(out):
    report = {"files": 0, "bytes": 0, "skipped": [], "too_big": []}
    for d in list(EXPS.values()) + ["harness", "results", "docs"]:
        shutil.rmtree(os.path.join(out, d), ignore_errors=True)
    # runs
    for exp in EXPS:
        src = os.path.join(RUNS, exp)
        if os.path.isdir(src):
            copy_tree(src, os.path.join(out, EXPS[exp], "runs"), report)
    # networks the model wrote (primary and all replicates): Exp 4, 5, 6
    for f in glob.glob(os.path.join(RUNS, "exp5_write_from_scratch", "*", "net.npz")):
        copy_file(f, os.path.join(out, "exp5_write_from_scratch", "networks",
                                  os.path.basename(os.path.dirname(f)) + ".npz"), report)
    for f in glob.glob(os.path.join(RUNS, "exp4_fix_car", "*", "edited_actor.npz")):
        copy_file(f, os.path.join(out, "exp4_fix_car", "networks",
                                  os.path.basename(os.path.dirname(f)) + "_edited_actor.npz"), report)
    for d in glob.glob(os.path.join(RUNS, "exp6_go_from_scratch", "budget_*")):
        net = os.path.join(d, "work", "net")
        st = os.path.join(d, "state.json")
        if os.path.isdir(net) and os.path.exists(st) and json.load(open(st))["done"]:
            copy_tree(net, os.path.join(out, "exp6_go_from_scratch", "networks", os.path.basename(d)), report)
    # eval: the scores behind each number
    sc = os.path.join(ROOT, "results", "phase1", "scores.json")
    if os.path.exists(sc):
        rows = json.load(open(sc))
        for exp in ("exp1_which_game", "exp2_diffusion", "exp4_fix_car", "exp5_write_from_scratch"):
            os.makedirs(os.path.join(out, exp, "eval"), exist_ok=True)
            json.dump([r for r in rows if r["exp"] == exp], open(os.path.join(out, exp, "eval", "scores.json"), "w"),
                      indent=1)
    for src, exp in (("results/phase2/exp3_scores.json", "exp3_language"),
                     ("results/phase3/exp6_scores.json", "exp6_go_from_scratch"),
                     ("results/phase2/exp7_scores.json", "exp7_sycophancy")):
        if os.path.exists(os.path.join(ROOT, src)):
            copy_file(os.path.join(ROOT, src), os.path.join(out, exp, "eval", "scores.json"), report)
    # results: checks, deviations, comparison tables
    copy_tree(os.path.join(ROOT, "results"), os.path.join(out, "results"), report)
    copy_file(os.path.join(ROOT, "REPLICATION_PLAN.md"), os.path.join(out, "results", "REPLICATION_PLAN.md"), report)
    # harness: our code (the wsp package and these tools)
    copy_tree(os.path.join(ROOT, "wsp"), os.path.join(out, "harness", "wsp"), report)
    copy_tree(os.path.join(ROOT, "tools"), os.path.join(out, "harness", "tools"), report)
    # experiment READMEs and the blog
    copy_tree(os.path.join(ROOT, "site", "readmes"), out, report)
    copy_tree(os.path.join(ROOT, "site", "docs"), os.path.join(out, "docs"), report)
    json.dump(report, open(os.path.join(ROOT, "results", "release_report.json"), "w"), indent=1)
    print(f"{report['files']} files, {report['bytes'] / 1e6:.1f} MB; skipped {len(report['skipped'])}, "
          f"too big {len(report['too_big'])}")
    for r in report["too_big"]:
        print("  too big:", r)


if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1]))
