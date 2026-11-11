"""Experiment 8: Opus edits Qwen3.5-0.8B's weights to make it better at math (Python kernel + forward passes).

The task message is the original run's final `message.txt` for the budget (runs/p<B>c/message.txt: the turn-1 prompt
with the "EXPLAIN EVERY STEP" paragraph), followed by notices 001 and 002, all in turn 1 (the original delivered the
explanation rule from turn 16-52 and the notices mid-run; deviations.md). Each reply's first block decides the action:
  ```python   one run in the run's persistent kernel (wsp/exp8/kernel.py; at most 300)
              -> "OUTPUT of Python run k of at most 300:\n```\n...\n```"
  ```forward  one forward-pass request (wsp/exp8/fwd.py); a failed request is not counted
              -> pass_message(...) ("Forward pass n of B done (...)" / "The forward pass failed and was not counted: ...")
A block with no explanation before it is sent back unrun (original wording), at most twice in a row. A reply without a
block is the final answer once all passes are used; before that it gets a reminder (our wording; the logs have none).
Every 40 turns the conversation is restarted with the original handoff message (task + state + notebook + last result).

  python wsp/exp8_run.py run <budget> <rep> [effort]       default effort max
"""
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from wsp import cc, remote as R  # noqa: E402

LOGS = os.path.join(ROOT, "lm_understand-logs", "exp8_math")
OUT = os.path.join(ROOT, "runs", "opus55", "exp8_math")
MAX_RUNS, HANDOFF_EVERY, MAX_UNEXPLAINED = 300, 40, 2
BLOCK = re.compile(r"```(python|forward)[^\n]*\n(.*?)(?:```|\Z)", re.S)
EXPLAIN = ("EXPLAIN EVERY STEP. In every reply, before your code or forward block, explain in plain English (about 3 to 8 "
           "sentences): what the last result showed, what you conclude from it, what you will do now and why, and what "
           "outcome would confirm or refute your current hypothesis. In your final answer, also summarize the path that "
           "led to your edits. These explanations are read by the experimenters.")
UNEXPLAINED = ("Your reply had no explanation, so its block was NOT run (no Python run or forward pass was used). "
               + EXPLAIN + " Send the block again, preceded by your explanation.")
NOTE = "[note from the experimenters] "


def task_message(budget):
    return open(os.path.join(LOGS, "runs", f"p{budget}c", "message.txt")).read()


def first_message(budget):
    notices = [open(os.path.join(LOGS, "notices", n)).read().strip() for n in ("001_pass_size.txt",
                                                                              "002_tool_facts.txt")]
    return task_message(budget) + "".join("\n\n" + NOTE + n for n in notices)


class Run:
    def __init__(self, budget, rep, effort="max", max_turns=1000, tag=""):
        self.budget, self.effort, self.max_turns = budget, effort, max_turns
        self.run_id = f"p{budget}_r{rep}" + ("" if effort == "max" else f"_{effort}") + tag
        self.folder = os.path.join(OUT, self.run_id)
        os.makedirs(self.folder, exist_ok=True)
        self.sp = os.path.join(self.folder, "state.json")
        if os.path.exists(self.sp):
            self.s = json.load(open(self.sp))
        else:
            conv = cc.new_conversation()
            self.s = {"budget": budget, "effort": effort, "run_id": self.run_id, "session_id": conv["session_id"],
                      "cwd": conv["cwd"], "session_turns": 0, "turns": 0, "since_handoff": 0, "handoffs": 0,
                      "runs": 0, "passes": 0, "requests": 0, "unexplained": 0, "unexplained_streak": 0,
                      "reminders": 0, "next": first_message(budget), "done": False, "failures": 0}
            open(os.path.join(self.folder, "message.txt"), "w").write(self.s["next"])
            self.save()
        self.kernel = None

    # ------------------------------------------------------------------ bookkeeping
    def save(self):
        json.dump(self.s, open(self.sp, "w"), indent=1)

    def log(self, msg):
        line = time.strftime("[%Y-%m-%dT%H:%M:%S] ") + msg
        print(f"{self.run_id} {line}", flush=True)
        with open(os.path.join(self.folder, "loop.log"), "a") as f:
            f.write(line + "\n")

    def record(self, rec):
        with open(os.path.join(self.folder, "tool_calls.jsonl"), "a") as f:
            f.write(json.dumps({"t": time.time(), **rec}) + "\n")

    def K(self):
        if self.kernel is None:
            from wsp.exp8.kernel import Kernel
            self.kernel = Kernel(self.run_id, log=self.log)
        return self.kernel

    # ------------------------------------------------------------------ handoff (the original's wording)
    def handoff_message(self, last_result):
        k = self.K()
        work = f"{R.BASE}/work/exp8/{self.run_id}/work"
        files = k.ls_work()
        ed = R.sh(f"ls {work}/edited 2>/dev/null | sed 's/\\.npy$//'", check=False).stdout.decode().split()
        notes = R.sh(f"tail -c 8000 {work}/notes.md 2>/dev/null", check=False).stdout.decode(errors="replace")
        return (task_message(self.budget) +
                f"\n\nTHIS IS A RESTARTED CONVERSATION (to keep it short): {self.s['turns']} replies happened before. "
                f"Forward passes used: {self.s['passes']} of {self.budget}; Python runs used: {self.s['runs']} of "
                f"{MAX_RUNS}. Your kernel is still running with its variables. Tensors currently edited: {sorted(ed)}. "
                f"Files in /work: {files}.\nYOUR NOTEBOOK (/work/notes.md, last 8,000 characters):\n```\n{notes}\n```\n"
                f"Continue from here.\n\nLast result:\n{last_result}")

    # ------------------------------------------------------------------ one reply -> the next message
    def handle(self, text):
        s = self.s
        m = BLOCK.search(text)
        if m is None:
            if s["passes"] < self.budget:
                s["reminders"] += 1
                self.log(f"final answer with {self.budget - s['passes']} passes left: reminder")
                return (f"You have used {s['passes']} of {self.budget} forward passes. The session continues until all "
                        f"{self.budget} forward passes are used (replying without a code block does not end it "
                        "earlier).")
            return None
        expl = text[:m.start()].strip()
        if len(expl) < 20 and s["unexplained_streak"] < MAX_UNEXPLAINED:
            s["unexplained"] += 1
            s["unexplained_streak"] += 1
            self.log("block without explanation: sent back unrun")
            return UNEXPLAINED
        s["unexplained_streak"] = 0
        with open(os.path.join(self.folder, "explanations.md"), "a") as f:
            f.write(f"## turn {s['turns']}\n\n{expl}\n\n")
        kind, body = m.group(1), m.group(2)
        if kind == "python":
            if s["runs"] >= MAX_RUNS:
                return f"You have used all {MAX_RUNS} Python runs."
            out = self.K().run(body)
            s["runs"] += 1
            k = s["runs"]
            from wsp.exp8.kernel import NOTE_RESTART
            note = ""
            if out.startswith(NOTE_RESTART):
                note, out = NOTE_RESTART + "\n", out[len(NOTE_RESTART):].lstrip("\n")
            self.record({"kind": "python", "run": k, "code": body, "out": out[:8000]})
            self.log(f"python run {k} ({len(body):,} chars of code, {len(out):,} chars out); "
                     f"passes {s['passes']}/{self.budget}")
            return f"{note}OUTPUT of Python run {k} of at most {MAX_RUNS}:\n```\n{out}\n```"
        # forward pass
        if s["passes"] >= self.budget:
            return (f"No forward passes left (all {self.budget} used). Make your final choice and reply without a "
                    "code block.")
        from wsp.exp8.fwd import forward_pass, pass_message
        s["requests"] += 1
        k = s["requests"]
        os.makedirs(os.path.join(self.folder, "requests"), exist_ok=True)
        open(os.path.join(self.folder, "requests", f"fwd_{k}.claimed"), "w").write(body)
        res = forward_pass(body, k, self.run_id, kernel=self.K())
        ok = json.loads(res).get("ok", False)
        if ok:
            s["passes"] += 1
        open(os.path.join(self.folder, "requests", f"fwd_{k}.done.json"), "w").write(res)
        self.record({"kind": "forward", "request": k, "ok": ok, "pass": s["passes"] if ok else None})
        self.log(f"forward request {k}: {'pass %d/%d' % (s['passes'], self.budget) if ok else 'FAILED (not counted)'}")
        return pass_message(res, k, s["passes"], self.budget)

    # ------------------------------------------------------------------ the loop
    def unhandled_reply(self):
        k = self.s["turns"]
        d = os.path.join(self.folder, "turns", f"t{k:03d}")
        try:
            if k and not self.s["done"] and open(os.path.join(d, "prompt.txt")).read() == self.s["next"]:
                return open(os.path.join(d, "answer.md")).read()
        except OSError:
            pass
        return None

    def after_reply(self, text):
        s = self.s
        nxt = self.handle(text)
        if nxt is None:
            open(os.path.join(self.folder, "final_answer.md"), "w").write(text)
            s["done"], s["next"] = True, None
        elif s["since_handoff"] >= HANDOFF_EVERY:      # a fresh conversation with the original handoff message
            conv = cc.new_conversation()
            s["session_id"], s["cwd"], s["session_turns"], s["since_handoff"] = conv["session_id"], conv["cwd"], 0, 0
            s["handoffs"] += 1
            s["next"] = self.handoff_message(nxt)
            self.log(f"handoff {s['handoffs']}: new conversation")
        else:
            s["next"] = nxt
        self.save()

    def run(self):
        text = self.unhandled_reply()
        if text is not None:
            self.log(f"turn {self.s['turns']}: re-handling the saved reply after a harness crash")
            self.after_reply(text)
        while not self.s["done"] and self.s["turns"] < self.max_turns:
            s = self.s
            k = s["turns"] + 1
            conv = {"session_id": s["session_id"], "cwd": s["cwd"], "turns": s["session_turns"]}
            self.log(f"turn {k}: {'new conversation' if s['session_turns'] == 0 else 'resume'} ({len(s['next']):,} chars)")
            rec = cc.turn(conv, s["next"], self.effort, os.path.join(self.folder, "turns", f"t{k:03d}"), log=self.log)
            if not rec["ok"]:
                s["failures"] += 1
                self.save()
                self.log(f"turn {k}: FAILED ({rec['stop_reason']}); stopping")
                return False
            s["turns"], s["session_turns"], s["since_handoff"] = k, s["session_turns"] + 1, s["since_handoff"] + 1
            self.save()
            self.after_reply(rec["text"])
        if self.s["done"]:
            self.log(f"final answer; {self.s['runs']} runs, {self.s['passes']} passes, {self.s['handoffs']} handoffs")
            self.K().close()                # stops the kernel; /work (the submitted /work/edited) stays on the VM
        return self.s["done"]


if __name__ == "__main__":
    if sys.argv[1] == "run":
        ok = Run(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else "max").run()
        print("done" if ok else "stopped")
