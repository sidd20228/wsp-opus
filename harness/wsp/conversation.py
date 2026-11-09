"""A resumable multi-turn conversation with Opus 5.5 (Experiments 3, 6, 7, 8).

Layout of a conversation folder (as the logs):
  message.txt          the first message
  turns/tNNN/          prompt.txt, events.jsonl, answer.md, run.json, stderr.log (from wsp.cc)
  tool_calls.jsonl     one line per harness action (code run, pass, ...)
  state.json           session id, working folder, turns, counters, the next message to send, done
  final_answer.md      the final reply
  loop.log             one line per turn

The experiment supplies `handle(reply_text, state) -> next message or None` (None = the reply was the final
answer). After a crash, `run()` continues from state.json: the pending message is sent again in the same session.
"""
import json
import os
import re
import time

from wsp import cc

BLOCK = r"```{lang}[^\n]*\n(.*?)(?:```|\Z)"


def block(text, lang):
    """The first ```<lang> block of a reply (an unterminated block runs to the end), or None."""
    m = re.search(BLOCK.format(lang=re.escape(lang)), text, re.S)
    return m.group(1) if m else None


class Conversation:
    def __init__(self, folder, effort, first_message=None, max_turns=1000):
        self.folder, self.effort, self.max_turns = folder, effort, max_turns
        os.makedirs(folder, exist_ok=True)
        self.state_path = os.path.join(folder, "state.json")
        if os.path.exists(self.state_path):
            self.state = json.load(open(self.state_path))
        else:
            conv = cc.new_conversation()
            self.state = {"session_id": conv["session_id"], "cwd": conv["cwd"], "turns": 0, "effort": effort,
                          "next": first_message, "done": False, "counters": {}, "failures": 0}
            open(os.path.join(folder, "message.txt"), "w").write(first_message)
            self.save()

    def save(self):
        json.dump(self.state, open(self.state_path, "w"), indent=1)

    def log(self, msg):
        line = time.strftime("[%Y-%m-%dT%H:%M:%S] ") + msg
        print(f"{os.path.basename(self.folder)} {line}", flush=True)
        with open(os.path.join(self.folder, "loop.log"), "a") as f:
            f.write(line + "\n")

    def tool_call(self, rec):
        with open(os.path.join(self.folder, "tool_calls.jsonl"), "a") as f:
            f.write(json.dumps({"t": time.time(), **rec}) + "\n")

    def count(self, key, inc=0):
        c = self.state["counters"]
        c[key] = c.get(key, 0) + inc
        self.save()
        return c[key]

    def _unhandled_reply(self):
        """The last completed turn's reply if the harness crashed before acting on it (the pending message is still
        the one that turn already sent), else None."""
        k = self.state["turns"]
        if k < 1 or self.state["done"]:
            return None
        d = os.path.join(self.folder, "turns", f"t{k:03d}")
        try:
            sent = open(os.path.join(d, "prompt.txt")).read()
            reply = open(os.path.join(d, "answer.md")).read()
        except OSError:
            return None
        return reply if sent == self.state["next"] else None

    def _repair_counters(self):
        """Counters bumped by a harness step that then crashed: reset them from the logged tool calls."""
        p = os.path.join(self.folder, "tool_calls.jsonl")
        recs = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
        c = self.state["counters"]
        for key, field in (("runs", "run"), ("tool_calls", "call")):
            if key in c:
                c[key] = max([r[field] for r in recs if isinstance(r.get(field), int)] or [0])
        self.save()

    def run(self, handle):
        text = self._unhandled_reply()
        if text is not None:   # act on the saved reply instead of sending the old message again
            self._repair_counters()
            self.log(f"turn {self.state['turns']}: re-handling the saved reply after a harness crash")
            nxt = handle(text, self)
            if nxt is None:
                open(os.path.join(self.folder, "final_answer.md"), "w").write(text)
                self.state["done"], self.state["next"] = True, None
            else:
                self.state["next"] = nxt
            self.save()
        while not self.state["done"] and self.state["turns"] < self.max_turns:
            msg = self.state["next"]
            k = self.state["turns"] + 1
            conv = {"session_id": self.state["session_id"], "cwd": self.state["cwd"], "turns": self.state["turns"]}
            self.log(f"turn {k}: {'new session' if k == 1 else 'resume'} ({len(msg):,} chars)")
            rec = cc.turn(conv, msg, self.effort, os.path.join(self.folder, "turns", f"t{k:03d}"), log=self.log)
            if not rec["ok"]:
                self.state["failures"] += 1
                self.state["last_failure"] = {"turn": k, "stop_reason": rec["stop_reason"],
                                              "attempts": len(rec["attempts"])}
                self.save()
                self.log(f"turn {k}: FAILED ({rec['stop_reason']}); stopping this conversation")
                return False
            self.state["turns"] = k
            self.state["usage_last"] = rec.get("usage")
            self.save()
            text = rec["text"]
            nxt = handle(text, self)
            if nxt is None:
                open(os.path.join(self.folder, "final_answer.md"), "w").write(text)
                self.state["done"] = True
                self.state["next"] = None
                self.save()
                self.log(f"turn {k}: final answer ({len(text):,} chars)")
            else:
                self.state["next"] = nxt
                self.save()
        return self.state["done"]
