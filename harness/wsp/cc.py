"""One Opus 5.5 turn through the Claude Code CLI (`claude -p`), the analog of the original `codex exec` /
`codex exec resume` calls (REPLICATION_PLAN.md, section 2).

Every call:
  - runs in a clean environment (env -i), so nothing from the calling app leaks into the model's context;
  - has no tools, no MCP servers, no settings, skills, slash commands or Chrome;
  - uses the original one-line system prompt and the experiment's effort level;
  - runs in its own empty working folder (one per conversation) outside the project.

Files written to `turn_dir` (same names as the logs where they exist):
  prompt.txt     what was sent
  events.jsonl   the raw stream-json event stream of the CLI
  answer.md      the model's reply (the result text)
  run.json       model, effort, session, command, usage, stop reason, wall time, tool audit, attempts
  stderr.log     the CLI's stderr (last attempt)
"""
import glob
import json
import os
import shutil
import subprocess
import time
import uuid

MODEL = "claude-opus-5-5"
SYSTEM = "Answer the user's request."
CLAUDE = shutil.which("claude") or "/opt/homebrew/bin/claude"
CWD_ROOT = os.path.expanduser("~/.wsp/c")          # neutral, empty working folders, one per conversation
PROJECTS = os.path.expanduser("~/.claude/projects")
BACKOFF_MIN = [2, 5, 15, 30, 60, 60, 60, 60]        # waits between failed attempts (usage limits, outages)

FLAGS = ["--tools", "", "--system-prompt", SYSTEM, "--safe-mode", "--setting-sources", "",
         "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}', "--disable-slash-commands", "--no-chrome",
         "--autocompact", "1000000", "--output-format", "stream-json", "--verbose"]


def clean_env():
    home = os.path.expanduser("~")
    user = os.environ.get("USER", os.path.basename(home))
    return {"HOME": home, "USER": user, "LOGNAME": user, "PATH": "/opt/homebrew/bin:/usr/bin:/bin",
            "TERM": "dumb", "LANG": "en_US.UTF-8", "DISABLE_AUTOUPDATER": "1"}


def cli_version():
    r = subprocess.run([CLAUDE, "--version"], capture_output=True, text=True, env=clean_env())
    return r.stdout.strip()


def new_conversation():
    """A fresh session id and its own empty working folder."""
    sid = str(uuid.uuid4())
    cwd = os.path.join(CWD_ROOT, sid[:8])
    os.makedirs(cwd, exist_ok=True)
    return {"session_id": sid, "cwd": cwd, "turns": 0}


def _session_file(sid):
    hits = glob.glob(os.path.join(PROJECTS, "*", sid + ".jsonl"))
    return hits[0] if hits else None


def _parse(stdout):
    events, init, result, rate = [], None, None, []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        events.append(d)
        t = d.get("type")
        if t == "system" and d.get("subtype") == "init":
            init = d
        elif t == "result":
            result = d
        elif t == "rate_limit_event":
            rate.append(d)
    blocks = [c.get("type") for d in events if d.get("type") == "assistant"
              for c in d.get("message", {}).get("content", [])]
    compacted = any(d.get("type") == "system" and "compact" in str(d.get("subtype", "")) for d in events)
    return events, init, result, rate, blocks, compacted


def turn(conv, prompt, effort, turn_dir, timeout_s=4 * 3600, max_attempts=len(BACKOFF_MIN) + 1,
         persist=True, log=print):
    """Send one user message in conversation `conv` (from new_conversation()). Returns the run record."""
    os.makedirs(turn_dir, exist_ok=True)
    with open(os.path.join(turn_dir, "prompt.txt"), "w") as f:
        f.write(prompt)
    first = conv["turns"] == 0
    sess_args = (["--session-id", conv["session_id"]] if first else ["--resume", conv["session_id"]])
    if not persist:
        sess_args = ["--no-session-persistence"]
    cmd = [CLAUDE, "-p", "--model", MODEL, "--effort", effort] + FLAGS + sess_args

    backup = None
    sf = None if first else _session_file(conv["session_id"])
    if sf:
        backup = sf + ".bak"
        shutil.copyfile(sf, backup)

    attempts = []
    rec = None
    for a in range(max_attempts):
        t0 = time.time()
        try:
            p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=conv["cwd"],
                               env=clean_env(), timeout=timeout_s)
            stdout, stderr, code = p.stdout, p.stderr, p.returncode
        except subprocess.TimeoutExpired as e:
            stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            stderr, code = f"[timed out after {timeout_s} s]", -9
        wall = time.time() - t0
        events, init, result, rate, blocks, compacted = _parse(stdout)
        res = result or {}
        text = res.get("result") or ""
        stop = res.get("stop_reason")
        ok = (code == 0 and result is not None and not res.get("is_error") and res.get("subtype") == "success"
              and stop not in ("refusal", "max_tokens") and text.strip() != "")
        att = {"attempt": a + 1, "exit": code, "wall_s": round(wall, 1), "subtype": res.get("subtype"),
               "is_error": res.get("is_error"), "stop_reason": stop, "result_head": text[:300],
               "stderr_tail": stderr[-500:], "rate_limit_events": rate}
        attempts.append(att)
        with open(os.path.join(turn_dir, "events.jsonl"), "w") as f:
            f.write(stdout)
        with open(os.path.join(turn_dir, "stderr.log"), "w") as f:
            f.write(stderr)
        tools = (init or {}).get("tools")
        rec = {"model": MODEL, "effort": effort, "session_id": conv["session_id"], "cwd": conv["cwd"],
               "turn": conv["turns"] + 1, "cli_version": cli_version(), "command": cmd,
               "ok": ok, "stop_reason": stop, "wall_s": round(wall, 1),
               "usage": res.get("usage"), "model_usage": res.get("modelUsage"),
               "cost_usd_list_equivalent": res.get("total_cost_usd"), "num_turns": res.get("num_turns"),
               "audit": {"tool_free": tools == [] and "tool_use" not in blocks, "init_tools": tools,
                         "init_model": (init or {}).get("model"), "assistant_blocks": blocks,
                         "compacted": compacted},
               "attempts": attempts}
        if ok:
            break
        # failed: put the session back exactly as it was before this turn, then wait and retry
        if backup and os.path.exists(backup):
            shutil.copyfile(backup, sf)
        elif first:
            leftover = _session_file(conv["session_id"])
            if leftover:
                os.remove(leftover)
        if stop == "max_tokens":
            break                                   # a reply that hit the output limit is a result, not an outage
        if stop == "refusal" and sum(x["stop_reason"] == "refusal" for x in attempts) >= 2:
            break                                   # refused twice: record as a failure
        if a + 1 < max_attempts:
            wait = BACKOFF_MIN[min(a, len(BACKOFF_MIN) - 1)] * 60
            log(f"[cc] turn failed (exit {code}, {res.get('subtype')}, stop {stop}): {text[:120]!r} "
                f"{stderr[-200:]!r}; retry in {wait // 60} min")
            time.sleep(wait)
    if backup and os.path.exists(backup):
        os.remove(backup)
    with open(os.path.join(turn_dir, "answer.md"), "w") as f:
        f.write(text)
    with open(os.path.join(turn_dir, "run.json"), "w") as f:
        json.dump(rec, f, indent=1)
    if rec and rec["ok"]:
        conv["turns"] += 1
    rec["text"] = text
    return rec
