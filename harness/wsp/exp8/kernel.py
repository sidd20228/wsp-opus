"""The editor's persistent CPU Python kernel (Experiment 8) on the H100 VM, inside bubblewrap.

    from wsp.exp8.kernel import Kernel
    k = Kernel("p50_r0")          # starts (or re-attaches to) the run's kernel; /work = ~/wsp-opus/work/exp8/p50_r0/work
    text = k.run(code)            # the output as the harness showed it (inside "OUTPUT of Python run i ...")
    k.close()                     # at the end of the run (stops the kernel; /work stays)

The sandbox is like the original harness's: bwrap --unshare-all (no network) --new-session --clearenv, /usr and the
kernel venv read-only, /model and /practice read-only binds, /work the run folder (read-write; the forward-pass tool
writes its results straight into it), RLIMIT_AS 110 GiB, and the kernel restarted when it goes over 96 GiB resident.
One long-lived supervisor per run keeps a kernel process whose globals persist (prelude: NAMES, load, load_original,
save_edit, drop_edit, edited, tok, ...); runs reach it through a unix socket in ~/wsp-opus/work/exp8/<run>/kernel/.

Output format (as the original): stdout and stderr in order, a last-line expression's value, non-ASCII as '?', a
non-blank line shown at most 2 times, cut at 10,000 characters, "[timed out after 900 s: the kernel was restarted ...]"
and the memory / died notices, "(no output)" if empty. If the kernel was gone before a run (e.g. the VM rebooted) a new
one is started and the output begins with "[note: the kernel was restarted, so its variables are gone]".
"""
import json
import os
import shlex
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from wsp import remote as R  # noqa: E402
import vmjob  # noqa: E402

TIMEOUT = 900
MAX_OUT = 10_000
MAX_REPEAT = 2
DATA = f"{R.BASE}/data/exp8"
CODE = f"{R.BASE}/code/wsp/exp8"
VENV = f"{R.BASE}/venvs/exp8_kernel"
NOTE_RESTART = "[note: the kernel was restarted, so its variables are gone]"

EVAL_PROMPT_FORMAT = (
    "The evaluation prompt, exactly as the model sees it (thinking on), for a problem P:\n"
    "PREFIX + P + SUFFIX, with\n"
    "PREFIX = '<|im_start|>user\\n'\n"
    "SUFFIX = '\\nPlease reason step by step, and put your final answer within \\\\boxed{}.<|im_end|>\\n"
    "<|im_start|>assistant\\n<think>\\n'\n"
    "The model's answer follows SUFFIX directly. To score a text with a teacher-forced pass, send PREFIX + P + SUFFIX + "
    "answer as one raw text in \"texts\" (\"chats\" re-renders assistant turns and drops their reasoning).\n")


def format_output(raw, status="ok", timeout=TIMEOUT, returncode=None):
    """The harness's view of one run's raw output."""
    text = raw.encode("ascii", "replace").decode("ascii")
    seen, kept, dropped = {}, [], 0
    for line in text.split("\n"):
        if line.strip():
            seen[line] = seen.get(line, 0) + 1
            if seen[line] > MAX_REPEAT:
                dropped += 1
                continue
        kept.append(line)
    text = "\n".join(kept)
    if len(text) > MAX_OUT:
        text = text[:MAX_OUT] + f"\n[output cut at {MAX_OUT:,} characters]"
    if dropped:
        text += f"\n[harness: {dropped} repeated lines not shown (a line is shown at most {MAX_REPEAT} times per output)]"
    if status == "timeout":
        text += (("\n" if text else "") +
                 f"[timed out after {int(timeout)} s: the kernel was restarted, so its variables are gone]")
    elif status == "memory":
        text += (("\n" if text else "") +
                 "[the kernel went over the 96 GB memory limit and was restarted, so its variables are gone]")
    elif status == "died":
        text += (("\n" if text else "") +
                 f"[the kernel died (exit code {returncode}) and was restarted, so its variables are gone]")
    return text if text.strip() else "(no output)"


class Kernel:
    def __init__(self, run_id, threads=16, log=print):
        assert run_id and all(c.isalnum() or c in "-_" for c in run_id), run_id
        self.run_id, self.threads, self.log = run_id, threads, log
        self.root = f"{R.BASE}/work/exp8/{run_id}"
        self.sock = f"{self.root}/kernel/kernel.sock"
        self.history = []
        if not self.alive():
            self._start()
        else:
            self.log(f"[kernel] attached to the running kernel of {run_id}")

    # ------------------------------------------------------------------------------------------ VM side
    def _launch_script(self):
        q = shlex.quote
        r = self.root
        t = str(self.threads)
        bw = ["prlimit", f"--as={110 << 30}", "bwrap", "--unshare-all", "--new-session", "--clearenv",
              "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
              "--symlink", "usr/bin", "/bin", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
              "--ro-bind", f"{R.BASE}/python", f"{R.BASE}/python", "--ro-bind", VENV, VENV,
              "--ro-bind", f"{DATA}/models/Qwen3.5-0.8B-text", "/model", "--ro-bind", f"{DATA}/practice", "/practice",
              "--bind", f"{r}/work", "/work", "--bind", f"{r}/kernel", "/kernel",
              "--ro-bind", f"{CODE}/kernel_server.py", "/opt/kernel/kernel_server.py",
              "--ro-bind", f"{CODE}/kernel_prelude.py", "/opt/kernel/kernel_prelude.py",
              "--chdir", "/", "--setenv", "HOME", "/tmp", "--setenv", "PATH", "/usr/bin",
              "--setenv", "OMP_NUM_THREADS", t, "--setenv", "MKL_NUM_THREADS", t, "--setenv", "OPENBLAS_NUM_THREADS", t,
              "--setenv", "CUDA_VISIBLE_DEVICES", "", "--setenv", "PYTHONUNBUFFERED", "1",
              "--setenv", "PYTHONDONTWRITEBYTECODE", "1", "--setenv", "KERNEL_SOCK", "/kernel/kernel.sock",
              f"{VENV}/bin/python", "-u", "/opt/kernel/kernel_server.py", "supervisor"]
        return "\n".join([
            f"mkdir -p {q(r)}/work/edited {q(r)}/kernel && rm -f {q(self.sock)}",
            f"test -e {q(r)}/work/eval_prompt_format.txt || cat > {q(r)}/work/eval_prompt_format.txt <<'__EOF__'\n"
            f"{EVAL_PROMPT_FORMAT}__EOF__",
            f"cd {q(r)}",
            "setsid nohup " + " ".join(q(c) for c in bw) +
            f" > {q(r)}/kernel/supervisor.log 2>&1 < /dev/null & echo $! > {q(r)}/kernel/pid",
            f"for i in $(seq 1 240); do test -S {q(self.sock)} && break; sleep 0.5; done; test -S {q(self.sock)}"])

    def _start(self):
        self.stop()
        R.sh(self._launch_script(), timeout=300)
        self.log(f"[kernel] started the kernel of {self.run_id}")

    def stop(self):
        """kill the run's supervisor and kernel (the bwrap process tree)."""
        p = f"{self.root}/kernel/pid"
        R.sh(f"killtree() {{ for c in $(pgrep -P $1); do killtree $c; done; kill -KILL $1 2>/dev/null; }}; "
             f"test -f {p} && killtree $(cat {p}); rm -f {p} {shlex.quote(self.sock)}; true", check=False)

    def _request(self, req, timeout, tries="20"):
        """one request to the kernel, as a detached VM job (a network blip does not lose the run or its output)."""
        cmd = (f"KERNEL_SOCK={shlex.quote(self.sock)} KERNEL_CONNECT_TRIES={tries} {VENV}/bin/python -I "
               f"{CODE}/kernel_server.py client")
        self._n = getattr(self, "_n", 0) + 1
        try:
            out = vmjob.run(f"kernel_{self.run_id}_{int(time.time() * 1000)}_{self._n}", cmd, input_text=json.dumps(req),
                            timeout=timeout, poll=2, first_poll=0.1, log=f"{self.root}/kernel/client.log")
        except RuntimeError as e:
            return {"status": "transport", "output": str(e)}
        try:
            return json.loads(out)
        except (TypeError, ValueError):
            return {"status": "transport", "output": (out or "")[-2000:]}

    def alive(self):
        r = self._request({"op": "ping"}, timeout=60, tries="2")
        return r.get("status") == "ok"

    # ------------------------------------------------------------------------------------------ public API
    def run(self, code, timeout=TIMEOUT):
        """Run one code block; returns the output text as the harness showed it to the model."""
        note = None
        r = self._request({"code": code, "timeout": timeout}, timeout=timeout + 300)
        if r.get("status") == "nokernel":
            self.log("[kernel] no kernel running; starting a new one")
            self._start()
            note = NOTE_RESTART
            r = self._request({"code": code, "timeout": timeout}, timeout=timeout + 300)
        if r.get("status") in ("transport", "error", "nokernel"):
            self.log(f"[kernel] request failed: {r}")
            if not self.alive():
                self._start()
            r = {"output": "", "status": "died", "returncode": None}
        if r.get("restarted_before"):
            note = NOTE_RESTART
        self.history.append({"t": time.time(), "status": r.get("status"), "secs": r.get("secs"),
                             "peak_rss_gb": r.get("peak_rss_gb")})
        text = format_output(r.get("output", ""), r.get("status", "ok"), timeout, r.get("returncode"))
        return (note + "\n" + text) if note else text

    def restart(self):
        """Restart the kernel process (variables lost)."""
        return self._request({"op": "restart"}, timeout=400)

    def commit(self):          # kept for the Modal-era API: /work is the run folder itself
        return True

    def reload(self):
        return True

    def ls_work(self):
        return json.loads(R.sh(f"python3 -c \"import os,json;print(json.dumps(sorted(os.listdir('{self.root}/work'))))\""
                               ).stdout)

    def read_work_file(self, rel):
        return R.sh(f"cat {shlex.quote(self.root + '/work/' + rel)}").stdout

    def close(self):
        self.stop()
