"""The editor's persistent CPU Python kernel (Experiment 8) as a long-lived Modal Sandbox.

    from wsp.exp8.kernel import Kernel
    k = Kernel("p50_r0")                 # creates (or re-attaches to) the run's sandbox; /work = volume wsp-exp8-work-p50_r0
    text = k.run(code)                   # the output as the harness showed it (inside "OUTPUT of Python run i ...")
    k.close()                            # at the end of the run (terminates the sandbox; /work stays in the volume)

The sandbox: no network, CPU only (16 cores, memory request 16 GiB, limit 104 GiB; the kernel is restarted when it
goes over 96 GiB resident), /model and /practice read-only volumes, /work the run's read-write volume (version 2),
committed after every run so that the forward-pass server sees the current edit. The kernel process has the prelude
of the first-turn prompt (NAMES, load, load_original, save_edit, drop_edit, edited, tok, ...).

Output (as the original harness): the block's stdout and stderr in order; the value of a last-line expression;
non-ASCII characters as '?'; a non-blank line is shown at most 2 times ("[harness: N repeated lines not shown (a line
is shown at most 2 times per output)]"); cut at 10,000 characters ("[output cut at 10,000 characters]"); a block over
900 s is stopped and the kernel restarted; a kernel that dies (memory) is restarted; both say that the variables are
gone. "(no output)" if there is nothing to show.
"""
import json
import os
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

TIMEOUT = 900
MAX_OUT = 10_000
MAX_REPEAT = 2
LIFETIME = 24 * 3600          # Modal's maximum sandbox lifetime; a new sandbox is made when it ends

EVAL_PROMPT_FORMAT = (
    "The evaluation prompt, exactly as the model sees it (thinking on), for a problem P:\n"
    "PREFIX + P + SUFFIX, with\n"
    "PREFIX = '<|im_start|>user\\n'\n"
    "SUFFIX = '\\nPlease reason step by step, and put your final answer within \\\\boxed{}.<|im_end|>\\n"
    "<|im_start|>assistant\\n<think>\\n'\n"
    "The model's answer follows SUFFIX directly. To score a text with a teacher-forced pass, send PREFIX + P + SUFFIX + "
    "answer as one raw text in \"texts\" (\"chats\" re-renders assistant turns and drops their reasoning).\n")

NOTE_RESTART = "[note: the kernel was restarted, so its variables are gone]"


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
    def __init__(self, run_id, cpu=16.0, memory_gib=(16, 104), create=True, log=print):
        self.run_id = run_id
        self.cpu, self.memory = cpu, (memory_gib[0] * 1024, memory_gib[1] * 1024)
        self.log = log
        self.sb = None
        self.pending_note = None
        self.history = []          # (time, status, secs, peak_rss_gb) per run
        if create:
            self._attach_or_create()

    # ------------------------------------------------------------------------------------------- sandbox
    @property
    def name(self):
        return f"kernel-{self.run_id}"

    def _app(self):
        return modal.App.lookup(C.APP_NAME, create_if_missing=True)

    def _attach_or_create(self):
        try:
            sb = modal.Sandbox.from_name(C.APP_NAME, self.name)
            if sb.poll() is None:
                self.sb = sb
                self.log(f"[kernel] attached to running sandbox {sb.object_id}")
                return
        except Exception:  # noqa: BLE001
            pass
        self._create()

    def _create(self):
        image = C.kernel_image()
        vols = {"/model": C.model_vol().read_only(), "/practice": C.practice_vol().read_only(),
                C.WORK_SYNC: C.work_vol(self.run_id)}
        self.sb = modal.Sandbox.create("python", "-u", "/opt/kernel/kernel_server.py", "supervisor", app=self._app(),
                                       name=self.name, image=image, block_network=True, timeout=LIFETIME,
                                       cpu=self.cpu, memory=self.memory, workdir="/", volumes=vols)
        self.pull_pending = False
        self.created = time.time()
        self.log(f"[kernel] created sandbox {self.sb.object_id}")
        self._wait_ready()
        self._ensure_work_files()

    def _wait_ready(self, timeout=300):
        t0 = time.time()
        while time.time() - t0 < timeout:
            r = self._request({"op": "ping"}, exec_timeout=60, quiet=True)
            if r and r.get("status") == "ok":
                return
            time.sleep(2)
        raise RuntimeError("kernel sandbox did not become ready")

    def _ensure_work_files(self):
        """/work/eval_prompt_format.txt (notice 002 refers to it) and /work/edited."""
        self._sh("mkdir -p /work/edited")
        if self._sh("test -e /work/eval_prompt_format.txt && echo yes || echo no").strip() == "no":
            with self.sb.open("/work/eval_prompt_format.txt", "w") as f:
                f.write(EVAL_PROMPT_FORMAT)
        self.commit()

    def alive(self):
        return self.sb is not None and self.sb.poll() is None

    def _request(self, req, exec_timeout, quiet=False):
        try:
            p = self.sb.exec("python", "-u", "/opt/kernel/kernel_server.py", "client", timeout=int(exec_timeout))
            p.stdin.write(json.dumps(req).encode())
            p.stdin.write_eof()
            p.stdin.drain()
            out = p.stdout.read()
            p.wait()
            return json.loads(out) if out.strip() else None
        except Exception as e:  # noqa: BLE001
            if not quiet:
                self.log(f"[kernel] request failed: {e!r}")
            return None

    # ------------------------------------------------------------------------------------------ public API
    def run(self, code, timeout=TIMEOUT):
        """Run one code block; returns the output text as the harness showed it to the model."""
        note = None
        if not self.alive():
            self.log("[kernel] sandbox is gone; creating a new one")
            self._create()
            note = NOTE_RESTART
        r = self._request({"code": code, "timeout": timeout}, exec_timeout=timeout + 240)
        if r is None:                      # the sandbox itself died during the run (or its lifetime ended)
            if not self.alive():
                self._create()
            r = {"output": "", "status": "died", "returncode": None, "secs": None, "peak_rss_gb": None}
        if r.get("status") == "error":     # supervisor-side failure: report it as the kernel dying
            self.log("[kernel] supervisor error: " + r.get("output", "")[-2000:])
            r = {"output": "", "status": "died", "returncode": None}
        if r.get("restarted_before"):
            note = NOTE_RESTART
        self.commit()
        self.history.append({"t": time.time(), "status": r.get("status"), "secs": r.get("secs"),
                             "peak_rss_gb": r.get("peak_rss_gb")})
        text = format_output(r.get("output", ""), r.get("status", "ok"), timeout, r.get("returncode"))
        if note:
            text = note + "\n" + text
        return text

    def _sh(self, cmd, timeout=1800):
        p = self.sb.exec("bash", "-c", cmd, timeout=timeout)
        out, err = p.stdout.read(), p.stderr.read()
        rc = p.wait()
        if rc:
            raise RuntimeError(f"{cmd!r} failed ({rc}): {err[-1000:]}")
        return out

    def commit(self):
        """Copy /work to the run's volume and commit it (the forward-pass server reads the current edit from it).
        /work is the sandbox's local disk; the volume (mounted at C.WORK_SYNC) mirrors it, so that the kernel's open
        files never block a volume reload and /work survives a new sandbox."""
        delete = "" if getattr(self, "pull_pending", False) else "--delete "
        for i in range(3):
            try:
                self._sh(f"rsync -a {delete}/work/ {C.WORK_SYNC}/ && sync {C.WORK_SYNC}")
                return True
            except Exception as e:  # noqa: BLE001
                self.log(f"[kernel] commit failed ({e!r}); retry {i + 1}")
                time.sleep(5)
        return False

    def reload(self):
        """Bring the files the forward-pass server wrote to the volume into /work."""
        self.pull_pending = True
        for i in range(10):
            try:
                self.sb.reload_volumes()
                self._sh(f"rsync -a {C.WORK_SYNC}/ /work/")
                self.pull_pending = False
                return True
            except Exception as e:  # noqa: BLE001
                self.log(f"[kernel] reload failed ({e!r}); retry {i + 1}")
                time.sleep(10)
        return False

    def restart(self):
        """Restart the kernel process (variables lost), e.g. when the original harness restarted."""
        return self._request({"op": "restart"}, exec_timeout=360)

    def ls_work(self):
        p = self.sb.exec("python", "-c", "import os,json; print(json.dumps(sorted(os.listdir('/work'))))")
        out = p.stdout.read()
        p.wait()
        return json.loads(out)

    def read_work_file(self, rel):
        with self.sb.open(f"/work/{rel}", "rb") as f:
            return f.read()

    def close(self):
        if self.sb is not None:
            self.commit()
            self.sb.terminate()
            self.sb = None
