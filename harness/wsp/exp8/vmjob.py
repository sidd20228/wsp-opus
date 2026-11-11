"""Long VM jobs that survive network blips: the command runs detached on the VM (setsid nohup), writes its stdout to
a result file (renamed into place when the command exits), and the Mac polls for that file with short ssh calls that
are retried. A job is identified by its result path, so a caller that restarts can wait for the same job again."""
import os
import shlex
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from wsp import remote as R  # noqa: E402

JOBS = f"{R.BASE}/data/exp8/jobs"


def _sh(cmd, input=None, timeout=120, tries=30):
    """ssh with patience: retried for ~15 minutes of network trouble."""
    for i in range(tries):
        try:
            p = R.sh(cmd, input=input, timeout=timeout, check=False, retries=1)
            if p.returncode != 255:
                return p
        except Exception:  # noqa: BLE001
            pass
        time.sleep(min(30, 5 * (i + 1)))
    raise RuntimeError(f"the VM did not answer: {cmd[:120]}")


def start(job_id, cmd, input_text=None, log=None):
    """start `cmd` detached; its stdout goes to JOBS/<job_id>.out (complete when JOBS/<job_id>.done exists)."""
    q = shlex.quote
    base = f"{JOBS}/{job_id}"
    inp = f"{base}.in" if input_text is not None else "/dev/null"
    log = log or f"{base}.log"
    script = [f"mkdir -p {q(JOBS)}", f"rm -f {q(base)}.out {q(base)}.done"]
    if input_text is not None:
        script.append(f"cat > {q(inp)}")
    script.append(f"setsid nohup bash -c {q(cmd + f' < {inp} > {base}.out.tmp 2>> {log}; echo $? > {base}.rc; mv {base}.out.tmp {base}.out; touch {base}.done')} "
                  f"> /dev/null 2>&1 < /dev/null &")
    _sh("\n".join(script), input=None if input_text is None else input_text.encode())
    return base


def wait(base, timeout, poll=20, first_poll=None):
    """the job's stdout once it is done (None after `timeout` seconds)."""
    t0 = time.time()
    q = shlex.quote(base)
    delay = first_poll if first_poll is not None else poll
    while time.time() - t0 < timeout:
        out = _sh(f"if test -e {q}.done; then echo __DONE__; cat {q}.out; fi").stdout.decode(errors="replace")
        if out.startswith("__DONE__"):
            return out[len("__DONE__\n"):]
        time.sleep(delay)
        delay = min(poll, delay * 2)
    return None


def run(job_id, cmd, input_text=None, timeout=24 * 3600, log=None, poll=20, first_poll=None):
    return wait(start(job_id, cmd, input_text, log), timeout, poll, first_poll)
