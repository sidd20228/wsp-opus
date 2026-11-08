"""The model's Python sandbox on Modal (the analog of the original bubblewrap sandbox): no network, no secrets,
the experiment's package set, a time limit per run and an output cap, output formatted as the original harness did
(stdout, then "[stderr]" and its tail; "[timed out after N s]"; "[output cut at N characters]").

Two modes:
  run_fresh(code, ...)       a fresh process per run (Experiment 3)
  Session(...).run(code)     one long-lived sandbox per conversation whose /work folder persists between runs
                             (Experiments 6 and 8)
"""
import time

import modal

APP_NAME = "wsp-sandbox"
PKGS = ["numpy==1.26.4", "scipy==1.13.1", "scikit-learn==1.5.2", "pandas==2.2.3", "sympy==1.13.3",
        "networkx==3.3"]
_app = None


def app():
    global _app
    if _app is None:
        _app = modal.App.lookup(APP_NAME, create_if_missing=True)
    return _app


def base_image(extra_pip=()):
    return (modal.Image.debian_slim(python_version="3.10")
            .pip_install(*PKGS, *extra_pip)
            .pip_install("torch==2.4.1", index_url="https://download.pytorch.org/whl/cpu")
            .env({"OMP_NUM_THREADS": "8", "CUDA_VISIBLE_DEVICES": ""}))


def _format(stdout, stderr, timed_out, timeout, max_out):
    if timed_out:
        out = f"[timed out after {timeout} s]"
        if stdout.strip():
            out = stdout + "\n" + out
    else:
        out = stdout + (("\n[stderr]\n" + stderr[-6000:]) if stderr.strip() else "")
    if len(out) > max_out:
        out = out[:max_out] + f"\n[output cut at {max_out:,} characters]"
    return out or "(no output)"


def run_fresh(code, image, prelude="", timeout=600, max_out=20_000, cpu=8.0, memory=32768):
    """Run `prelude + code` in a fresh sandbox process. Returns (formatted output, seconds)."""
    t0 = time.time()
    sb = modal.Sandbox.create("python", "-I", "-c", prelude + "\n" + code, app=app(), image=image,
                              block_network=True, timeout=timeout + 120, cpu=cpu, memory=memory, workdir="/work")
    timed_out = False
    try:
        sb.wait(raise_on_termination=False)
        stdout, stderr = sb.stdout.read(), sb.stderr.read()
    finally:
        sb.terminate()
    secs = time.time() - t0
    if secs > timeout + 60 or sb.returncode in (-9, 137) and secs > timeout:
        timed_out = True
    return _format(stdout, stderr, timed_out, timeout, max_out), round(secs, 2)


class Session:
    """A long-lived sandbox for one conversation; /work persists between runs. Each run is a fresh Python process
    (`python -I -c prelude + code`) inside it, with its own time limit."""

    def __init__(self, image, lifetime_s=24 * 3600, cpu=8.0, memory=98304, volumes=None):
        self.image, self.cpu, self.memory, self.volumes = image, cpu, memory, volumes or {}
        self.sb = modal.Sandbox.create("sleep", "infinity", app=app(), image=image, block_network=True,
                                       timeout=lifetime_s, cpu=cpu, memory=memory, workdir="/work",
                                       volumes=self.volumes)

    def run(self, code, prelude="", timeout=900, max_out=20_000):
        t0 = time.time()
        p = self.sb.exec("timeout", "--signal=KILL", str(timeout), "python", "-I", "-c", prelude + "\n" + code,
                         workdir="/work")
        stdout, stderr = p.stdout.read(), p.stderr.read()
        p.wait()
        timed_out = p.returncode in (124, 137, -9) and time.time() - t0 >= timeout - 1
        return _format(stdout, stderr, timed_out, timeout, max_out), round(time.time() - t0, 2)

    def put(self, local_path, remote_path):
        self.sb.filesystem.copy_from_local(local_path, remote_path)

    def get(self, remote_path, local_path):
        self.sb.filesystem.copy_to_local(remote_path, local_path)

    def close(self):
        self.sb.terminate()
