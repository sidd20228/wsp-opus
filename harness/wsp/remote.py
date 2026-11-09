"""The H100 VM backend (replaces Modal for the sandboxes and the CPU/GPU tools).

The drivers keep running on this machine (Opus is called through the local Claude Code CLI); code the model writes
runs on the VM inside bubblewrap, exactly as the original harness/sandbox.py did: --unshare-all (no network),
--clearenv, /usr read-only, the sandbox venv read-only, /work bound read-write, a time limit (timeout -s KILL) and a
memory limit (prlimit --as, the original's RLIMIT_AS).

  VM layout (BASE = ~/wsp-opus): python/ (uv's CPython builds), venvs/<name>/, data/, releases/, work/<exp>/<conv>/
"""
import os
import shlex
import subprocess
import time
import uuid

HOST = os.environ.get("WSP_VM", "<vm-user>@<vm-host>")
PORT = os.environ.get("WSP_VM_PORT", "1024")
BASE = "<vm-home>/wsp-opus"
CM = os.path.expanduser("~/.ssh/cm/%r@%h:%p")
SSH_OPTS = ["-o", "ControlMaster=auto", "-o", f"ControlPath={CM}", "-o", "ControlPersist=8h",
            "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6", "-o", "BatchMode=yes", "-p", PORT]
SSH = ["ssh", *SSH_OPTS, HOST]
RSYNC_E = "ssh " + " ".join(shlex.quote(o) for o in SSH_OPTS)


def sh(cmd, input=None, timeout=None, check=True, retries=3):
    """Run a shell command on the VM. Returns CompletedProcess (bytes). Retries on ssh transport errors (255)."""
    for attempt in range(retries):
        p = subprocess.run(SSH + [cmd], input=input, capture_output=True, timeout=timeout)
        if p.returncode != 255:
            break
        time.sleep(5 * (attempt + 1))
    if check and p.returncode != 0:
        raise RuntimeError(f"remote command failed ({p.returncode}): {cmd[:200]}\n{p.stderr.decode()[-2000:]}")
    return p


def put_text(text, remote_path):
    sh(f"mkdir -p {shlex.quote(os.path.dirname(remote_path))} && cat > {shlex.quote(remote_path)}",
       input=text.encode())


def rsync_to(local, remote, delete=False):
    """Copy a local file or folder to the VM (folder: contents of local/ into remote/)."""
    src = local.rstrip("/") + ("/" if os.path.isdir(local) else "")
    sh(f"mkdir -p {shlex.quote(remote if os.path.isdir(local) else os.path.dirname(remote))}")
    cmd = ["rsync", "-a", "-e", RSYNC_E] + (["--delete"] if delete else []) + [src, f"{HOST}:{remote}"]
    subprocess.run(cmd, check=True, capture_output=True)


def rsync_from(remote, local, delete=False, excludes=()):
    os.makedirs(local, exist_ok=True)
    cmd = ["rsync", "-a", "-e", RSYNC_E] + (["--delete"] if delete else [])
    for e in excludes:
        cmd += ["--exclude", e]
    cmd += [f"{HOST}:{remote.rstrip('/')}/", local.rstrip("/") + "/"]
    subprocess.run(cmd, check=True, capture_output=True)


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


def sandbox(argv, venv, work, ro_binds=(), timeout=600, max_out=20_000, mem_gb=96, files=None, threads=8):
    """Run `argv` (python args after the interpreter) in bubblewrap on the VM.

    venv: name under venvs/; work: remote folder bound at /work (created if missing); ro_binds: [(src, dst)];
    files: {sandbox path under /tmp/in: text} written to a private folder and bound read-only at /tmp/in.
    Returns (formatted output, seconds)."""
    vp = f"{BASE}/venvs/{venv}"
    tag = uuid.uuid4().hex[:12]
    inbox = f"{BASE}/tmp/{tag}"
    script = [f"mkdir -p {shlex.quote(work)} {inbox}"]
    for name, text in (files or {}).items():
        script.append(f"cat > {inbox}/{name} <<'__WSP_EOF_{tag}__'\n{text}\n__WSP_EOF_{tag}__")
    binds = ["--ro-bind", f"{BASE}/python", f"{BASE}/python", "--ro-bind", vp, vp]
    for src, dst in ro_binds:
        binds += ["--ro-bind", src, dst]
    if files:
        binds += ["--ro-bind", inbox, "/tmp/in"]
    cmd = (["timeout", "--signal=KILL", str(timeout), "prlimit", f"--as={mem_gb << 30}",
            "bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
            "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
            "--symlink", "usr/bin", "/bin", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"] + binds +
           ["--bind", work, "/work", "--chdir", "/work", "--setenv", "HOME", "/tmp", "--setenv", "PATH", "/usr/bin",
            "--setenv", "OMP_NUM_THREADS", str(threads), "--setenv", "MKL_NUM_THREADS", str(threads),
            "--setenv", "OPENBLAS_NUM_THREADS", str(threads), "--setenv", "CUDA_VISIBLE_DEVICES", "",
            f"{vp}/bin/python"] + list(argv))
    script.append(" ".join(shlex.quote(c) for c in cmd) + "; rc=$?")
    script.append(f"rm -rf {inbox}; exit $rc")
    t0 = time.time()
    p = sh("\n".join(script), check=False, timeout=timeout + 300)
    secs = time.time() - t0
    timed_out = p.returncode in (124, 137) and secs >= timeout - 1
    return _format(p.stdout.decode(errors="replace"), p.stderr.decode(errors="replace"), timed_out, timeout,
                   max_out), round(secs, 2)
