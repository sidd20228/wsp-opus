"""The editor's forward-pass tool (Experiment 8), on the H100 VM; called by the conversation driver on the Mac.

    from wsp.exp8.fwd import forward_pass, pass_message
    text = forward_pass(request_json, k, run_id)          # JSON text (indent 1); "ok": true or false
    ok = json.loads(text)["ok"]                           # false: the pass failed and must NOT be counted
    msg = pass_message(text, k, n_done=used + 1, budget=50)

k is the request number and names the files (/work/gen_<k>.jsonl, /work/gen_<k>_{original,edited}.jsonl,
/work/gen_<k>[_<model>]_acts/, /work/fwd_<k>/); the run's /work is ~/wsp-opus/work/exp8/<run>/work on the VM, the same
folder the kernel binds at /work, so the kernel sees the results at once. The summary has the original's fields
(summary.py). The GPU work runs in vm_gpu.py on the VM; it waits for the GPU lock (one GPU job at a time).
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
from summary import pass_message, validate  # noqa: E402,F401
import common as C  # noqa: E402

GPU_PY = f"{R.BASE}/venvs/exp8_gpu/bin/python"
VM_GPU = f"{R.BASE}/code/wsp/exp8/vm_gpu.py"


def forward_pass(request_json, k, run_id, kernel=None, timeout=3 * 3600):
    """Run one forward-pass request; returns the summary as JSON text. "ok": false = failed, not counted.
    (`kernel` is accepted for the Modal-era signature; nothing needs syncing on the VM.)"""
    t0 = time.time()
    text = request_json if isinstance(request_json, str) else json.dumps(request_json)
    try:
        validate(json.loads(text), os.path.join(C.LOGS, "practice"))
    except Exception as e:  # noqa: BLE001
        return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}", "s": round(time.time() - t0, 1)},
                          indent=1)
    p = R.sh(f"{GPU_PY} {VM_GPU} fwd {shlex.quote(run_id)} {int(k)} 2>> {R.BASE}/logs/exp8/fwd_{run_id}.log",
             input=text.encode(), timeout=timeout, check=False, retries=1)
    out = p.stdout.decode(errors="replace").strip()
    try:
        res = json.loads(out)
    except ValueError:
        res = {"ok": False, "error": f"RuntimeError: the forward-pass server did not answer (exit {p.returncode})",
               "s": round(time.time() - t0, 1)}
    return json.dumps(res, indent=1)
