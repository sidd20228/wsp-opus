"""The editor's persistent Python kernel, inside the Modal sandbox (no network). Three roles:

  python kernel_server.py supervisor   the sandbox's main process: owns the kernel process, serves requests on
                                       /tmp/kernel.sock, enforces the time limit (default 900 s) and the memory limit
                                       (96 GiB resident), restarts the kernel when it dies
  python kernel_server.py child <fd>   the kernel: one Python process whose globals persist between code blocks
                                       (prelude first); stdout and stderr of a block go to a file (fd level, so C-level
                                       output and warnings are captured too)
  python kernel_server.py client       one request: code on stdin, the JSON result on stdout

A block runs like a notebook cell: compiled with filename "<run>", and if its last statement is an expression its
value is shown (repr, unless None). Exceptions print their traceback (frames of this file are hidden).
"""
import ast
import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback

SOCK = os.environ.get("KERNEL_SOCK", "/tmp/kernel.sock")      # VM: a folder bound into the sandbox
WORK_SYNC = "/mnt/work_sync"
PRELUDE = os.environ.get("KERNEL_PRELUDE", "/opt/kernel/kernel_prelude.py")
MAX_RSS = 96 * 1024 ** 3
RAW_CAP = 4_000_000        # characters of raw output kept (the harness shows at most 10,000 after filtering)


# ------------------------------------------------------------------------------------------------------- child
def _send(sock, obj):
    sock.sendall((json.dumps(obj) + "\n").encode())


def _recv(sock, buf):
    while b"\n" not in buf[0]:
        chunk = sock.recv(1 << 20)
        if not chunk:
            raise EOFError
        buf[0] += chunk
    line, buf[0] = buf[0].split(b"\n", 1)
    return json.loads(line)


def _run_cell(code, g):
    tree = ast.parse(code, filename="<run>", mode="exec")
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
    exec(compile(tree, "<run>", "exec"), g)
    if last is not None:
        v = eval(compile(last, "<run>", "eval"), g)
        if v is not None:
            print(repr(v))


def _print_exc():
    et, ev, tb = sys.exc_info()
    while tb is not None and tb.tb_frame.f_code.co_filename == __file__:     # hide the kernel's own frames
        tb = tb.tb_next
    sys.stderr.write("".join(traceback.format_exception(et, ev, tb)))


def child(fd):
    sock = socket.socket(fileno=int(fd))
    try:
        with open(f"/proc/{os.getpid()}/oom_score_adj", "w") as f:       # the kernel, not the supervisor, is OOM-killed
            f.write("1000")
    except OSError:
        pass
    g = {"__name__": "__main__", "__builtins__": __builtins__}
    exec(compile(open(PRELUDE).read(), "<prelude>", "exec"), g)
    _send(sock, {"ready": True})
    buf = [b""]
    while True:
        try:
            req = _recv(sock, buf)
        except EOFError:
            return
        sys.stdout.flush()
        sys.stderr.flush()
        out = os.open(req["out"], os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        save1, save2 = os.dup(1), os.dup(2)
        os.dup2(out, 1)
        os.dup2(out, 2)
        os.close(out)
        try:
            _run_cell(req["code"], g)
        except SystemExit:
            _print_exc()
        except BaseException:  # noqa: BLE001
            _print_exc()
        finally:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:  # noqa: BLE001
                pass
            os.dup2(save1, 1)
            os.dup2(save2, 2)
            os.close(save1)
            os.close(save2)
        _send(sock, {"done": True})


# -------------------------------------------------------------------------------------------------- supervisor
class Supervisor:
    def __init__(self):
        self.proc = None
        self.sock = None
        self.buf = [b""]
        self.n = 0
        self.starts = 0

    def start_child(self):
        a, b = socket.socketpair()
        self.proc = subprocess.Popen([sys.executable, "-u", __file__, "child", str(b.fileno())], pass_fds=[b.fileno()],
                                     cwd="/work", stdin=subprocess.DEVNULL)
        b.close()
        self.sock, self.buf = a, [b""]
        self.starts += 1
        self.sock.settimeout(300)
        try:
            msg = _recv(self.sock, self.buf)
            assert msg.get("ready")
        except Exception as e:  # noqa: BLE001
            self.kill()
            raise RuntimeError(f"the kernel did not start: {e!r}")
        self.sock.settimeout(None)

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def kill(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGKILL)
            self.proc.wait()
        if self.sock is not None:
            self.sock.close()
        self.proc, self.sock = None, None

    def rss(self):
        try:
            for line in open(f"/proc/{self.proc.pid}/status"):
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        except OSError:
            return 0
        return 0

    def execute(self, code, timeout):
        restarted = False
        if not self.alive():
            if self.starts:
                restarted = True
            self.kill()
            self.start_child()
        self.n += 1
        out = f"/tmp/kout_{self.n}.txt"
        t0 = time.time()
        _send(self.sock, {"code": code, "out": out})
        self.sock.settimeout(0.2)
        status, peak = "ok", 0
        while True:
            try:
                msg = _recv(self.sock, self.buf)
                if msg.get("done"):
                    break
            except socket.timeout:
                pass
            except (EOFError, OSError):
                status = "died"
                break
            if not self.alive():
                status = "died"
                break
            r = self.rss()
            peak = max(peak, r)
            if r > MAX_RSS:
                status = "memory"
                break
            if time.time() - t0 > timeout:
                status = "timeout"
                break
        secs = time.time() - t0
        code_rc = None
        if status != "ok":
            if status == "died":
                time.sleep(0.5)
                code_rc = self.proc.poll() if self.proc is not None else None
            self.kill()
            try:                       # a fresh kernel right away (the output of this run says it was restarted)
                self.start_child()
            except Exception:  # noqa: BLE001
                pass
        else:
            self.sock.settimeout(None)
        try:
            with open(out, "r", errors="replace") as f:
                raw = f.read(RAW_CAP + 1)
            os.unlink(out)
        except OSError:
            raw = ""
        return {"output": raw, "status": status, "secs": round(secs, 2), "returncode": code_rc,
                "restarted_before": restarted, "peak_rss_gb": round(peak / 1024 ** 3, 2),
                "raw_truncated": len(raw) > RAW_CAP}


def supervisor():
    os.makedirs("/work", exist_ok=True)
    if os.path.isdir(WORK_SYNC):           # a new sandbox of a running run: restore /work from the volume
        subprocess.run(["rsync", "-a", WORK_SYNC + "/", "/work/"], check=False)
    sup = Supervisor()
    sup.start_child()
    if os.path.exists(SOCK):
        os.unlink(SOCK)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK)
    srv.listen(4)
    print("kernel supervisor ready", flush=True)
    while True:
        conn, _ = srv.accept()
        try:
            buf = [b""]
            req = _recv(conn, buf)
            if req.get("op") == "restart":
                sup.kill()
                sup.start_child()
                res = {"status": "restarted"}
            elif req.get("op") == "ping":
                res = {"status": "ok", "alive": sup.alive(), "starts": sup.starts, "runs": sup.n}
            else:
                res = sup.execute(req["code"], float(req.get("timeout", 900)))
            _send(conn, res)
        except Exception:  # noqa: BLE001
            try:
                _send(conn, {"status": "error", "output": traceback.format_exc()})
            except OSError:
                pass
        finally:
            conn.close()


def client():
    req = json.loads(sys.stdin.read())
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    tries = int(os.environ.get("KERNEL_CONNECT_TRIES", "600"))
    for i in range(tries):
        try:
            s.connect(SOCK)
            break
        except OSError:
            if i == tries - 1:
                sys.stdout.write(json.dumps({"status": "nokernel"}))
                return
            time.sleep(0.5)
    _send(s, req)
    buf = [b""]
    res = _recv(s, buf)
    sys.stdout.write(json.dumps(res))
    sys.stdout.flush()


if __name__ == "__main__":
    role = sys.argv[1]
    if role == "supervisor":
        supervisor()
    elif role == "child":
        child(sys.argv[2])
    else:
        client()
