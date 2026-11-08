"""The "edits" weight-writing language of Experiments 4 and 5, as the prompts define it.

  W1[12, 0] = 2.5                      one entry
  W1[0:6, 3] = [1, 0, -1, 0, 2, 0]     explicit values (as many values as entries)
  b1[0:4] = -1                         fill a range (a:b half-open, a:b:s strided, ':' the whole axis)
  W2[0:2, 0:3] = [[1, -1, 0.5], [0, 2, 0]]
  W3[:, 2] *= 2                        scale (also += and -=)
  ZERO actor4.W                        set a whole tensor to 0 (Exp 4)

'#' starts a comment; a long list may continue over the next lines. Statements apply in order. Only the LAST edits
block of a reply is read. A statement that cannot be read is skipped (the others still apply). Values: plain decimal
numbers with |v| <= 1e6. Both "[[" and "[ [" list forms are accepted (the original reader was fixed to accept the
latter before the final reading; exp5 README).
"""
import ast
import re

import numpy as np

BLOCK = re.compile(r"```edits[^\n]*\n(.*?)(```|\Z)", re.S)
STMT = re.compile(r"^([A-Za-z_][\w.]*)\s*\[(.*?)\]\s*(=|\+=|-=|\*=)\s*(.+)$", re.S)
VMAX = 1e6


def last_block(reply):
    blocks = BLOCK.findall(reply)
    if not blocks:
        return None, False
    body, end = blocks[-1]
    return body, end != "```"


def _strip_comment(line):
    i = line.find("#")
    return line if i < 0 else line[:i]


def statements(body):
    out, buf, depth = [], "", 0
    for raw in body.splitlines():
        line = _strip_comment(raw).strip()
        if not line and not buf:
            continue
        buf = (buf + " " + line).strip()
        depth = buf.count("[") - buf.count("]")
        if depth <= 0 or buf.upper().startswith("ZERO "):
            out.append(buf)
            buf = ""
    if buf:
        out.append(buf)
    return out


def _index(spec, ndim):
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != ndim:
        raise ValueError(f"{len(parts)} indices for a {ndim}-d tensor")
    idx = []
    for p in parts:
        if ":" in p:
            bits = p.split(":")
            if len(bits) > 3:
                raise ValueError(f"bad slice {p!r}")
            nums = [int(b) if b.strip() else None for b in bits]
            idx.append(slice(*nums))
        else:
            idx.append(int(p))
    return tuple(idx)


def apply(reply, tensors):
    """Apply the last edits block of `reply` to `tensors` (dict name -> float array, modified in place).
    Returns the report (block, statements, applied, entries_written, errors, cut_off, tensors_touched)."""
    body, cut = last_block(reply)
    rep = {"block": body is not None, "statements": 0, "applied": 0, "entries_written": 0, "errors": [],
           "cut_off": cut, "tensors_touched": []}
    if body is None:
        return rep
    touched = []
    for s in statements(body):
        rep["statements"] += 1
        try:
            if s.upper().startswith("ZERO "):
                name = s[5:].strip()
                if name not in tensors:
                    raise KeyError(f"unknown tensor {name}")
                tensors[name][...] = 0
                n = tensors[name].size
            else:
                m = STMT.match(s)
                if not m:
                    raise ValueError("cannot read statement")
                name, ispec, op, vtext = m.groups()
                if name not in tensors:
                    raise KeyError(f"unknown tensor {name}")
                t = tensors[name]
                idx = _index(ispec, t.ndim)
                sel_shape = t[idx].shape
                v = ast.literal_eval(vtext.strip())
                v = np.asarray(v, dtype=np.float64)
                if not np.all(np.isfinite(v)) or np.any(np.abs(v) > VMAX):
                    raise ValueError("value out of range")
                if v.ndim > 0 and v.shape != sel_shape:
                    if v.size == int(np.prod(sel_shape)):
                        v = v.reshape(sel_shape)
                    else:
                        raise ValueError(f"{v.size} values for {int(np.prod(sel_shape))} entries")
                cur = t[idx]
                new = {"=": lambda: np.broadcast_to(v, sel_shape), "+=": lambda: cur + v,
                       "-=": lambda: cur - v, "*=": lambda: cur * v}[op]()
                t[idx] = new
                n = int(np.prod(sel_shape)) if sel_shape else 1
            rep["applied"] += 1
            rep["entries_written"] += n
            if name not in touched:
                touched.append(name)
        except Exception as e:  # a statement that cannot be read is skipped
            rep["errors"].append({"statement": s[:200], "error": str(e)[:200]})
    rep["tensors_touched"] = touched
    return rep
