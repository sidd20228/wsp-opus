"""Experiment 4 helpers: read the actor exactly as shown in the prompt (integers x one scale per tensor) and apply a
reply's edits to it."""
import re

import numpy as np

from wsp import edits

HEAD = re.compile(r"^([\w.]+) \(([\d x]+)\), scale ([0-9.eE+-]+):")
NAMES = ["embed.W", "embed.b"] + [f"actor{k}.{t}" for k in range(5) for t in ("W", "b")] + \
        ["actor_head.W", "actor_head.b"]


def shown_weights(prompt):
    lines = prompt.split("THE WEIGHTS (complete actor).", 1)[1].splitlines()
    out, i = {}, 0
    while i < len(lines):
        m = HEAD.match(lines[i])
        if not m:
            i += 1
            continue
        name, dims, scale = m.group(1), [int(x) for x in m.group(2).split(" x ")], float(m.group(3))
        i += 1
        if len(dims) == 2:
            rows = []
            for r in range(dims[0]):
                head, vals = lines[i + r].split(":", 1)
                assert int(head) == r, (name, r, head)
                rows.append([int(v) for v in vals.split()])
            a = np.array(rows, dtype=np.float64)
            i += dims[0]
        else:
            a = np.array([int(v) for v in lines[i].split()], dtype=np.float64)
            i += 1
        assert list(a.shape) == dims, (name, a.shape, dims)
        out[name] = a * scale
    assert set(out) == set(NAMES), sorted(out)
    return out


def apply_reply(prompt, reply):
    base = shown_weights(prompt)
    net = {k: v.copy() for k, v in base.items()}
    rep = edits.apply(reply, net)
    changed = {k: int(np.sum(net[k] != base[k])) for k in NAMES}
    total = sum(v.size for v in base.values())
    rep_changes = {"changed": sum(changed.values()), "entries": total,
                   "frac_changed": sum(changed.values()) / total, "per_tensor": changed}
    return net, rep, rep_changes
