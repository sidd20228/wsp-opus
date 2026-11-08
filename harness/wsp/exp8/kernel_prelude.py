# The editor's kernel prelude (Experiment 8): the names the first-turn prompt promises. Executed into the kernel's
# globals before the first code block (and again after every restart). The kernel's globals after it are those the
# original showed: json, os, np, torch, safe_open, Tokenizer, MODEL, WORK, NAMES, load_original, load, FROZEN,
# save_edit, drop_edit, edited, tok, Path.
import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from tokenizers import Tokenizer

MODEL = "/model"
WORK = "/work"
torch.set_grad_enabled(False)

with safe_open(os.path.join(MODEL, "model.safetensors"), framework="pt") as _f:
    NAMES = sorted(_f.keys())
FROZEN = {"model.language_model.norm.weight", "model.language_model.embed_tokens.weight", "lm_head.weight"}


def load_original(name):
    """The original weights of tensor `name` as a float32 torch tensor."""
    with safe_open(os.path.join(MODEL, "model.safetensors"), framework="pt") as f:
        return f.get_tensor(name).float()


def load(name):
    """Tensor `name` as a float32 torch tensor: your current edit if there is one, else the original."""
    p = Path(WORK) / "edited" / (name + ".npy")
    if p.exists():
        return torch.from_numpy(np.load(p)).float()
    return load_original(name)


def save_edit(name, tensor):
    """Save `tensor` (same shape as the original) as your edit of `name` (/work/edited/<name>.npy, float32; the
    evaluation casts it to bfloat16). Refused: the final norm, the embedding / output layer, and an edit that only
    multiplies the whole tensor by a constant."""
    if name in FROZEN:
        raise ValueError(f"save_edit refused: {name} is out of scope (the final norm and the input embedding / output "
                         "layer cannot be edited)")
    if name not in NAMES:
        raise KeyError(f"save_edit: unknown tensor name {name!r}")
    t = tensor.detach().cpu() if isinstance(tensor, torch.Tensor) else torch.as_tensor(np.asarray(tensor))
    t = t.to(torch.float32).contiguous()
    w0 = load_original(name)
    if tuple(t.shape) != tuple(w0.shape):
        raise ValueError(f"save_edit: shape {tuple(t.shape)} differs from the original {tuple(w0.shape)} of {name}")
    if not bool(torch.isfinite(t).all()):
        raise ValueError(f"save_edit: the tensor for {name} has non-finite values")
    a, b = t.double().reshape(-1), w0.double().reshape(-1)
    den = float(b @ b)
    if den > 0:
        c = float(a @ b) / den
        resid = float((a - c * b).norm()) / den ** 0.5
        if resid < 1e-6 and abs(c - 1.0) > 1e-6:
            raise ValueError(f"save_edit refused: this edit only multiplies the whole tensor {name} by a constant "
                             f"({c:.6g}); whole-tensor rescaling is out of scope")
    d = Path(WORK) / "edited"
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / (name + ".npy.tmp")
    with open(tmp, "wb") as f:
        np.save(f, t.numpy())
    os.replace(tmp, d / (name + ".npy"))


def drop_edit(name):
    """Remove your edit of `name` (it reverts to the original)."""
    p = Path(WORK) / "edited" / (name + ".npy")
    if p.exists():
        p.unlink()


def edited():
    """The names of the tensors you have edited."""
    d = Path(WORK) / "edited"
    return sorted(p.name[:-4] for p in d.glob("*.npy")) if d.exists() else []


tok = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json"))
del _f
