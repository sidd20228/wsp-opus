The last run (106) ranked other MLP neurons in layers 16–23 by how much their activation tracks the doubt decisions in texts A and B, along with their logit-lens effect on doubt words versus commit words. A few look like additional doubt-related units, mainly L16 N998, L20 N405/N1111 and L23 N168. I won't stack more edits yet. The only edit with a large measured effect is the brake (L22 N573 down column ×2: +32.0 ± 2.2 points paired on D1). Its strength was picked a priori, it has no dose-response, and it hasn't been confirmed on a fresh split.

The next priorities are:
1. A dose-response for the brake, paired on the same D1 items and seeds as gen_2 (original) and gen_11 (×2).
2. A fresh-split confirmation.
3. A second mechanism for the remaining restatement loops.

Since the kernel was carried over, this block checks its state first. It verifies that /work/edited really is only column 573 ×2. It also rebuilds compact per-sample tables (finished, correct, length, loop onset) for gen_2, gen_11 and gen_13, which every paired comparison below will use.

```python
# Hypothesis/check: kernel state intact; /work/edited == L22 down col 573 x2 only; build per-sample tables for gen_2/11/13 for paired comparisons
import json, os, numpy as np, torch, re, collections
uv = [k for k,v in globals().items() if not k.startswith('_') and not callable(v) and type(v).__name__ not in ('module',)]
print("n user vars:", len(uv)); print(sorted(uv)[:120])
W0 = load_original('model.language_model.layers.22.mlp.down_proj.weight'); W1 = load('model.language_model.layers.22.mlp.down_proj.weight')
d = (W1-W0); cols = torch.nonzero(d.abs().sum(0)>0).flatten().tolist()
print("changed cols:", cols, " ratio:", float((W1[:,573]/W0[:,573]).median()))
def readgen(p):
    rows=[json.loads(l) for l in open(p)]
    return rows
g2=readgen('/work/gen_2.jsonl'); g11=readgen('/work/gen_11.jsonl'); g13=readgen('/work/gen_13.jsonl')
print(len(g2),len(g11),len(g13)); print("keys:", list(g11[0].keys()))
```